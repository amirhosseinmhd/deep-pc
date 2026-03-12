#!/usr/bin/env python
"""Test hypothesis: resnet gets nonzero weight updates at ALL layers because
the SEPARATE skip_model computation introduces float32 rounding noise.

In resnet, the energy function computes:
  err = z_l - scaling * model_l(z_{l-1}) - skip_l(z_{l-1})

The ffwd init computes:
  z_l = scaling * model_l(z_{l-1}) + skip_l(z_{l-1})

These differ in arithmetic ORDER:
  init:   z = (A + B)        where A = scaling*model(z), B = skip(z)
  energy: err = z - A - B = (z - A) - B

In float32: (A + B) - A - B ≠ 0 due to rounding.

In dyt_v2, skip is INSIDE the layer:
  model_l(z) = DyT(W * act(z)) + z
  err = z_l - model_l(z_{l-1})

Since z_l = model_l(z_{l-1}) was stored, err = 0 EXACTLY.
No rounding noise → no seed for Adam → deep layers frozen.

TEST:
1. Verify resnet energy noise scales with activity magnitude
2. Show that Adam amplifies ~1e-14 noise into ~1e-3 weight updates
3. Show dyt_v2 with external skip would fix the problem
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import jpc
import optax

from common.data import set_seed, get_mnist_loaders
from variants import get_variant

DEPTH = 50
WIDTH = 128
SEED = 42
BATCH_SIZE = 128


def test_float32_noise():
    """Show that separate skip computation creates float32 noise."""
    print("=" * 70)
    print("TEST 1: Float32 rounding noise from separate skip computation")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)

    # Create resnet model
    variant = get_variant("resnet")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu")

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, _ = variant.init_activities(model, img_batch)
    model_layers = model["model"]
    skip_model = model["skip_model"]

    print("\n  Simulating energy error computation manually:")
    print(f"  {'Layer':<8}{'||err||':<16}{'activity_norm':<16}{'err/act_norm':<16}")

    # Layer 0: z_0 vs model[0](x)
    z_recomp = vmap(model_layers[0])(img_batch)
    err = activities[0] - z_recomp
    err_norm = float(jnp.linalg.norm(err))
    act_norm = float(jnp.linalg.norm(activities[0]))
    print(f"  {0:<8}{err_norm:<16.6e}{act_norm:<16.4f}{err_norm/act_norm:<16.6e}")

    # Hidden layers: z_l vs model[l](z_{l-1}) + skip[l](z_{l-1})
    for l in range(1, len(model_layers) - 1):
        model_out = vmap(model_layers[l])(activities[l-1])
        skip_out = vmap(skip_model[l])(activities[l-1])

        # Method 1: as in energy function: z_l - model_out - skip_out
        err_energy = activities[l] - model_out - skip_out

        # Method 2: as in init: z_l was (model_out + skip_out), so err should be 0
        recomp = model_out + skip_out
        err_recomp = activities[l] - recomp

        err_norm = float(jnp.linalg.norm(err_energy))
        err_recomp_norm = float(jnp.linalg.norm(err_recomp))
        act_norm = float(jnp.linalg.norm(activities[l]))

        if l < 5 or l > DEPTH - 6 or l == DEPTH // 2:
            print(f"  {l:<8}{err_norm:<16.6e}{act_norm:<16.4f}"
                  f"{err_norm/act_norm:<16.6e}  recomp_err={err_recomp_norm:.2e}")


def test_adam_amplification():
    """Show that Adam amplifies tiny gradients via division by epsilon."""
    print("\n" + "=" * 70)
    print("TEST 2: Adam amplification of near-zero gradients")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)

    # resnet
    variant = get_variant("resnet")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu")
    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, _ = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(model)

    # Get parameter gradients (WITHOUT any inference)
    grads = jpc.compute_pc_param_grads(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type="sp",
    )

    model_grads = grads[0]  # gradients for model layers
    print("\n  Per-layer weight gradient norms (at ffwd init, NO inference):")
    show_idxs = [0, 1, 2, 5, 10, 20, 30, 40, 48, 49]
    for i in show_idxs:
        if i < len(model_grads):
            layer = model_grads[i]
            # Extract weight gradient
            linear = layer.layers[1]
            if hasattr(linear, 'weight') and linear.weight is not None:
                g = linear.weight
                gnorm = float(jnp.linalg.norm(g))
                gmax = float(jnp.max(jnp.abs(g)))
                print(f"    Layer {i}: ||dE/dW||={gnorm:.6e}, max|dE/dW|={gmax:.6e}")

    # Now do the same with SGD instead of Adam - should give ~0 updates
    print("\n  Weight updates with SGD (lr=1e-3) vs Adam (lr=1e-3):")

    # SGD update
    sgd_optim = optax.sgd(1e-3)
    sgd_state = sgd_optim.init(variant.get_optimizer_target(model))
    old_weights = [jnp.array(w) for w in variant.get_weight_arrays(model)]

    result_sgd = jpc.update_pc_params(
        params=params, activities=activities,
        optim=sgd_optim, opt_state=sgd_state,
        output=label_batch, input=img_batch,
        param_type="sp",
    )
    model_sgd = variant.post_learning_step(model, result_sgd, None)
    sgd_weights = [jnp.array(w) for w in variant.get_weight_arrays(model_sgd)]

    # Adam update
    adam_optim = optax.adam(1e-3)
    adam_state = adam_optim.init(variant.get_optimizer_target(model))
    result_adam = jpc.update_pc_params(
        params=params, activities=activities,
        optim=adam_optim, opt_state=adam_state,
        output=label_batch, input=img_batch,
        param_type="sp",
    )
    model_adam = variant.post_learning_step(model, result_adam, None)
    adam_weights = [jnp.array(w) for w in variant.get_weight_arrays(model_adam)]

    print(f"  {'Layer':<8}{'SGD ||dW||':<16}{'Adam ||dW||':<16}{'ratio':<12}")
    for i in show_idxs:
        if i < len(old_weights):
            sgd_wu = float(jnp.linalg.norm(sgd_weights[i] - old_weights[i]))
            adam_wu = float(jnp.linalg.norm(adam_weights[i] - old_weights[i]))
            ratio = adam_wu / max(sgd_wu, 1e-30)
            print(f"  {i:<8}{sgd_wu:<16.6e}{adam_wu:<16.6e}{ratio:<12.2e}")


def test_dyt_v2_with_sgd():
    """Show that dyt_v2 gets exactly zero updates (not even float32 noise)."""
    print("\n" + "=" * 70)
    print("TEST 3: dyt_v2 weight gradients at ffwd init (no inference)")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant("dyt_v2")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu",
                                  init_alpha=0.5)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, _ = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(model)

    grads = jpc.compute_pc_param_grads(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type="sp",
    )

    model_grads = grads[0]
    print("\n  Per-layer weight gradient norms (dyt_v2, ffwd init, NO inference):")
    show_idxs = [0, 1, 2, 5, 10, 20, 30, 40, 48, 49]
    for i in show_idxs:
        if i < len(model_grads):
            layer = model_grads[i]
            # dyt_v2 layers have .linear attribute
            if hasattr(layer, 'linear') and hasattr(layer.linear, 'weight'):
                g = layer.linear.weight
                if g is not None:
                    gnorm = float(jnp.linalg.norm(g))
                    print(f"    Layer {i}: ||dE/dW||={gnorm:.6e}")
                else:
                    print(f"    Layer {i}: gradient is None")
            elif hasattr(layer, 'layers'):
                # might be sequential
                lin = layer.layers[1] if len(layer.layers) > 1 else None
                if lin and hasattr(lin, 'weight') and lin.weight is not None:
                    g = lin.weight
                    gnorm = float(jnp.linalg.norm(g))
                    print(f"    Layer {i}: ||dE/dW||={gnorm:.6e}")
            else:
                print(f"    Layer {i}: (skipping, structure unknown)")


if __name__ == "__main__":
    test_float32_noise()
    test_adam_amplification()
    test_dyt_v2_with_sgd()
