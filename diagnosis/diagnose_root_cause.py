#!/usr/bin/env python
"""Pinpoint the exact root cause of dyt_v2 zero updates.

Hypothesis tests:
H1: The DyT (tanh squashing) kills the signal → test with use_dyt=False
H2: Internal vs external skip matters → test dyt_v2 layer WITHOUT skip
    vs resnet layer WITH skip, both fed same inputs
H3: It's about how jpc.pc_energy_fn handles skip_model=None vs not None
    → manually compute weight gradients both ways and compare
H4: Adam vs SGD → verify if resnet early layer updates are real or noise
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import jpc
import optax
import equinox as eqx

from common.data import set_seed, get_mnist_loaders
from variants import get_variant

DEPTH = 10
WIDTH = 128
SEED = 42
BATCH_SIZE = 128


def test_h1_dyt_disabled():
    """H1: Is DyT squashing the problem? Test with use_dyt=False."""
    print("=" * 70)
    print("H1: dyt_v2 with use_dyt=False (no tanh, just W*relu(z)+z)")
    print("    If still broken → DyT is NOT the cause")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)

    from variants.dyt_v2 import FCResNetDyT_v2
    from config import INPUT_DIM, OUTPUT_DIM

    # Create model with DyT disabled
    model = FCResNetDyT_v2(
        key=key, in_dim=INPUT_DIM, width=WIDTH, depth=DEPTH,
        out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5,
        dyt_enabled_layers=[]  # disable DyT at ALL layers
    )

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    # ffwd init (same as dyt_v2 variant)
    activities = []
    h = img_batch
    for layer in model.layers:
        h = vmap(layer)(h)
        activities.append(h)

    params = (model.layers, None)

    # Per-layer energy
    layer_e = jpc.pc_energy_fn(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type="sp", record_layers=True,
    )

    # Weight gradients
    grads = jpc.compute_pc_param_grads(
        params=params, activities=activities,
        y=label_batch, x=img_batch, param_type="sp",
    )

    print("\n  Per-layer energy and weight gradient norms:")
    model_grads = grads[0]
    for i in range(DEPTH):
        e = float(layer_e[i])
        g = model_grads[i]
        gnorm = 0.0
        if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
            gnorm = float(jnp.linalg.norm(g.linear.weight))
        print(f"    Layer {i}: energy={e:.6e}, ||dE/dW||={gnorm:.6e}")


def test_h2_external_skip():
    """H2: Same layer but with skip handled externally by jpc."""
    print("\n" + "=" * 70)
    print("H2: dyt_v2-like layer but with EXTERNAL skip (like resnet)")
    print("    Create layers WITHOUT internal skip, pass skip_model to jpc")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)

    from variants.dyt_v2 import FCResNetDyT_v2, HiddenLayerDyT_v2, DyTLayer
    from variants.dyt import InputLayer, OutputLayer
    from config import INPUT_DIM, OUTPUT_DIM
    import equinox.nn as nn

    # Create a normal dyt_v2 model
    model_orig = FCResNetDyT_v2(
        key=key, in_dim=INPUT_DIM, width=WIDTH, depth=DEPTH,
        out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5,
    )

    # Now create equivalent layers WITHOUT internal skip
    class HiddenNoSkip(eqx.Module):
        """Hidden layer: act_fn -> DyT(linear(x)), NO skip."""
        act_fn: object = eqx.field(static=True)
        dyt: DyTLayer
        linear: nn.Linear

        def __call__(self, x):
            h = self.act_fn(x)
            return self.dyt(self.linear(h))  # NO + x

    class OutputNoSkip(eqx.Module):
        """Output layer without skip."""
        act_fn: object = eqx.field(static=True)
        dyt: DyTLayer
        linear: nn.Linear

        def __call__(self, x):
            h = self.act_fn(x)
            h = self.dyt(h)
            return self.linear(h)

    # Build layers without skip, reusing same weights
    layers_no_skip = []
    layers_no_skip.append(model_orig.layers[0])  # input layer (no skip)

    for i in range(1, DEPTH - 1):
        orig_layer = model_orig.layers[i]
        layers_no_skip.append(HiddenNoSkip(
            act_fn=orig_layer.act_fn,
            dyt=orig_layer.dyt,
            linear=orig_layer.linear,
        ))

    # output layer (no skip in original either)
    layers_no_skip.append(model_orig.layers[-1])

    # Create skip model (identity at hidden layers, like jpc.make_skip_model)
    skip_model = jpc.make_skip_model(DEPTH)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    # ffwd init WITH skip model
    activities = jpc.init_activities_with_ffwd(
        model=layers_no_skip, input=img_batch,
        skip_model=skip_model, param_type="sp",
    )

    params = (layers_no_skip, skip_model)

    # Per-layer energy
    layer_e = jpc.pc_energy_fn(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type="sp", record_layers=True,
    )

    # Weight gradients
    grads = jpc.compute_pc_param_grads(
        params=params, activities=activities,
        y=label_batch, x=img_batch, param_type="sp",
    )

    print("\n  Per-layer energy and weight gradient norms (EXTERNAL skip):")
    model_grads = grads[0]
    for i in range(DEPTH):
        e = float(layer_e[i])
        g = model_grads[i]
        gnorm = 0.0
        if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
            gnorm = float(jnp.linalg.norm(g.linear.weight))
        elif hasattr(g, 'layers'):
            lin = g.layers[1] if len(g.layers) > 1 else None
            if lin and hasattr(lin, 'weight') and lin.weight is not None:
                gnorm = float(jnp.linalg.norm(lin.weight))
        print(f"    Layer {i}: energy={e:.6e}, ||dE/dW||={gnorm:.6e}")

    # Compare: original dyt_v2 (internal skip)
    print("\n  ORIGINAL dyt_v2 (internal skip) for comparison:")
    activities_orig = []
    h = img_batch
    for layer in model_orig.layers:
        h = vmap(layer)(h)
        activities_orig.append(h)

    params_orig = (model_orig.layers, None)
    layer_e_orig = jpc.pc_energy_fn(
        params=params_orig, activities=activities_orig,
        y=label_batch, x=img_batch,
        param_type="sp", record_layers=True,
    )
    grads_orig = jpc.compute_pc_param_grads(
        params=params_orig, activities=activities_orig,
        y=label_batch, x=img_batch, param_type="sp",
    )
    model_grads_orig = grads_orig[0]
    for i in range(DEPTH):
        e = float(layer_e_orig[i])
        g = model_grads_orig[i]
        gnorm = 0.0
        if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
            gnorm = float(jnp.linalg.norm(g.linear.weight))
        print(f"    Layer {i}: energy={e:.6e}, ||dE/dW||={gnorm:.6e}")


def test_h4_resnet_sgd_vs_adam():
    """H4: Are resnet early-layer updates real gradient or Adam noise?"""
    print("\n" + "=" * 70)
    print("H4: Resnet weight updates — SGD vs Adam (NO inference, ffwd init only)")
    print("    If SGD gives ~0 at early layers → updates are from Adam noise")
    print("    If SGD gives ~0.1 at early layers → real gradient signal")
    print("=" * 70)

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant("resnet")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu")

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, _ = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(model)
    old_weights = [jnp.array(w) for w in variant.get_weight_arrays(model)]

    for optim_name, optim_fn in [("SGD(1e-3)", optax.sgd(1e-3)),
                                   ("Adam(1e-3)", optax.adam(1e-3))]:
        opt_state = optim_fn.init(variant.get_optimizer_target(model))
        result = jpc.update_pc_params(
            params=params, activities=activities,
            optim=optim_fn, opt_state=opt_state,
            output=label_batch, input=img_batch, param_type="sp",
        )
        model_new = variant.post_learning_step(model, result, None)
        new_weights = [jnp.array(w) for w in variant.get_weight_arrays(model_new)]
        print(f"\n  {optim_name} — weight update norms (NO inference):")
        for i in range(DEPTH):
            wu = float(jnp.linalg.norm(new_weights[i] - old_weights[i]))
            print(f"    Layer {i}: ||dW||={wu:.6e}")


if __name__ == "__main__":
    test_h1_dyt_disabled()
    test_h2_external_skip()
    test_h4_resnet_sgd_vs_adam()
