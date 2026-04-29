#!/usr/bin/env python
"""Compare sp (resnet) vs mupc at depth 50 to find why mupc gets updates
at every layer while sp shows training-iteration delay.

Hypotheses:
H1: mupc scaling changes the effective Jacobian, helping propagation during inference
H2: Weight init (O(1) for mupc vs O(1/sqrt(N)) for sp) affects gradient magnitudes
H3: There's a numerical precision issue - sp has larger intermediate values
    that cause ffwd init errors to compound at float32 precision
H4: The skip connection dominance in mupc (tiny W contribution) means
    activities ≈ z_{l-1}, creating a different gradient structure
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import jpc
import optax
import numpy as np

from common.data import set_seed, get_mnist_loaders
from variants import get_variant

DEPTH = 50
WIDTH = 128
SEED = 42
ACTIVITY_LR = 0.5
PARAM_LR = 1e-3
BATCH_SIZE = 128
# mupc uses tanh, resnet uses relu
SP_ACT_FN = "relu"
MUPC_ACT_FN = "tanh"


def run_one_training_step(variant_name, act_fn, depth=DEPTH):
    """Run 1 full training step and return diagnostics."""
    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant(variant_name)

    model = variant.create_model(key, depth=depth, width=WIDTH, act_fn=act_fn)
    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    # Step 1: ffwd init
    activities, batch_stats, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    param_type = variant.get_param_type()

    # Check initial per-layer energy
    layer_energies_init = jpc.pc_energy_fn(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type=param_type, record_layers=True,
    )

    # Check initial activity gradient
    def energy_fn(acts):
        return jpc.pc_energy_fn(
            params=params, activities=acts,
            y=label_batch, x=img_batch,
            param_type=param_type,
        )
    grads_init = jax.grad(energy_fn)(activities)

    # Step 2: inference
    activity_optim = optax.sgd(ACTIVITY_LR)
    activity_opt_state = activity_optim.init(activities)
    n_inference = round(depth * 10)  # inference_multiplier = 10

    for t in range(n_inference):
        result = jpc.update_pc_activities(
            params=params, activities=activities,
            optim=activity_optim, opt_state=activity_opt_state,
            output=label_batch, input=img_batch,
            param_type=param_type,
        )
        activities = result["activities"]
        activity_opt_state = result["opt_state"]

    # Post-inference energy
    layer_energies_post = jpc.pc_energy_fn(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type=param_type, record_layers=True,
    )

    # Post-inference activity gradient
    grads_post = jax.grad(energy_fn)(activities)

    # Step 3: param update - check weight gradients
    param_optim = optax.adam(PARAM_LR)
    param_opt_state = param_optim.init(variant.get_optimizer_target(model))

    old_weights = variant.get_weight_arrays(model)
    old_weights = [jnp.array(w) for w in old_weights]

    result = jpc.update_pc_params(
        params=params, activities=activities,
        optim=param_optim, opt_state=param_opt_state,
        output=label_batch, input=img_batch,
        param_type=param_type,
    )

    model_new = variant.post_learning_step(model, result, batch_stats)
    new_weights = variant.get_weight_arrays(model_new)

    weight_update_norms = [
        float(jnp.linalg.norm(jnp.array(nw) - ow))
        for ow, nw in zip(old_weights, new_weights)
    ]

    return {
        "layer_energies_init": [float(e) for e in layer_energies_init],
        "layer_energies_post": [float(e) for e in layer_energies_post],
        "activity_grad_init": [float(jnp.mean(jnp.abs(g))) for g in grads_init],
        "activity_grad_post": [float(jnp.mean(jnp.abs(g))) for g in grads_post],
        "weight_update_norms": weight_update_norms,
        "activities": activities,
        "params": params,
        "param_type": param_type,
    }


def print_comparison(name, sp_vals, mupc_vals, n_show=None):
    """Print side-by-side comparison."""
    n = len(sp_vals)
    if n_show is None:
        # Show a subset: first 5, middle 5, last 5
        indices = list(range(min(5, n)))
        if n > 10:
            mid = n // 2
            indices += list(range(mid - 2, mid + 3))
        indices += list(range(max(n - 5, 0), n))
        indices = sorted(set(indices))
    else:
        indices = list(range(n))

    print(f"  {'Idx':<6}{'SP (resnet)':<16}{'muPC':<16}{'ratio':<12}")
    for i in indices:
        sp_v = sp_vals[i] if i < len(sp_vals) else 0
        mu_v = mupc_vals[i] if i < len(mupc_vals) else 0
        if abs(sp_v) > 1e-30 and abs(mu_v) > 1e-30:
            ratio = mu_v / sp_v
            print(f"  {i:<6}{sp_v:<16.4e}{mu_v:<16.4e}{ratio:<12.2e}")
        else:
            print(f"  {i:<6}{sp_v:<16.4e}{mu_v:<16.4e}{'--':<12}")


def main():
    print(f"Depth={DEPTH}, width={WIDTH}, "
          f"sp_act={SP_ACT_FN}, mupc_act={MUPC_ACT_FN}, "
          f"activity_lr={ACTIVITY_LR}, inference_mult=10")

    print("\n" + "=" * 70)
    print(f"Running SP (resnet) variant with {SP_ACT_FN}...")
    print("=" * 70)
    sp = run_one_training_step("resnet", SP_ACT_FN)

    print("\n" + "=" * 70)
    print(f"Running muPC variant with {MUPC_ACT_FN}...")
    print("=" * 70)
    mupc = run_one_training_step("mupc", MUPC_ACT_FN)

    # ===== Compare =====
    print("\n" + "=" * 70)
    print("COMPARISON: Per-layer energy BEFORE inference")
    print("=" * 70)
    print_comparison("energy_init", sp["layer_energies_init"], mupc["layer_energies_init"])

    print("\n" + "=" * 70)
    print("COMPARISON: Per-layer energy AFTER inference")
    print("=" * 70)
    print_comparison("energy_post", sp["layer_energies_post"], mupc["layer_energies_post"])

    print("\n" + "=" * 70)
    print("COMPARISON: Activity gradient norms BEFORE inference (at ffwd init)")
    print("=" * 70)
    print_comparison("grad_init", sp["activity_grad_init"], mupc["activity_grad_init"])

    print("\n" + "=" * 70)
    print("COMPARISON: Activity gradient norms AFTER inference")
    print("=" * 70)
    print_comparison("grad_post", sp["activity_grad_post"], mupc["activity_grad_post"])

    print("\n" + "=" * 70)
    print("COMPARISON: Weight update norms (after 1 training step)")
    print("=" * 70)
    print_comparison("weight_update", sp["weight_update_norms"], mupc["weight_update_norms"])

    # ===== H3: Check activity magnitudes through the network =====
    print("\n" + "=" * 70)
    print("H3: Activity magnitudes through network (ffwd init)")
    print("   Large values → more float32 precision loss → nonzero errors?")
    print("=" * 70)
    sp_act_norms = [float(jnp.mean(jnp.linalg.norm(a, axis=1)))
                    for a in sp["activities"]]
    mupc_act_norms = [float(jnp.mean(jnp.linalg.norm(a, axis=1)))
                      for a in mupc["activities"]]
    print_comparison("act_norm", sp_act_norms, mupc_act_norms)

    # ===== H1: Check Jacobian structure =====
    print("\n" + "=" * 70)
    print("H1: Effective Jacobian spectral norms (with scaling)")
    print("   J_l = s_l * W_l * diag(act'(z)) + I  (for skip connections)")
    print("=" * 70)
    # SP
    sp_model_layers, sp_skip = sp["params"]
    mupc_model_layers, mupc_skip = mupc["params"]

    sp_scalings = jpc._get_param_scalings(
        sp_model_layers, jnp.zeros((1, 784)),
        skip_model=sp_skip, param_type="sp"
    )
    mupc_scalings = jpc._get_param_scalings(
        mupc_model_layers, jnp.zeros((1, 784)),
        skip_model=mupc_skip, param_type="mupc"
    )
    print(f"  SP scalings: {sp_scalings[0]:.4f}, {sp_scalings[1]:.4f} (hidden), {sp_scalings[-1]:.4f}")
    print(f"  muPC scalings: {mupc_scalings[0]:.6f}, {mupc_scalings[1]:.6f} (hidden), {mupc_scalings[-1]:.6f}")

    # Check raw weight norms
    print("\n  Raw weight spectral norms (before scaling):")
    sp_wnorms = []
    mupc_wnorms = []
    indices = [0, 1, 5, 10, 25, 48, 49]
    for i in indices:
        sp_W = sp_model_layers[i].layers[1].weight
        mupc_W = mupc_model_layers[i].layers[1].weight
        sp_sn = float(jnp.linalg.svdvals(sp_W)[0])
        mupc_sn = float(jnp.linalg.svdvals(mupc_W)[0])
        sp_eff = sp_sn * sp_scalings[min(i, len(sp_scalings)-1)]
        mupc_eff = mupc_sn * mupc_scalings[min(i, len(mupc_scalings)-1)]
        print(f"    Layer {i}: SP ||W||={sp_sn:.4f} eff={sp_eff:.4f}, "
              f"muPC ||W||={mupc_sn:.4f} eff={mupc_eff:.4f}")


if __name__ == "__main__":
    main()
