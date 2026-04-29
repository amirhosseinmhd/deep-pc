#!/usr/bin/env python
"""
THE FULL STORY: Why external skip works and internal skip doesn't.

Head-to-head comparison of:
  - dyt   (v1): skip INSIDE the layer, no jpc skip_model
  - dyt_v2:     skip OUTSIDE (jpc skip_model), layer does DyT(W*relu(x)) only

We trace every step of the PC algorithm and show exactly where they diverge.
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


def run_comparison():
    set_seed(SEED)
    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    key = jr.PRNGKey(SEED)

    # ================================================================
    # STEP 1: Create both models with identical weights
    # ================================================================
    print("=" * 70)
    print("STEP 1: MODEL CREATION")
    print("=" * 70)

    # dyt v1 (internal skip)
    v1 = get_variant("dyt")
    model_v1 = v1.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu",
                                init_alpha=0.5)

    # dyt v2 (external skip)
    v2 = get_variant("dyt_v2")
    model_v2 = v2.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu",
                                init_alpha=0.5)

    # Verify weights are identical
    w_v1 = v1.get_weight_arrays(model_v1)
    w_v2 = v2.get_weight_arrays(model_v2)
    max_diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(w_v1, w_v2))
    print(f"  Max weight difference between v1 and v2: {max_diff}")
    print(f"  (Should be 0.0 — same random key)")

    # ================================================================
    # STEP 2: Forward pass initialization
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 2: FORWARD PASS INIT (how activities are initialized)")
    print("=" * 70)

    # v1: custom init (skip inside layer)
    activities_v1, _, eff_model_v1 = v1.init_activities(model_v1, img_batch)
    params_v1 = v1.get_params_for_jpc(eff_model_v1)

    # v2: jpc init (skip outside via skip_model)
    activities_v2, _, eff_model_v2 = v2.init_activities(model_v2, img_batch)
    params_v2 = v2.get_params_for_jpc(eff_model_v2)

    # Are the activities the same?
    print("\n  Activity comparison (max |v1 - v2| per layer):")
    for i in range(min(5, len(activities_v1))):
        diff = float(jnp.max(jnp.abs(activities_v1[i] - activities_v2[i])))
        print(f"    Layer {i}: {diff:.6e}")
    print(f"    ...")
    for i in range(len(activities_v1) - 3, len(activities_v1)):
        diff = float(jnp.max(jnp.abs(activities_v1[i] - activities_v2[i])))
        print(f"    Layer {i}: {diff:.6e}")
    print("  (Small diffs expected — same math, potentially different float32 ordering)")

    # ================================================================
    # STEP 3: Prediction errors RIGHT AFTER INIT (before any inference)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 3: PREDICTION ERRORS RIGHT AFTER INIT")
    print("This is the KEY difference.")
    print("=" * 70)

    # v1: energy per layer
    energies_v1 = jpc.pc_energy_fn(
        params=params_v1, activities=activities_v1,
        y=label_batch, x=img_batch, param_type="sp",
        record_layers=True,
    )

    # v2: energy per layer
    energies_v2 = jpc.pc_energy_fn(
        params=params_v2, activities=activities_v2,
        y=label_batch, x=img_batch, param_type="sp",
        record_layers=True,
    )

    print(f"\n  {'Layer':<8} {'v1 (internal skip)':<22} {'v2 (external skip)':<22}")
    print(f"  {'':<8} {'err = z - layer(z_prev)':<22} {'err = z - model(z) - skip(z)':<22}")
    print(f"  {'-'*60}")

    # Output energy is first (index 0 in energies), then hidden in order, then input
    # Actually from the code: energies = [output_energy] + [hidden energies 1..N-2] + [input_energy]
    n_layers = len(activities_v1)
    for i in range(min(len(energies_v1), 10)):
        e1 = float(energies_v1[i])
        e2 = float(energies_v2[i])
        if i == 0:
            label = "output"
        elif i < len(energies_v1) - 1:
            label = f"hidden {i}"
        else:
            label = "input"
        print(f"  {label:<8} {e1:<22.6e} {e2:<22.6e}")
    print(f"  ...")
    for i in range(max(0, len(energies_v1) - 3), len(energies_v1)):
        e1 = float(energies_v1[i])
        e2 = float(energies_v2[i])
        label = f"hidden {i}" if i < len(energies_v1) - 1 else "input"
        print(f"  {label:<8} {e1:<22.6e} {e2:<22.6e}")

    v1_hidden_zeros = sum(1 for i in range(1, len(energies_v1) - 1) if float(energies_v1[i]) == 0.0)
    v2_hidden_zeros = sum(1 for i in range(1, len(energies_v2) - 1) if float(energies_v2[i]) == 0.0)
    print(f"\n  Hidden layers with EXACTLY 0.0 energy:")
    print(f"    v1: {v1_hidden_zeros} / {len(energies_v1) - 2}")
    print(f"    v2: {v2_hidden_zeros} / {len(energies_v2) - 2}")

    # ================================================================
    # STEP 4: Activity gradients at step 0 (before any inference)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 4: ACTIVITY GRADIENTS AT STEP 0 (what drives inference)")
    print("=" * 70)

    def energy_fn_v1(acts):
        return jpc.pc_energy_fn(params=params_v1, activities=acts,
                                 y=label_batch, x=img_batch, param_type="sp")

    def energy_fn_v2(acts):
        return jpc.pc_energy_fn(params=params_v2, activities=acts,
                                 y=label_batch, x=img_batch, param_type="sp")

    grads_v1 = jax.grad(energy_fn_v1)(activities_v1)
    grads_v2 = jax.grad(energy_fn_v2)(activities_v2)

    print(f"\n  {'Layer':<8} {'v1 mean|grad|':<20} {'v2 mean|grad|':<20}")
    print(f"  {'-'*50}")
    for i in [0, 1, 2, 3, 44, 45, 46, 47, 48, 49]:
        if i < len(grads_v1):
            g1 = float(jnp.mean(jnp.abs(grads_v1[i])))
            g2 = float(jnp.mean(jnp.abs(grads_v2[i])))
            print(f"  {i:<8} {g1:<20.6e} {g2:<20.6e}")

    # ================================================================
    # STEP 5: Run inference and track step-by-step
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 5: INFERENCE (500 steps, activity_lr=0.5)")
    print("Tracking when the wavefront reaches each layer")
    print("=" * 70)

    N_STEPS = 500
    ACTIVITY_LR = 0.5

    for name, params, activities in [
        ("v1 (internal skip)", params_v1, [jnp.array(a) for a in activities_v1]),
        ("v2 (external skip)", params_v2, [jnp.array(a) for a in activities_v2]),
    ]:
        optim = optax.sgd(ACTIVITY_LR)
        opt_state = optim.init(activities)
        param_type = "sp"

        first_nonzero = [None] * len(activities)
        threshold = 1e-20  # extremely sensitive

        for t in range(N_STEPS):
            old_activities = [jnp.array(a) for a in activities]

            result = jpc.update_pc_activities(
                params=params, activities=activities,
                optim=optim, opt_state=opt_state,
                output=label_batch, input=img_batch,
                param_type=param_type,
            )
            activities = result["activities"]
            opt_state = result["opt_state"]

            for i in range(len(activities)):
                delta = float(jnp.max(jnp.abs(activities[i] - old_activities[i])))
                if first_nonzero[i] is None and delta > threshold:
                    first_nonzero[i] = t

        print(f"\n  {name}: first step with |delta_z| > {threshold}")
        print(f"  {'Layer':<8} {'First step':<12} {'Dist from output':<20}")
        # Show selected layers
        for i in [0, 1, 2, 10, 20, 30, 40, 44, 45, 46, 47, 48, 49]:
            if i < len(activities):
                step = first_nonzero[i]
                step_str = str(step) if step is not None else "NEVER"
                dist = (len(activities) - 2) - i
                print(f"  {i:<8} {step_str:<12} {dist:<20}")

        # Weight gradients after inference
        print(f"\n  {name}: weight gradient norms AFTER {N_STEPS} inference steps")
        grads = jpc.compute_pc_param_grads(
            params=params, activities=activities,
            y=label_batch, x=img_batch, param_type=param_type,
        )
        model_grads = grads[0]

        # Summarize by region
        deep_grads = []
        mid_grads = []
        near_grads = []
        for i in range(len(model_grads)):
            g = model_grads[i]
            gnorm = 0.0
            if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
                gnorm = float(jnp.linalg.norm(g.linear.weight))
            elif hasattr(g, 'layers') and len(g.layers) > 1:
                lin = g.layers[1]
                if hasattr(lin, 'weight') and lin.weight is not None:
                    gnorm = float(jnp.linalg.norm(lin.weight))

            dist = (len(activities) - 2) - i
            if dist > 30:
                deep_grads.append(gnorm)
            elif dist > 10:
                mid_grads.append(gnorm)
            else:
                near_grads.append(gnorm)

            if i in [0, 1, 10, 20, 30, 40, 45, 48, 49]:
                print(f"    Layer {i} (dist={dist}): ||dE/dW|| = {gnorm:.6e}")

        print(f"\n    Summary:")
        print(f"      Deep layers (dist > 30):  mean ||dE/dW|| = "
              f"{sum(deep_grads)/max(len(deep_grads),1):.6e}")
        print(f"      Mid layers  (10 < dist ≤ 30): mean ||dE/dW|| = "
              f"{sum(mid_grads)/max(len(mid_grads),1):.6e}")
        print(f"      Near layers (dist ≤ 10): mean ||dE/dW|| = "
              f"{sum(near_grads)/max(len(near_grads),1):.6e}")

    # ================================================================
    # STEP 6: What Adam does with these gradients
    # ================================================================
    print(f"\n{'=' * 70}")
    print("STEP 6: ADAM AMPLIFICATION")
    print("=" * 70)

    lr = 1e-3
    eps = 1e-8
    print(f"  Adam parameters: lr={lr}, eps={eps}")
    print(f"\n  For gradient g, Adam step ≈ lr * g / (|g| + eps)")
    for g_val in [0.0, 1e-14, 1e-10, 1e-8, 1e-6, 1e-3, 1.0]:
        if g_val == 0.0:
            update = 0.0
        else:
            update = lr * g_val / (abs(g_val) + eps)
        amplification = update / g_val if g_val != 0 else 0
        print(f"    g = {g_val:.1e} → update = {update:.1e} "
              f"(amplification: {amplification:.1e}x)")


if __name__ == "__main__":
    run_comparison()
