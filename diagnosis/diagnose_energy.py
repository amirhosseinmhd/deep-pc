#!/usr/bin/env python
"""Diagnose zero per-layer energy at depth ~10.

Tests multiple inference multipliers and traces per-step propagation.
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

DEPTH = 10
WIDTH = 128
ACT_FN = "relu"
SEED = 42
ACTIVITY_LR = 1e-2
BATCH_SIZE = 128


def get_layer_energies(params, activities, label_batch, img_batch, param_type):
    return jpc.pc_energy_fn(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type=param_type, record_layers=True,
    )


def run_inference_and_track(variant, model, img_batch, label_batch, n_steps):
    """Run inference and return per-layer energy at each step."""
    activities, _, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    param_type = variant.get_param_type()

    activity_optim = optax.sgd(ACTIVITY_LR)
    activity_opt_state = activity_optim.init(activities)

    # Track energy at each step
    step_energies = []
    step_energies.append(get_layer_energies(params, activities, label_batch, img_batch, param_type))

    # Also track activity change per layer at each step
    step_activity_deltas = []

    for t in range(n_steps):
        old_activities = [jnp.array(a) for a in activities]
        result = jpc.update_pc_activities(
            params=params, activities=activities,
            optim=activity_optim, opt_state=activity_opt_state,
            output=label_batch, input=img_batch,
            param_type=param_type,
        )
        activities = result["activities"]
        activity_opt_state = result["opt_state"]

        # Activity change per layer
        deltas = [float(jnp.mean(jnp.abs(activities[i] - old_activities[i])))
                  for i in range(len(activities))]
        step_activity_deltas.append(deltas)

        step_energies.append(get_layer_energies(params, activities, label_batch, img_batch, param_type))

    return step_energies, step_activity_deltas, activities


def diagnose_variant(variant_name):
    print(f"\n{'='*70}")
    print(f"Variant: {variant_name}, depth={DEPTH}, width={WIDTH}, act_fn={ACT_FN}")
    print(f"{'='*70}")

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant(variant_name)
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn=ACT_FN)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    # ---- Test with 10x multiplier (100 steps) ----
    n_steps = DEPTH * 10  # 100 steps
    print(f"\nRunning {n_steps} inference steps (multiplier=10)...")
    step_energies, step_deltas, final_activities = run_inference_and_track(
        variant, model, img_batch, label_batch, n_steps
    )

    n_layers = len(step_energies[0])

    # Show energy at key checkpoints
    checkpoints = [0, 10, 20, 50, 100]
    print(f"\nPer-layer energy at inference step checkpoints:")
    header = f"{'Layer':<8}" + "".join(f"{'step='+str(s):<14}" for s in checkpoints)
    print(header)
    for l in range(n_layers):
        row = f"  L{l:<5}"
        for s in checkpoints:
            if s < len(step_energies):
                row += f"{float(step_energies[s][l]):<14.4e}"
        print(row)

    # Show activity deltas at key steps - which layers are actually updating?
    print(f"\nMean |activity change| per layer at each inference step:")
    delta_checkpoints = [0, 1, 2, 5, 9, 19, 49, 99]
    header = f"{'Layer':<8}" + "".join(f"{'step='+str(s):<14}" for s in delta_checkpoints)
    print(header)
    n_act_layers = len(step_deltas[0])
    for l in range(n_act_layers):
        row = f"  A{l:<5}"
        for s in delta_checkpoints:
            if s < len(step_deltas):
                row += f"{step_deltas[s][l]:<14.4e}"
        print(row)

    # ---- Check the gradient structure directly ----
    print(f"\n--- Gradient analysis at initialization ---")
    activities_init, _, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    param_type = variant.get_param_type()

    # Compute gradient of total energy w.r.t. each activity layer
    def total_energy_fn(acts):
        return jpc.pc_energy_fn(
            params=params, activities=acts,
            y=label_batch, x=img_batch,
            param_type=param_type,
        )

    grads = jax.grad(total_energy_fn)(activities_init)
    print("Gradient norm w.r.t. each activity layer (at ffwd init):")
    for i, g in enumerate(grads):
        gnorm = float(jnp.mean(jnp.abs(g)))
        print(f"  Activity[{i}]: mean |grad| = {gnorm:.6e}")

    # ---- Now perturb a deep layer and check if energy changes ----
    print(f"\n--- Perturbation test: does perturbing layer 3 affect energy? ---")
    activities_perturbed = [jnp.array(a) for a in activities_init]
    activities_perturbed[3] = activities_perturbed[3] + 0.1 * jnp.ones_like(activities_perturbed[3])

    e_orig = get_layer_energies(params, activities_init, label_batch, img_batch, param_type)
    e_pert = get_layer_energies(params, activities_perturbed, label_batch, img_batch, param_type)
    print("Layer energies: original vs perturbed activity[3]:")
    for i in range(n_layers):
        print(f"  Layer {i}: {float(e_orig[i]):.6e}  ->  {float(e_pert[i]):.6e}")


if __name__ == "__main__":
    for vname in ["resnet"]:
        diagnose_variant(vname)
