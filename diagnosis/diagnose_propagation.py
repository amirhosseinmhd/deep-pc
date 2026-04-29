#!/usr/bin/env python
"""Why does inference only propagate ~10 layers despite J ≈ I?

Track the wavefront step-by-step: at which inference step does each
layer first get a nonzero activity update?

Also measure the damping: the prediction error self-term
(err_l = z_l - f_l(z_{l-1})) acts as a restoring force that competes
with the propagation term (-J^T * err_{l+1}).
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
ACTIVITY_LR = 0.5


def track_wavefront(variant_name, act_fn, n_steps=500):
    """Track when each layer first gets nonzero activity gradient and error."""
    print(f"\n{'='*70}")
    print(f"WAVEFRONT TRACKING: {variant_name}, act_fn={act_fn}, "
          f"depth={DEPTH}, activity_lr={ACTIVITY_LR}")
    print(f"{'='*70}")

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant(variant_name)
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn=act_fn,
                                  init_alpha=0.5)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    param_type = variant.get_param_type()

    activity_optim = optax.sgd(ACTIVITY_LR)
    activity_opt_state = activity_optim.init(activities)

    n_layers = len(activities)

    # Track first nonzero step for each layer
    first_nonzero = [None] * n_layers  # inference step when |delta_z| > threshold
    threshold = 1e-15  # very sensitive threshold

    # Track wavefront at selected steps
    report_steps = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 499]

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

        # Check which layers got updated
        for i in range(n_layers):
            delta = float(jnp.max(jnp.abs(activities[i] - old_activities[i])))
            if first_nonzero[i] is None and delta > threshold:
                first_nonzero[i] = t

        if t in report_steps:
            # Show activity deltas at this step
            deltas = [float(jnp.max(jnp.abs(activities[i] - old_activities[i])))
                      for i in range(n_layers)]
            # Find the deepest layer with nonzero delta
            deepest = -1
            for i in range(n_layers):
                if deltas[i] > threshold:
                    deepest = i
            # Count from OUTPUT (layer n-2 is last hidden)
            layers_from_output = (n_layers - 2) - deepest if deepest >= 0 else -1
            print(f"\n  Step {t}: wavefront reached layer {deepest} "
                  f"({layers_from_output} layers from output)")
            # Show a few layers around the wavefront
            show = list(range(max(0, deepest-2), min(n_layers, deepest+3)))
            show = [0, 1, 2] + show + [n_layers-3, n_layers-2, n_layers-1]
            show = sorted(set(show))
            for i in show:
                if i < n_layers:
                    print(f"    Activity[{i}]: max|delta|={deltas[i]:.4e}")

    # Summary: when did each layer first wake up?
    print(f"\n  WAVEFRONT SUMMARY (first step with |delta| > {threshold}):")
    print(f"  {'Layer':<8}{'First step':<12}{'Layers from output':<20}")
    for i in range(n_layers):
        step_str = str(first_nonzero[i]) if first_nonzero[i] is not None else "never"
        dist = (n_layers - 2) - i
        print(f"  {i:<8}{step_str:<12}{dist:<20}")

    # After all inference: check actual weight gradients
    print(f"\n  WEIGHT GRADIENTS after {n_steps} inference steps:")
    grads = jpc.compute_pc_param_grads(
        params=params, activities=activities,
        y=label_batch, x=img_batch,
        param_type=param_type,
    )
    model_grads = grads[0]
    for i in range(n_layers):
        g = model_grads[i]
        # Try to extract weight gradient norm
        gnorm = 0.0
        if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
            gnorm = float(jnp.linalg.norm(g.linear.weight))
        elif hasattr(g, 'layers') and len(g.layers) > 1:
            lin = g.layers[1]
            if hasattr(lin, 'weight') and lin.weight is not None:
                gnorm = float(jnp.linalg.norm(lin.weight))
        print(f"    Layer {i}: ||dE/dW|| = {gnorm:.6e}")

    return first_nonzero


def measure_damping():
    """Measure the effective damping: how much does a perturbation decay per step."""
    print(f"\n{'='*70}")
    print("DAMPING MEASUREMENT")
    print("At ffwd init, perturb ONE layer and watch the signal propagate")
    print(f"{'='*70}")

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant("dyt_v2")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="relu",
                                  init_alpha=0.5)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    param_type = variant.get_param_type()

    # Perturb layer 45 (5 from output) by a small amount
    perturb_layer = 45
    perturb_size = 0.01
    activities_perturbed = [jnp.array(a) for a in activities]
    activities_perturbed[perturb_layer] = (
        activities_perturbed[perturb_layer] +
        perturb_size * jnp.ones_like(activities_perturbed[perturb_layer])
    )

    # Compute energy gradient at the perturbed state
    def energy_fn(acts):
        return jpc.pc_energy_fn(
            params=params, activities=acts,
            y=label_batch, x=img_batch,
            param_type=param_type,
        )

    grads = jax.grad(energy_fn)(activities_perturbed)
    print(f"\n  Perturbed layer {perturb_layer} by {perturb_size}")
    print(f"  Activity gradient norms at perturbed state:")
    for i in range(max(0, perturb_layer - 5), min(len(grads), perturb_layer + 5)):
        gnorm = float(jnp.mean(jnp.abs(grads[i])))
        print(f"    dE/dz[{i}]: {gnorm:.6e}")

    print(f"\n  Layer {perturb_layer} gradient (self-damping): "
          f"{float(jnp.mean(jnp.abs(grads[perturb_layer]))):.6e}")
    if perturb_layer > 0:
        print(f"  Layer {perturb_layer-1} gradient (downward propagation): "
              f"{float(jnp.mean(jnp.abs(grads[perturb_layer-1]))):.6e}")
    print(f"  Layer {perturb_layer+1} gradient (upward propagation): "
          f"{float(jnp.mean(jnp.abs(grads[perturb_layer+1]))):.6e}")

    damping = float(jnp.mean(jnp.abs(grads[perturb_layer])))
    coupling_down = float(jnp.mean(jnp.abs(grads[perturb_layer-1])))
    coupling_up = float(jnp.mean(jnp.abs(grads[perturb_layer+1])))
    print(f"\n  Damping-to-coupling ratio (downward): {damping/coupling_down:.2f}")
    print(f"  Damping-to-coupling ratio (upward): {damping/coupling_up:.2f}")
    print(f"  Effective propagation factor: {coupling_down/damping:.4f}")


if __name__ == "__main__":
    # Track wavefront for dyt_v2 (internal skip - old behavior)
    # We use the FIXED version but with depth 50 to see propagation limits
    track_wavefront("dyt_v2", "relu", n_steps=500)
    measure_damping()
