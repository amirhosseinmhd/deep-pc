#!/usr/bin/env python
"""Diagnose why dyt_v2 has zero energy at deep layers.

Compare dyt_v2 vs resnet at depth 50, both with relu.
Focus on:
1. Initial per-layer energy (should both be ~0 from ffwd init)
2. Activity gradients at init
3. After inference: which layers wake up
4. After 1 training step: weight update norms per layer
5. After N training steps: when do deep layers start getting updates
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
ACT_FN = "relu"
INFERENCE_MULT = 10


def diagnose_variant(variant_name, act_fn=ACT_FN, n_train_iters=5):
    print(f"\n{'='*70}")
    print(f"Variant: {variant_name}, depth={DEPTH}, act_fn={act_fn}, "
          f"activity_lr={ACTIVITY_LR}, inf_mult={INFERENCE_MULT}")
    print(f"{'='*70}")

    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant(variant_name)
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn=act_fn,
                                  init_alpha=0.5)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    data_iter = iter(train_loader)

    activity_optim = optax.sgd(ACTIVITY_LR)
    param_optim = optax.adam(PARAM_LR)
    param_opt_state = param_optim.init(variant.get_optimizer_target(model))

    # Track weight update norms across training iterations
    all_weight_update_norms = []

    for train_iter in range(n_train_iters):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # Init activities
        activities, batch_stats, effective_model = variant.init_activities(
            model, img_batch)
        params = variant.get_params_for_jpc(effective_model)
        param_type = variant.get_param_type()

        if train_iter == 0:
            # Detailed diagnostics on first iteration
            print(f"\n--- Iteration 0: Initial state ---")
            print(f"  skip_model is None: {params[1] is None}")
            print(f"  param_type: {param_type}")
            print(f"  Num layers: {len(params[0])}")
            print(f"  Num activities: {len(activities)}")

            # Per-layer energy at init
            layer_e = jpc.pc_energy_fn(
                params=params, activities=activities,
                y=label_batch, x=img_batch,
                param_type=param_type, record_layers=True,
            )
            print(f"\n  Per-layer energy at ffwd init:")
            for i, e in enumerate(layer_e):
                if i < 5 or i > len(layer_e) - 6 or abs(float(e)) > 1e-10:
                    print(f"    Layer {i}: {float(e):.6e}")
            if len(layer_e) > 10:
                print(f"    ... (layers 5-{len(layer_e)-6} all ~0)")

            # Activity norms
            print(f"\n  Activity norms at ffwd init:")
            show_idxs = [0, 1, 2, 5, 10, 20, 30, 40, 48, 49]
            for i in show_idxs:
                if i < len(activities):
                    norm = float(jnp.mean(jnp.linalg.norm(activities[i], axis=1)))
                    print(f"    Activity[{i}]: norm={norm:.4e}, shape={activities[i].shape}")

            # Activity gradients at init
            def energy_fn(acts):
                return jpc.pc_energy_fn(
                    params=params, activities=acts,
                    y=label_batch, x=img_batch,
                    param_type=param_type,
                )
            grads = jax.grad(energy_fn)(activities)
            print(f"\n  Activity gradient norms at ffwd init:")
            for i in show_idxs:
                if i < len(grads):
                    gnorm = float(jnp.mean(jnp.abs(grads[i])))
                    print(f"    dE/dz[{i}]: {gnorm:.6e}")

        # Inference
        activity_opt_state = activity_optim.init(activities)
        n_inf = round(DEPTH * INFERENCE_MULT)
        for _ in range(n_inf):
            result = jpc.update_pc_activities(
                params=params, activities=activities,
                optim=activity_optim, opt_state=activity_opt_state,
                output=label_batch, input=img_batch,
                param_type=param_type,
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        if train_iter == 0:
            # Post-inference energy
            layer_e_post = jpc.pc_energy_fn(
                params=params, activities=activities,
                y=label_batch, x=img_batch,
                param_type=param_type, record_layers=True,
            )
            print(f"\n  Per-layer energy AFTER {n_inf} inference steps:")
            for i, e in enumerate(layer_e_post):
                if i < 5 or i > len(layer_e_post) - 6 or abs(float(e)) > 1e-10:
                    print(f"    Layer {i}: {float(e):.6e}")

        # Snapshot weights
        old_weights = [jnp.array(w) for w in variant.get_weight_arrays(model)]

        # Param update
        result = jpc.update_pc_params(
            params=params, activities=activities,
            optim=param_optim, opt_state=param_opt_state,
            output=label_batch, input=img_batch,
            param_type=param_type,
        )
        param_opt_state = result["opt_state"]
        model = variant.post_learning_step(model, result, batch_stats)

        # Weight update norms
        new_weights = [jnp.array(w) for w in variant.get_weight_arrays(model)]
        wu_norms = [float(jnp.linalg.norm(nw - ow))
                    for ow, nw in zip(old_weights, new_weights)]
        all_weight_update_norms.append(wu_norms)

        # Print weight update summary
        n_layers = len(wu_norms)
        nonzero = sum(1 for w in wu_norms if w > 1e-10)
        max_wu = max(wu_norms)
        print(f"\n  Iter {train_iter}: {nonzero}/{n_layers} layers with nonzero "
              f"weight update, max={max_wu:.4e}")

        # Show which layers got updates
        show_idxs_w = [0, 1, 2, 5, 10, 20, 30, 40, n_layers-3, n_layers-2, n_layers-1]
        for i in show_idxs_w:
            if i < n_layers:
                print(f"    Layer {i}: ||dW||={wu_norms[i]:.6e}")

    return all_weight_update_norms


def main():
    print("=" * 70)
    print("DIAGNOSING dyt_v2 zero-energy problem")
    print("=" * 70)

    wu_dyt = diagnose_variant("dyt_v2", act_fn="relu", n_train_iters=5)

    print("\n\n" + "=" * 70)
    print("COMPARISON: resnet (known working)")
    print("=" * 70)
    wu_res = diagnose_variant("resnet", act_fn="relu", n_train_iters=5)


if __name__ == "__main__":
    main()
