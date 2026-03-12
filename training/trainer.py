"""Unified training loop that collects ALL metrics in one pass."""

import numpy as np
import jax.numpy as jnp
from jax import vmap
import jpc
import optax

from common.data import set_seed, get_mnist_loaders
from common.metrics import MetricsCollector
from common.utils import selected_layer_indices


def train_and_record(
    variant,
    model,
    depth,
    seed,
    activity_lr,
    param_lr,
    batch_size,
    n_train_iters,
    test_every,
    act_fn,
    track_weight_updates=True,
    track_activity_norms=True,
    track_grad_norms=True,
    track_layer_energy=True,
    inference_multiplier=1.0,
    activity_init="ffwd",
    param_optim_type="adam",
    use_wandb=False,
):
    """Train a PCN and record all metrics in a single pass.

    Args:
        variant: a PCVariant instance (handles model-specific logic)
        model: the model object (dict for baseline/resnet, custom Module for others)
        depth: network depth (number of layers)
        seed: random seed
        activity_lr: learning rate for activity updates (SGD)
        param_lr: learning rate for parameter updates (Adam)
        batch_size: training batch size
        n_train_iters: total number of training iterations
        test_every: evaluate every N iterations
        act_fn: activation function name
        track_*: flags to enable/disable specific metric tracking

    Returns:
        dict with numpy arrays for all tracked metrics
    """
    set_seed(seed)

    activity_optim = optax.sgd(activity_lr)
    if param_optim_type == "sgd":
        param_optim = optax.sgd(param_lr)
    else:
        param_optim = optax.adam(param_lr, eps=1e-12)
    param_opt_state = param_optim.init(variant.get_optimizer_target(model))

    train_loader, test_loader = get_mnist_loaders(batch_size)

    metrics = MetricsCollector(
        track_weight_updates=track_weight_updates,
        track_activity_norms=track_activity_norms,
        track_grad_norms=track_grad_norms,
        track_layer_energy=track_layer_energy,
    )

    if use_wandb:
        from common.wandb_logger import log_step_metrics
        wandb_layer_idxs = list(range(depth))

    data_iter = iter(train_loader)
    for iter_num in range(n_train_iters):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)

        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # === Step 1: Init activities (variant-specific) ===
        activities, batch_stats, effective_model = variant.init_activities(
            model, img_batch
        )
        if activity_init == "zeros":
            activities = [jnp.zeros_like(a) for a in activities]

        # Record pre-inference activity norms
        if track_activity_norms:
            metrics.record_activity_norms_pre(activities)

        # Snapshot weights before update
        if track_weight_updates:
            old_weights = variant.get_weight_arrays(model)
            old_weights = [jnp.array(w) for w in old_weights]

        # === Step 2: PC Inference ===
        params_for_jpc = variant.get_params_for_jpc(effective_model)
        activity_opt_state = activity_optim.init(activities)

        for t in range(round(depth * inference_multiplier)):
            result = jpc.update_pc_activities(
                params=params_for_jpc,
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=label_batch,
                input=img_batch,
                param_type=variant.get_param_type(),
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        preds = vmap(params_for_jpc[0][-1])(activities[-2])
        train_loss = float(jpc.mse_loss(preds, label_batch))
        metrics.record_train_loss(train_loss)
        train_acc = float(
            jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100
        )

        # Record post-inference activity norms
        if track_activity_norms:
            metrics.record_activity_norms_post(activities)

        # Record per-layer energy (after inference, before param update)
        if track_layer_energy:
            try:
                layer_energies = jpc.pc_energy_fn(
                    params=params_for_jpc,
                    activities=activities,
                    y=label_batch,
                    x=img_batch,
                    param_type=variant.get_param_type(),
                    record_layers=True,
                )
                metrics.record_layer_energy(layer_energies)
            except Exception:
                # record_layers may not be supported in all jpc versions
                pass

        # === Step 3: Learning ===
        result = jpc.update_pc_params(
            params=params_for_jpc,
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=label_batch,
            input=img_batch,
            param_type=variant.get_param_type(),
        )
        param_opt_state = result["opt_state"]

        # Record gradient norms if available
        if track_grad_norms and "grads" in result:
            grads_model = result["grads"][0]  # first element of params tuple
            metrics.record_grad_norms(grads_model)

        # === Step 4: Post-learning (variant-specific) ===
        model = variant.post_learning_step(model, result, batch_stats)

        # Record weight update norms
        if track_weight_updates:
            new_weights = variant.get_weight_arrays(model)
            metrics.record_weight_update_norms(old_weights, new_weights)

        # --- W&B step logging ---
        if use_wandb:
            wb_metrics = {"train_loss": train_loss, "train_acc": train_acc}

            if track_weight_updates and metrics.weight_update_norms:
                last_wu = metrics.weight_update_norms[-1]
                for idx in wandb_layer_idxs:
                    if idx < len(last_wu):
                        wb_metrics[f"weight_updates/layer_{idx+1}"] = last_wu[idx]

            if track_activity_norms and metrics.activity_norms_init:
                last_init = metrics.activity_norms_init[-1]
                last_post = metrics.activity_norms_post[-1]
                for idx in wandb_layer_idxs:
                    if idx < len(last_init):
                        wb_metrics[f"activity_norms_init/layer_{idx+1}"] = last_init[idx]
                        wb_metrics[f"activity_norms_post/layer_{idx+1}"] = last_post[idx]

            if track_grad_norms and metrics.grad_norms:
                last_gn = metrics.grad_norms[-1]
                for idx in wandb_layer_idxs:
                    if idx < len(last_gn):
                        wb_metrics[f"grad_norms/layer_{idx+1}"] = last_gn[idx]

            if track_layer_energy and metrics.energy_per_layer:
                last_e = metrics.energy_per_layer[-1]
                for idx in wandb_layer_idxs:
                    if idx < len(last_e):
                        wb_metrics[f"energy/layer_{idx+1}"] = last_e[idx]

            log_step_metrics(iter_num, wb_metrics)

        # Divergence check
        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

        # Periodic evaluation
        if ((iter_num + 1) % test_every) == 0:
            avg_acc = variant.evaluate(model, test_loader)
            metrics.record_test(iter_num + 1, avg_acc)
            print(
                f"  Iter {iter_num+1}, loss={train_loss:.4f}, "
                f"train acc={train_acc:.2f}, test acc={avg_acc:.2f}"
            )
            if use_wandb:
                log_step_metrics(iter_num, {
                    "test_acc": avg_acc,
                    "test_iter": iter_num + 1,
                })

    return metrics.finalize()
