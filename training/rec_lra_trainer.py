"""Custom training loop for rec-LRA variant.

rec-LRA uses a fundamentally different training procedure from standard PC:
  - No iterative inference (single forward pass)
  - Backward target-generation sweep with error skip connections
  - Hebbian (outer-product) weight updates

This trainer mirrors the structure of trainer.py but replaces the
jpc.update_pc_activities/params calls with rec-LRA's own algorithm.
"""

import numpy as np
import jax.numpy as jnp
import jpc
import optax

from common.data import set_seed, get_dataloaders
from common.metrics import MetricsCollector


def train_rec_lra(
    variant,
    model,
    depth,
    seed,
    param_lr,
    e_lr,
    batch_size,
    n_train_iters,
    test_every,
    act_fn,
    dataset="MNIST",
    beta=0.1,
    gamma_E=0.01,
    rec_lra_optim="sgd",
    rec_lra_loss="mse",
    rec_lra_e_update="hebbian",
    track_weight_updates=True,
    track_activity_norms=True,
    track_grad_norms=True,
    track_layer_energy=True,
    use_wandb=False,
):
    """Train a rec-LRA network and record all metrics.

    Args:
        variant: RecLRAVariant instance
        model: the model bundle dict
        depth: network depth
        seed: random seed
        param_lr: learning rate for W updates
        e_lr: learning rate for E updates
        batch_size: training batch size
        n_train_iters: total training iterations
        test_every: evaluate every N iterations
        act_fn: activation function name (unused, already in bundle)
        dataset: dataset name
        beta: target nudging strength
        gamma_E: E learning rate scale in Hebbian rule
        rec_lra_optim: "sgd" or "adam"
        rec_lra_loss: "mse" or "ce"
        track_*: metric tracking flags

    Returns:
        dict with numpy arrays for all tracked metrics
    """
    set_seed(seed)

    # Store gamma_E in the bundle for compute_hebbian_updates
    model["gamma_E"] = gamma_E

    # Set up optimizers if using Adam
    use_adam = rec_lra_optim == "adam"
    if use_adam:
        w_optim = optax.adam(param_lr, eps=1e-12)
        e_optim = optax.adam(e_lr, eps=1e-12)

        # Initialize per-layer optimizer states for W (variant handles layer structure)
        w_opt_state = variant.init_w_optim_states(model, w_optim)

        # Initialize per-layer optimizer states for E
        e_opt_state = [None] * len(model["E"])
        for l in range(len(model["E"])):
            if model["E"][l] is not None:
                e_opt_state[l] = e_optim.init(model["E"][l])

    train_loader, test_loader = get_dataloaders(dataset, batch_size)

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

        # Snapshot weights before update
        if track_weight_updates:
            old_weights = variant.get_weight_arrays(model)
            old_weights = [jnp.array(w) for w in old_weights]

        # === Step 1: Forward pass (RUNMODEL) ===
        z, h = variant.forward_pass(model, img_batch)

        # Record activity norms (post-forward = "init" in PC terminology)
        if track_activity_norms:
            metrics.record_activity_norms_pre(z)

        # === Step 2: Compute targets & errors (CALCERRRUNITS) ===
        e, d = variant.compute_targets_and_errors(
            model, z, h, label_batch, beta
        )

        # Record per-layer energy as sum of squared errors per layer
        if track_layer_energy:
            layer_energies = []
            for l in range(len(e)):
                if e[l] is not None:
                    energy_l = float(0.5 * jnp.mean(jnp.sum(e[l] ** 2, axis=1)))
                    layer_energies.append(energy_l)
                else:
                    layer_energies.append(0.0)
            metrics.energy_per_layer.append(layer_energies)

        # === Step 3: Compute updates (COMPUTEUPDATES) ===
        delta_W, delta_E = variant.compute_hebbian_updates(
            model, z, e, d, img_batch
        )

        # Variant 2 (rLRA-dx): override E updates with true gradient
        if rec_lra_e_update == "grad":
            delta_E = variant.compute_grad_E_updates(
                model, z, h, label_batch, beta
            )

        # Record "gradient" norms (Hebbian delta norms)
        if track_grad_norms:
            grad_norms = [float(jnp.linalg.norm(jnp.ravel(dw))) for dw in delta_W]
            metrics.grad_norms.append(grad_norms)

        # === Step 4: Apply updates ===
        if use_adam:
            model, w_opt_state, e_opt_state = variant.apply_optax_updates(
                model, delta_W, delta_E,
                w_optim, w_opt_state, e_optim, e_opt_state,
            )
        else:
            model = variant.apply_hebbian_updates(
                model, delta_W, delta_E, param_lr, e_lr
            )

        # Record activity norms post-update (same as pre since no inference loop)
        if track_activity_norms:
            metrics.record_activity_norms_post(z)

        # Record weight update norms
        if track_weight_updates:
            new_weights = variant.get_weight_arrays(model)
            metrics.record_weight_update_norms(old_weights, new_weights)

        # Compute loss
        preds = z[-1]
        if rec_lra_loss == "ce":
            train_loss = float(jpc.cross_entropy_loss(preds, label_batch))
        else:
            train_loss = float(jpc.mse_loss(preds, label_batch))
        metrics.record_train_loss(train_loss)

        train_acc = float(
            jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100
        )

        # W&B logging
        if use_wandb:
            wb_metrics = {"train_loss": train_loss, "train_acc": train_acc}

            if track_weight_updates and metrics.weight_update_norms:
                last_wu = metrics.weight_update_norms[-1]
                for idx in wandb_layer_idxs:
                    if idx < len(last_wu):
                        wb_metrics[f"weight_updates/layer_{idx+1}"] = last_wu[idx]

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
