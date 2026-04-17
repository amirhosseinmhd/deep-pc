"""Training loop for res-error-net (iterative PC with residual error highways).

Per batch:
  1. Feed-forward init of z from x
  2. Hard-clamp z[L-1] = y_batch
  3. Run T Euler steps of the inference ODE:
        ż^i = -∂F/∂z^i   (F = F_pc + F_highway)
  4. Compute ΔW (standard PC autodiff on F_pc)
  5. Compute ΔV (energy rule: α e^i (e^L)^T; or state rule)
  6. Re-project both to a Gaussian ball (paper convention from rec-LRA)
  7. Apply via optax (AdamW/Adam) or SGD
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jpc
import optax

from common.data import set_seed, get_dataloaders
from common.metrics import MetricsCollector
from variants.rec_lra_common import reproject_to_ball, add_input_noise


def train_res_error_net(
    variant,
    model,
    depth,
    seed,
    param_lr,
    v_lr,
    batch_size,
    n_train_iters,
    test_every,
    act_fn,
    dataset="MNIST",
    alpha=0.1,
    inference_T=20,
    inference_dt=0.1,
    v_update_rule="energy",
    optim_type="adamw",
    loss_type="mse",
    reproject_c=1.0,
    input_noise_sigma=0.1,
    weight_decay=1e-4,
    use_zca=False,
    track_weight_updates=True,
    track_activity_norms=True,
    track_grad_norms=True,
    track_layer_energy=True,
    use_wandb=False,
):
    """Train a res-error-net and record all metrics."""
    set_seed(seed)

    use_optax = optim_type in ("adam", "adamw")
    if use_optax:
        if optim_type == "adamw":
            w_optim = optax.adamw(param_lr, weight_decay=weight_decay, eps=1e-12)
            v_optim = optax.adamw(v_lr, weight_decay=weight_decay, eps=1e-12)
        else:
            w_optim = optax.adam(param_lr, eps=1e-12)
            v_optim = optax.adam(v_lr, eps=1e-12)
        w_opt_state = variant.init_w_optim_states(model, w_optim)
        v_opt_state = variant.init_v_optim_states(model, v_optim)

    noise_key = jr.PRNGKey(seed + 1)

    train_loader, test_loader = get_dataloaders(
        dataset, batch_size, use_zca=use_zca
    )

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

        img_batch = img_batch.numpy()
        label_batch = label_batch.numpy()

        if input_noise_sigma and input_noise_sigma > 0.0:
            noise_key, sub = jr.split(noise_key)
            img_batch = np.asarray(
                add_input_noise(sub, jnp.asarray(img_batch), input_noise_sigma)
            )

        if track_weight_updates:
            old_weights = variant.get_weight_arrays(model)
            old_weights = [jnp.array(w) for w in old_weights]

        # --- Step 1: Feed-forward init ---
        z_init, _ = variant.forward_pass(model, img_batch)

        if track_activity_norms:
            metrics.record_activity_norms_pre(z_init)

        # --- Step 2-3: Clamp output + T inference steps ---
        z, _ = variant.run_inference(
            model, img_batch, label_batch,
            alpha=alpha, dt=inference_dt, T=inference_T,
            z_init=z_init, record_energy=False,
        )

        # --- Errors at convergence (for logging & updates) ---
        e = variant.compute_errors(model, z, img_batch, label_batch)

        if track_layer_energy:
            layer_energies = [
                float(0.5 * jnp.mean(jnp.sum(el ** 2, axis=1)))
                for el in e
            ]
            metrics.energy_per_layer.append(layer_energies)

        # --- Step 4: ΔW from F_pc autodiff ---
        delta_W = variant.compute_W_updates(model, z, img_batch, label_batch)

        # --- Step 5: ΔV per configured rule ---
        delta_V = variant.compute_V_updates(
            model, z, img_batch, label_batch, alpha, rule=v_update_rule,
        )

        # --- Step 6: Re-project ---
        if reproject_c is not None and reproject_c > 0.0:
            delta_W = [reproject_to_ball(dw, reproject_c) for dw in delta_W]
            delta_V = [reproject_to_ball(dv, reproject_c) for dv in delta_V]

        if track_grad_norms:
            grad_norms = [float(jnp.linalg.norm(jnp.ravel(dw))) for dw in delta_W]
            metrics.grad_norms.append(grad_norms)

        # --- Step 7: Apply ---
        if use_optax:
            model, w_opt_state, v_opt_state = variant.apply_optax_updates(
                model, delta_W, delta_V,
                w_optim, w_opt_state, v_optim, v_opt_state,
            )
        else:
            model = variant.apply_sgd_updates(
                model, delta_W, delta_V, param_lr, v_lr
            )

        if track_activity_norms:
            metrics.record_activity_norms_post(z)

        if track_weight_updates:
            new_weights = variant.get_weight_arrays(model)
            metrics.record_weight_update_norms(old_weights, new_weights)

        # --- Loss / accuracy on feed-forward output (no inference, no clamp) ---
        z_ff, _ = variant.forward_pass(model, img_batch)
        preds = z_ff[-1]
        if loss_type == "ce":
            train_loss = float(jpc.cross_entropy_loss(preds, label_batch))
        else:
            train_loss = float(jpc.mse_loss(preds, label_batch))
        metrics.record_train_loss(train_loss)

        train_acc = float(jnp.mean(
            jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
        ) * 100)

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

            # V-highway norms
            V_arrays = variant.get_V_arrays(model)
            S = model["highway_indices"]
            for idx_v, i in enumerate(S):
                wb_metrics[f"v_norm/layer_{i}"] = float(
                    jnp.linalg.norm(jnp.ravel(V_arrays[idx_v]))
                )

            log_step_metrics(iter_num, wb_metrics)

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

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
