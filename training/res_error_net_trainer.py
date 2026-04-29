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
import equinox as eqx

from common.data import set_seed, get_dataloaders
from common.metrics import MetricsCollector
from variants.rec_lra_common import reproject_to_ball, add_input_noise
from variants.res_error_net import (
    _jit_forward_pass, _jit_compute_errors,
    _jit_compute_W_updates, _jit_compute_V_updates,
)


# Fused path: for variants that expose `compute_updates_fused`, compute
# e, ΔW, ΔV and per-layer energies in a single JIT — saves two forward
# passes per batch by reusing one autodiff trace through the network.
@eqx.filter_jit
def _jit_compute_updates_fused(variant, bundle, z, x_batch, y_batch,
                               alpha, rule, v_reg):
    return variant.compute_updates_fused(
        bundle, z, x_batch, y_batch, alpha, rule=rule, v_reg=v_reg,
    )


def _trajectory_heatmap(history, metric_key, title, cbar_label):
    """Heatmap of inference dynamics across training.

    rows = checkpoints (one per test_every iter), cols = inference step (0..T),
    color = metric value. Re-rendered each checkpoint and logged as a single
    wandb.Image so the panel stays readable regardless of checkpoint count.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    arr = np.asarray([h[metric_key] for h in history])  # (n_ckpt, T+1)
    iters = [h["iter"] for h in history]
    n_ckpt, n_step = arr.shape

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if n_ckpt == 1:
        # imshow with a single row collapses the y-extent; pad with itself so
        # the row is visible. Tick label still shows the actual iter.
        arr = np.vstack([arr, arr])
    im = ax.imshow(
        arr, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest",
    )
    ax.set_xlabel("inference step")
    ax.set_ylabel("training iter")
    ax.set_title(title)
    ax.set_xticks([0, n_step - 1])
    ax.set_xticklabels(["0 (pre)", f"{n_step - 1} (post)"])
    # y ticks: subsample to ~6 labels so dense histories stay readable
    if len(iters) <= 6:
        yticks = list(range(len(iters)))
        ylabels = [str(i) for i in iters]
    else:
        idx = np.linspace(0, len(iters) - 1, 6).astype(int)
        yticks = idx.tolist()
        ylabels = [str(iters[i]) for i in idx]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def train_res_error_net(
    variant,
    bundle,
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
    global_clip_norm=10.0,
    input_noise_sigma=0.1,
    weight_decay=1e-4,
    v_reg=0.0,
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
        w_opt_state = variant.init_w_optim_states(bundle, w_optim)
        v_opt_state = variant.init_v_optim_states(bundle, v_optim)

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
        # Pull labels from the variant so wandb keys are named (stem.conv,
        # blockN.conv1, head.linear, ...) rather than opaque "layer_{N}".
        # `weight_labels` is aligned with variant.get_weight_arrays() / delta_W.
        # `weight_log_idxs` is the subset to actually emit to wandb (one conv
        # per architectural block). Falls back to a prefix of range(depth)
        # if the variant predates the label methods.
        if hasattr(variant, "get_weight_labels"):
            weight_labels = variant.get_weight_labels(bundle)
        else:
            weight_labels = [f"layer{i+1}" for i in range(depth)]
        if hasattr(variant, "get_wandb_log_indices"):
            weight_log_idxs = variant.get_wandb_log_indices(bundle)
        else:
            weight_log_idxs = list(range(min(depth, len(weight_labels))))
        if hasattr(variant, "get_activity_labels"):
            activity_labels = variant.get_activity_labels(bundle)
        else:
            activity_labels = [f"layer{i+1}" for i in range(depth)]

    use_fused = hasattr(variant, "compute_updates_fused")

    # Accumulator for inference-diagnostic plots. Each test_every checkpoint
    # appends one entry; the wandb panels re-log all entries so trajectories
    # layer over training (not overwriting). One line per checkpoint iter.
    inference_history = []

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
            old_weights = variant.get_weight_arrays(bundle)
            old_weights = [jnp.array(w) for w in old_weights]

        # --- Step 1: Feed-forward init ---
        z_init, _ = _jit_forward_pass(variant, bundle, img_batch)

        if track_activity_norms:
            metrics.record_activity_norms_pre(z_init)

        # --- Step 2-3: Clamp output + T inference steps (JIT scan) ---
        z, _ = variant.run_inference(
            bundle, img_batch, label_batch,
            alpha=alpha, dt=inference_dt, T=inference_T,
            z_init=z_init, record_energy=False,
        )

        if use_fused:
            # Single JIT: e + ΔW + ΔV + per-layer energies from one forward+backward.
            e, delta_W, delta_V, energies_jax = _jit_compute_updates_fused(
                variant, bundle, z, img_batch, label_batch,
                alpha, v_update_rule, v_reg,
            )
            if track_layer_energy:
                metrics.energy_per_layer.append(energies_jax.tolist())
        else:
            # --- Errors at convergence (for logging & updates) ---
            e = _jit_compute_errors(variant, bundle, z, img_batch, label_batch)

            if track_layer_energy:
                energies_jax = jnp.stack(
                    [0.5 * jnp.mean(jnp.sum(el ** 2, axis=1)) for el in e]
                )
                metrics.energy_per_layer.append(energies_jax.tolist())

            # --- Step 4: ΔW from F_pc autodiff ---
            delta_W = _jit_compute_W_updates(
                variant, bundle, z, img_batch, label_batch, alpha
            )

            # --- Step 5: ΔV per configured rule (+ optional L2 on V) ---
            delta_V = _jit_compute_V_updates(
                variant, bundle, z, img_batch, label_batch, alpha,
                v_update_rule, v_reg,
            )

        # --- Step 6: Re-project (per-leaf) — optional, default on for rec-LRA
        # parity. For res-error-net experiments prefer --reproject-c 0 and
        # rely on global_clip_norm instead; per-leaf clipping destroys the
        # relative gradient magnitudes that Adam's normalizer calibrates on.
        if reproject_c is not None and reproject_c > 0.0:
            delta_W = [reproject_to_ball(dw, reproject_c) for dw in delta_W]
            delta_V = [reproject_to_ball(dv, reproject_c) for dv in delta_V]

        # --- Step 6b: Global-norm clipping on delta_W (res-error-net default).
        # Treats the full parameter gradient tree as one vector; rescales only
        # if its global Frobenius norm exceeds `global_clip_norm`. Preserves
        # cross-leaf ratios so Adam still sees a well-calibrated signal.
        grad_global_norm_pre = jnp.sqrt(sum(
            jnp.sum(jnp.square(jnp.ravel(dw))) for dw in delta_W
        ))
        if global_clip_norm is not None and global_clip_norm > 0.0:
            scale = jnp.where(
                grad_global_norm_pre > global_clip_norm,
                global_clip_norm / (grad_global_norm_pre + 1e-12),
                1.0,
            )
            delta_W = [dw * scale for dw in delta_W]
            grad_global_norm_post = grad_global_norm_pre * scale
        else:
            grad_global_norm_post = grad_global_norm_pre

        if track_grad_norms:
            grad_norms_jax = jnp.stack(
                [jnp.linalg.norm(jnp.ravel(dw)) for dw in delta_W]
            )
            metrics.grad_norms.append(grad_norms_jax.tolist())

        # --- Step 7: Apply ---
        if use_optax:
            bundle, w_opt_state, v_opt_state = variant.apply_optax_updates(
                bundle, delta_W, delta_V,
                w_optim, w_opt_state, v_optim, v_opt_state,
            )
        else:
            bundle = variant.apply_sgd_updates(
                bundle, delta_W, delta_V, param_lr, v_lr
            )

        if track_activity_norms:
            metrics.record_activity_norms_post(z)

        if track_weight_updates:
            new_weights = variant.get_weight_arrays(bundle)
            metrics.record_weight_update_norms(old_weights, new_weights)

        # --- Loss / accuracy on feed-forward output (no inference, no clamp) ---
        z_ff, _ = _jit_forward_pass(variant, bundle, img_batch)
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
            wb_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "grad_global_norm/pre_clip": float(grad_global_norm_pre),
                "grad_global_norm/post_clip": float(grad_global_norm_post),
            }

            if track_weight_updates and metrics.weight_update_norms:
                last_wu = metrics.weight_update_norms[-1]
                for idx in weight_log_idxs:
                    if idx < len(last_wu) and idx < len(weight_labels):
                        wb_metrics[f"weight_updates/{weight_labels[idx]}"] = last_wu[idx]
            if track_grad_norms and metrics.grad_norms:
                last_gn = metrics.grad_norms[-1]
                for idx in weight_log_idxs:
                    if idx < len(last_gn) and idx < len(weight_labels):
                        wb_metrics[f"grad_norms/{weight_labels[idx]}"] = last_gn[idx]
            if track_layer_energy and metrics.energy_per_layer:
                last_e = metrics.energy_per_layer[-1]
                # Energies are per PC-activity (len == L), not per leaf.
                for i, val in enumerate(last_e):
                    name = activity_labels[i] if i < len(activity_labels) else f"layer{i+1}"
                    wb_metrics[f"energy/{name}"] = val

            # V-highway norms
            V_arrays = variant.get_V_arrays(bundle)
            S = bundle["highway_indices"]
            for idx_v, i in enumerate(S):
                wb_metrics[f"v_norm/layer_{i}"] = float(
                    jnp.linalg.norm(jnp.ravel(V_arrays[idx_v]))
                )

            log_step_metrics(iter_num, wb_metrics)

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

        if ((iter_num + 1) % test_every) == 0:
            avg_acc = variant.evaluate(bundle, test_loader)
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

            # --- Inference diagnostic: F, loss, acc trajectory across T steps.
            # loss/acc are on μ^L = head(z^{L-2}) — the model's prediction from
            # the current internal state — since z[L-1] is clamped to y during
            # inference and would yield a trivial 0 loss / 100% acc.
            if use_wandb and hasattr(variant, "diagnose_inference"):
                import wandb

                diag = variant.diagnose_inference(
                    bundle, img_batch, label_batch,
                    alpha=alpha, dt=inference_dt, T=inference_T,
                    loss_type=loss_type,
                )
                inference_history.append({
                    "iter": iter_num + 1,
                    "F":    list(diag["energy"]),
                    "loss": list(diag["loss"]),
                    "acc":  list(diag["acc"]),
                })

                # Pre/post overlay: two series ("pre", "post") at the same
                # training-iter x. The vertical gap at each x visualizes the
                # drop (F, loss) or gain (acc) per checkpoint.
                iters_x = [h["iter"] for h in inference_history]
                F_pre  = [h["F"][0]    for h in inference_history]
                F_post = [h["F"][-1]   for h in inference_history]
                L_pre  = [h["loss"][0] for h in inference_history]
                L_post = [h["loss"][-1] for h in inference_history]
                A_pre  = [h["acc"][0]  for h in inference_history]
                A_post = [h["acc"][-1] for h in inference_history]

                # Heatmap trajectories: rows = checkpoints, cols = inference
                # step, color = metric. One image per metric, re-rendered each
                # checkpoint. Stays readable however many checkpoints accumulate.
                import matplotlib.pyplot as plt

                fig_F = _trajectory_heatmap(
                    inference_history, "F",
                    "Free energy across inference steps × training",
                    "F",
                )
                fig_L = _trajectory_heatmap(
                    inference_history, "loss",
                    "μ^L loss across inference steps × training",
                    "loss",
                )
                fig_A = _trajectory_heatmap(
                    inference_history, "acc",
                    "μ^L accuracy across inference steps × training",
                    "accuracy (%)",
                )

                log_step_metrics(iter_num, {
                    "inference/F_pre_post": wandb.plot.line_series(
                        xs=iters_x, ys=[F_pre, F_post], keys=["pre", "post"],
                        title="Free energy: pre vs post inference",
                        xname="training iter",
                    ),
                    "inference/loss_pre_post": wandb.plot.line_series(
                        xs=iters_x, ys=[L_pre, L_post], keys=["pre", "post"],
                        title="μ^L loss: pre vs post inference",
                        xname="training iter",
                    ),
                    "inference/acc_pre_post": wandb.plot.line_series(
                        xs=iters_x, ys=[A_pre, A_post], keys=["pre", "post"],
                        title="μ^L accuracy: pre vs post inference",
                        xname="training iter",
                    ),
                    "inference/F_heatmap":    wandb.Image(fig_F),
                    "inference/loss_heatmap": wandb.Image(fig_L),
                    "inference/acc_heatmap":  wandb.Image(fig_A),
                })
                plt.close(fig_F); plt.close(fig_L); plt.close(fig_A)

    return metrics.finalize()
