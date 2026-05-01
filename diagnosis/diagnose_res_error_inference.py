"""Inspect how res-error-net's inference dynamics settle the per-layer
representation early in training, and how that settling depends on the
inference horizon T.

Supports both variants via the `VARIANT` constant at the top of the file:
  "mlp"      → res-error-net (MLP backbone, MNIST/FashionMNIST).
  "resnet18" → res-error-net-resnet18 (ResNet-18 backbone, CIFAR-10).

Procedure
---------
1. Train the chosen variant for N_TRAIN_ITERS_MAX iterations, snapshotting
   the parameter bundle at iter ∈ SNAPSHOT_ITERS.
2. Hold out one fixed eval batch (first batch of the test loader, no noise).
3. For each snapshot, run *a single* T=DIAGNOSE_T_MAX clamped inference pass
   on the fixed batch, recording per inference step:
     - total free energy F(z),
     - per-layer state RMS ‖z^l‖,
     - per-layer step movement ‖z^l(t) − z^l(t-1)‖,
   plus full z snapshots at logarithmic checkpoints (CHECKPOINTS).
   Slicing this trajectory at DIAGNOSE_T_HORIZONS gives multiple horizons
   without re-simulating.
4. Plots per snapshot: F(t), per-layer state RMS, per-layer step movement,
   per-layer error RMS at checkpoints, distance to final-state, cosine
   alignment to final-state.
   Cross-snapshot: F(t) overlay, settling-time heatmap.
   Cross-T: per-layer convergence-ratio heatmap (layer × T) per iter.

Run:
    python diagnose_res_error_inference.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.jax_setup  # noqa: F401  -- must come before any jax import

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from config import (
    ExperimentConfig,
    VARIANT_RES_ERROR_NET, VARIANT_RES_ERROR_NET_RESNET18,
)
from variants import get_variant
from variants.rec_lra_common import reproject_to_ball, add_input_noise
# These JIT helpers are variant-agnostic — they only call `variant.<method>`,
# so the same closures work for both the MLP and ResNet-18 variants. The eqx
# JIT cache keys on the (variant class, bundle structure) tuple, so each
# variant gets its own compiled HLO.
from variants.res_error_net import (
    _jit_forward_pass, _jit_compute_errors,
    _jit_compute_W_updates, _jit_compute_V_updates,
    _jit_inference_step_adam,
)
from common.data import get_dataloaders, set_seed


# ---------- Run config ---------------------------------------------------

# "mlp"      → res-error-net (MLP backbone, MNIST/FashionMNIST).
# "resnet18" → res-error-net-resnet18 (ResNet-18 backbone, CIFAR-10).
VARIANT             = "resnet18"

ACT_FN              = "relu"     # used by both; "relu" is typical for the CNN
ALPHA               = 0.1
INFERENCE_METHOD    = "adam"     # "euler" or "adam"
INFERENCE_DT        = 0.1       # for "adam", interpreted as the Adam learning rate
INFERENCE_T_TRAIN   = 500        # T used during the training phase
ADAM_B1             = 0.9
ADAM_B2             = 0.999
ADAM_EPS            = 1e-8
N_TRAIN_ITERS_MAX   = 100        # we stop at the largest snapshot iter
SNAPSHOT_ITERS      = (1, 10, 100)
DIAGNOSE_T_MAX      = 1000       # the long inference pass at each snapshot
DIAGNOSE_T_HORIZONS = (10, 100, 1000)
CHECKPOINTS         = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
EVAL_BATCH_SIZE     = 128
TOL_SETTLED         = 1e-3       # threshold on RMS-step-movement

# MLP-specific (ignored when VARIANT="resnet18")
DEPTH               = 12
HIGHWAY_EVERY_K     = 3
FORWARD_SKIP_EVERY  = 3

# ResNet-18-specific (ignored when VARIANT="mlp"). Default layout gives 10 PC
# activities (stem + 8 blocks + head). Each inference step is ~100× more
# expensive than the MLP case — recommend dropping DIAGNOSE_T_MAX/HORIZONS to
# e.g. 100 and (1, 10, 100), and EVAL_BATCH_SIZE to 64 for this variant.
RESNET_CHANNELS         = [64, 64, 128, 256, 512]
RESNET_BLOCKS_PER_STAGE = 2
RESNET_NORMALIZATION    = "dyt"
RESNET_DYT_INIT_ALPHA   = 0.5

SAVE_DIR = (
    "diagnostics_res_error_net/inference_evolution"
    if VARIANT == "mlp"
    else "diagnostics_res_error_net_resnet18/inference_evolution"
)


# ---------- JIT-compiled per-step probes ---------------------------------

@eqx.filter_jit
def _jit_step_and_probe(variant, bundle, z, x, y, alpha, dt):
    z_new = variant.inference_step(bundle, z, x, y, alpha, dt)
    z_free_new = [z_new[l] for l in range(len(z_new) - 1)]
    F = variant.free_energy_z(z_free_new, bundle, x, y, alpha)
    state_rms = jnp.stack([jnp.sqrt(jnp.mean(zl ** 2)) for zl in z_new])
    step_rms = jnp.stack([
        jnp.sqrt(jnp.mean((z_new[l] - z[l]) ** 2)) for l in range(len(z_new))
    ])
    return z_new, F, state_rms, step_rms


@eqx.filter_jit
def _jit_step_and_probe_adam(variant, bundle, z, m, v, t, x, y,
                             alpha, lr, b1, b2, eps):
    z_new, m_new, v_new, t_new = _jit_inference_step_adam(
        variant, bundle, z, m, v, t, x, y, alpha, lr, b1, b2, eps,
    )
    z_free_new = [z_new[l] for l in range(len(z_new) - 1)]
    F = variant.free_energy_z(z_free_new, bundle, x, y, alpha)
    state_rms = jnp.stack([jnp.sqrt(jnp.mean(zl ** 2)) for zl in z_new])
    step_rms = jnp.stack([
        jnp.sqrt(jnp.mean((z_new[l] - z[l]) ** 2)) for l in range(len(z_new))
    ])
    return z_new, m_new, v_new, t_new, F, state_rms, step_rms


@eqx.filter_jit
def _jit_initial_F(variant, bundle, z, x, y, alpha):
    z_free = [z[l] for l in range(len(z) - 1)]
    return variant.free_energy_z(z_free, bundle, x, y, alpha)


# ---------- Variant dispatch helpers --------------------------------------

def _variant_kind_to_config_id(kind):
    if kind == "mlp":
        return VARIANT_RES_ERROR_NET
    if kind == "resnet18":
        return VARIANT_RES_ERROR_NET_RESNET18
    raise ValueError(f"Unknown VARIANT: {kind!r}")


def _resolve_cfg(cfg, kind):
    """Return a flat dict of {alpha, dt, T, v_lr, v_rule, v_reg} pulled from
    the right res_*/res_resnet_* fields on cfg."""
    if kind == "mlp":
        return dict(
            alpha=cfg.res_alpha,
            dt=cfg.res_inference_dt,
            T=cfg.res_inference_T,
            v_lr=cfg.res_v_lr,
            v_rule=cfg.res_v_update_rule,
            v_reg=cfg.res_v_reg,
        )
    return dict(
        alpha=cfg.res_resnet_alpha,
        dt=cfg.res_resnet_inference_dt,
        T=cfg.res_resnet_inference_T,
        v_lr=cfg.res_resnet_v_lr,
        v_rule=cfg.res_resnet_v_update_rule,
        v_reg=cfg.res_resnet_v_reg,
    )


def _create_model_kwargs(cfg, kind):
    """Variant-specific kwargs to hand to variant.create_model."""
    if kind == "mlp":
        return dict(
            highway_every_k=HIGHWAY_EVERY_K,
            forward_skip_every=FORWARD_SKIP_EVERY,
            v_init_scale=cfg.res_v_init_scale,
            res_init_scheme=cfg.res_init_scheme,
        )
    return dict(
        input_shape=(3, 32, 32),
        resnet_channels=RESNET_CHANNELS,
        blocks_per_stage=RESNET_BLOCKS_PER_STAGE,
        normalization=RESNET_NORMALIZATION,
        dyt_init_alpha=RESNET_DYT_INIT_ALPHA,
        highway_include_stem=cfg.res_resnet_highway_include_stem,
        v_init_scale=cfg.res_resnet_v_init_scale,
    )


# ---------- Manual training loop with snapshotting ------------------------

def train_with_snapshots(variant, bundle, cfg, vcfg):
    """Run a stripped-down version of train_res_error_net for the first
    N_TRAIN_ITERS_MAX iterations and capture the bundle at SNAPSHOT_ITERS.
    `vcfg` is the variant-specific subset returned by `_resolve_cfg`."""
    set_seed(cfg.seed)

    w_optim = optax.adamw(cfg.param_lr, weight_decay=cfg.weight_decay, eps=1e-12)
    v_optim = optax.adamw(vcfg["v_lr"], weight_decay=cfg.weight_decay, eps=1e-12)
    w_opt_state = variant.init_w_optim_states(bundle, w_optim)
    v_opt_state = variant.init_v_optim_states(bundle, v_optim)

    train_loader, _ = get_dataloaders(cfg.dataset, cfg.batch_size, use_zca=False)
    noise_key = jr.PRNGKey(cfg.seed + 1)

    snapshots = {}
    losses = []

    data_iter = iter(train_loader)
    for it in range(1, N_TRAIN_ITERS_MAX + 1):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)

        img_batch = img_batch.numpy()
        label_batch = label_batch.numpy()

        if cfg.input_noise_sigma and cfg.input_noise_sigma > 0.0:
            noise_key, sub = jr.split(noise_key)
            img_batch = np.asarray(
                add_input_noise(sub, jnp.asarray(img_batch), cfg.input_noise_sigma)
            )

        z_init, _ = _jit_forward_pass(variant, bundle, img_batch)
        z, _ = variant.run_inference(
            bundle, img_batch, label_batch,
            alpha=vcfg["alpha"], dt=vcfg["dt"], T=vcfg["T"],
            z_init=z_init, record_energy=False,
        )

        delta_W = _jit_compute_W_updates(
            variant, bundle, z, img_batch, label_batch, vcfg["alpha"]
        )
        delta_V = _jit_compute_V_updates(
            variant, bundle, z, img_batch, label_batch, vcfg["alpha"],
            vcfg["v_rule"], vcfg["v_reg"],
        )

        if cfg.reproject_c is not None and cfg.reproject_c > 0.0:
            delta_W = [reproject_to_ball(dw, cfg.reproject_c) for dw in delta_W]
            delta_V = [reproject_to_ball(dv, cfg.reproject_c) for dv in delta_V]

        if cfg.global_clip_norm is not None and cfg.global_clip_norm > 0.0:
            g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(jnp.ravel(dw))) for dw in delta_W))
            scale = jnp.where(g_norm > cfg.global_clip_norm,
                              cfg.global_clip_norm / (g_norm + 1e-12), 1.0)
            delta_W = [dw * scale for dw in delta_W]

        bundle, w_opt_state, v_opt_state = variant.apply_optax_updates(
            bundle, delta_W, delta_V,
            w_optim, w_opt_state, v_optim, v_opt_state,
        )

        # Cheap loss readout (feed-forward MSE)
        z_ff, _ = _jit_forward_pass(variant, bundle, img_batch)
        train_loss = float(jnp.mean((z_ff[-1] - label_batch) ** 2))
        losses.append(train_loss)

        if it in SNAPSHOT_ITERS:
            # bundle is rebuilt each apply_optax_updates → the old reference
            # is preserved automatically. Shallow-copy the dict to be safe.
            snapshots[it] = {**bundle, "model": list(bundle["model"]),
                             "V_list": list(bundle["V_list"])}
            print(f"  snapshot @ iter {it:>3d}  loss={train_loss:.4f}")

    return snapshots, np.array(losses)


# ---------- Per-snapshot inference probe ---------------------------------

def probe_inference(variant, bundle, x_batch, y_batch, alpha, dt, T_max,
                    method="euler", b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS):
    """Run T_max clamped inference steps; record per-step diagnostics and a
    set of full-z checkpoints. `method` selects euler or adam-on-z."""
    L = bundle["depth"]

    z_init, _ = _jit_forward_pass(variant, bundle, x_batch)
    z = list(z_init)
    z[L - 1] = jnp.asarray(y_batch)

    F0 = float(_jit_initial_F(variant, bundle, z, x_batch, y_batch, alpha))
    state_rms_0 = np.array([float(jnp.sqrt(jnp.mean(zl ** 2))) for zl in z])

    F_traj = np.zeros(T_max + 1, dtype=np.float64)
    state_rms_traj = np.zeros((T_max + 1, L), dtype=np.float64)
    step_rms_traj = np.zeros((T_max + 1, L), dtype=np.float64)
    F_traj[0] = F0
    state_rms_traj[0] = state_rms_0
    # step at t=0 is undefined; leave zeros.

    z_checkpoints = {0: [np.asarray(zl) for zl in z]}

    if method == "adam":
        m = [jnp.zeros_like(z[l]) for l in range(L - 1)]
        v_buf = [jnp.zeros_like(z[l]) for l in range(L - 1)]
        t_count = jnp.zeros((), dtype=jnp.int32)

    for t in range(1, T_max + 1):
        if method == "adam":
            z, m, v_buf, t_count, F, state_rms, step_rms = (
                _jit_step_and_probe_adam(
                    variant, bundle, z, m, v_buf, t_count,
                    x_batch, y_batch, alpha, dt, b1, b2, eps,
                )
            )
        else:
            z, F, state_rms, step_rms = _jit_step_and_probe(
                variant, bundle, z, x_batch, y_batch, alpha, dt
            )
        F_traj[t] = float(F)
        state_rms_traj[t] = np.asarray(state_rms)
        step_rms_traj[t] = np.asarray(step_rms)
        if t in CHECKPOINTS:
            z_checkpoints[t] = [np.asarray(zl) for zl in z]

    # error norms at checkpoints
    err_at_ckpt = {}
    for t, z_ck in z_checkpoints.items():
        e = _jit_compute_errors(
            variant, bundle, [jnp.asarray(zl) for zl in z_ck], x_batch, y_batch
        )
        err_at_ckpt[t] = np.array([float(jnp.sqrt(jnp.mean(el ** 2))) for el in e])

    # cosine alignment & dist to t=T_max state (per layer, RMS-normalised)
    z_final = z_checkpoints[T_max]
    cos_at_ckpt = {}
    dist_at_ckpt = {}
    for t, z_ck in z_checkpoints.items():
        cos_l, dist_l = [], []
        for l in range(L):
            a = z_ck[l].reshape(-1)
            b = z_final[l].reshape(-1)
            num = float(np.dot(a, b))
            den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            cos_l.append(num / den)
            dist_l.append(float(np.sqrt(np.mean((z_ck[l] - z_final[l]) ** 2))))
        cos_at_ckpt[t] = np.array(cos_l)
        dist_at_ckpt[t] = np.array(dist_l)

    # convergence ratio at canonical horizons (layer × T)
    horizon_conv_ratio = {}
    for T in DIAGNOSE_T_HORIZONS:
        if T not in z_checkpoints:
            T_use = min(z_checkpoints.keys(), key=lambda k: abs(k - T))
        else:
            T_use = T
        z_ck = z_checkpoints[T_use]
        ratio = []
        for l in range(L):
            num = float(np.sqrt(np.mean((z_ck[l] - z_final[l]) ** 2)))
            den = float(np.sqrt(np.mean(z_final[l] ** 2)) + 1e-12)
            ratio.append(num / den)
        horizon_conv_ratio[T] = np.array(ratio)

    return {
        "F": F_traj,
        "state_rms": state_rms_traj,
        "step_rms": step_rms_traj,
        "err_at_ckpt": err_at_ckpt,
        "cos_at_ckpt": cos_at_ckpt,
        "dist_at_ckpt": dist_at_ckpt,
        "horizon_conv_ratio": horizon_conv_ratio,
    }


# ---------- Plots --------------------------------------------------------

def _layer_colors(L):
    return plt.cm.viridis(np.linspace(0.05, 0.95, L))


def plot_per_snapshot(probe, iter_id, save_dir, layer_labels=None,
                      title_prefix="res-error-net"):
    L = probe["state_rms"].shape[1]
    if layer_labels is None:
        layer_labels = [f"l={l}" for l in range(L)]
    colors = _layer_colors(L)
    T_max = probe["state_rms"].shape[0] - 1
    t_axis = np.arange(T_max + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Free energy F(t)
    ax = axes[0, 0]
    F = np.asarray(probe["F"])
    ax.semilogy(t_axis, np.maximum(F, 1e-12))
    for T in DIAGNOSE_T_HORIZONS:
        ax.axvline(T, color="tab:red", linestyle="--", alpha=0.4,
                   label=f"T={T}")
    ax.set_xlabel("inference step t")
    ax.set_ylabel("F(z)  (log)")
    ax.set_title(f"free energy  (iter {iter_id})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # 2. Per-layer state RMS vs t
    ax = axes[0, 1]
    for l in range(L):
        ax.plot(t_axis, probe["state_rms"][:, l], color=colors[l],
                label=layer_labels[l], lw=1.0)
    ax.set_xlabel("inference step t")
    ax.set_ylabel("RMS ‖z^l‖")
    ax.set_title("per-layer state RMS vs t")
    ax.legend(ncol=2, fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)

    # 3. Per-layer step movement (semilogy)
    ax = axes[0, 2]
    move = probe["step_rms"][1:]   # t=0 movement undefined
    for l in range(L):
        ax.semilogy(np.arange(1, T_max + 1),
                    np.maximum(move[:, l], 1e-12), color=colors[l], lw=1.0)
    ax.axhline(TOL_SETTLED, color="k", linestyle=":", alpha=0.5,
               label=f"tol={TOL_SETTLED:.0e}")
    ax.set_xlabel("inference step t")
    ax.set_ylabel("RMS step ‖Δz^l‖  (log)")
    ax.set_title("settling speed per layer")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # 4. Per-layer error RMS at checkpoints
    ax = axes[1, 0]
    ckpts = sorted(probe["err_at_ckpt"].keys())
    err = np.stack([probe["err_at_ckpt"][t] for t in ckpts], axis=0)
    for l in range(L):
        ax.plot(ckpts, err[:, l], "o-", color=colors[l], lw=1.0, ms=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("inference step t  (symlog)")
    ax.set_ylabel("RMS ‖e^l‖  (log)")
    ax.set_title("per-layer prediction error at checkpoints")
    ax.grid(True, alpha=0.3, which="both")

    # 5. Distance to final state
    ax = axes[1, 1]
    dist = np.stack([probe["dist_at_ckpt"][t] for t in ckpts], axis=0)
    for l in range(L):
        ax.plot(ckpts, np.maximum(dist[:, l], 1e-12), "o-",
                color=colors[l], lw=1.0, ms=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("inference step t  (symlog)")
    ax.set_ylabel("RMS ‖z^l(t) − z^l(T_max)‖  (log)")
    ax.set_title("distance to final state (T_max=%d)" % T_max)
    ax.grid(True, alpha=0.3, which="both")

    # 6. Cosine alignment to final state
    ax = axes[1, 2]
    cos = np.stack([probe["cos_at_ckpt"][t] for t in ckpts], axis=0)
    im = ax.imshow(cos.T, aspect="auto", origin="lower",
                   vmin=-1.0, vmax=1.0, cmap="RdBu_r",
                   extent=[0, len(ckpts), -0.5, L - 0.5])
    ax.set_xticks(np.arange(len(ckpts)) + 0.5)
    ax.set_xticklabels(ckpts, rotation=45, fontsize=7)
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_xlabel("inference step t")
    ax.set_title("cos(z^l(t), z^l(T_max))")
    plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle(
        f"{title_prefix} inference dynamics  —  snapshot @ iter {iter_id}",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(save_dir, f"inference_iter{iter_id:03d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_cross_iter(probes, save_dir, layer_labels=None):
    iters = sorted(probes.keys())
    L = probes[iters[0]]["state_rms"].shape[1]
    if layer_labels is None:
        layer_labels = [f"l={l}" for l in range(L)]
    T_max = probes[iters[0]]["state_rms"].shape[0] - 1
    t_axis = np.arange(T_max + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. F(t) overlay
    ax = axes[0]
    for it in iters:
        F = probes[it]["F"]
        ax.semilogy(t_axis, np.maximum(F, 1e-12), label=f"iter {it}")
    for T in DIAGNOSE_T_HORIZONS:
        ax.axvline(T, color="tab:red", linestyle="--", alpha=0.3)
    ax.set_xlabel("inference step t")
    ax.set_ylabel("F(z)  (log)")
    ax.set_title("free energy vs t  —  across snapshots")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # 2. Settling time per layer per iter
    ax = axes[1]
    settle_grid = np.full((len(iters), L), np.nan)
    for i, it in enumerate(iters):
        move = probes[it]["step_rms"][1:]
        for l in range(L):
            below = move[:, l] < TOL_SETTLED
            if below.any():
                streak, t_first = 0, None
                for tt, b in enumerate(below, start=1):
                    if b:
                        streak += 1
                        if streak >= 5:
                            t_first = tt - 4
                            break
                    else:
                        streak = 0
                if t_first is not None:
                    settle_grid[i, l] = t_first
    im = ax.imshow(settle_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(layer_labels, fontsize=7, rotation=45)
    ax.set_yticks(np.arange(len(iters)))
    ax.set_yticklabels([f"iter {it}" for it in iters], fontsize=8)
    ax.set_xlabel("layer")
    ax.set_title(f"settling time t* (RMS step < {TOL_SETTLED:.0e}, 5-step streak)")
    plt.colorbar(im, ax=ax, fraction=0.04, label="t*")
    for i in range(len(iters)):
        for l in range(L):
            v = settle_grid[i, l]
            txt = "—" if np.isnan(v) else f"{int(v)}"
            ax.text(l, i, txt, ha="center", va="center",
                    color="white" if (np.isnan(v) or v > np.nanmean(settle_grid))
                    else "black", fontsize=6)

    # 3. mid-layer step movement overlay
    ax = axes[2]
    l_mid = L // 2
    mid_label = layer_labels[l_mid]
    for it in iters:
        move = probes[it]["step_rms"][1:, l_mid]
        ax.semilogy(np.arange(1, T_max + 1),
                    np.maximum(move, 1e-12), label=f"iter {it}")
    ax.axhline(TOL_SETTLED, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("inference step t")
    ax.set_ylabel(f"RMS step ‖Δz^{{{mid_label}}}‖  (log)")
    ax.set_title(f"mid-layer ({mid_label}) settling — across snapshots")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("cross-snapshot inference dynamics", fontsize=12)
    fig.tight_layout()
    out = os.path.join(save_dir, "cross_iter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_cross_T(probes, save_dir, layer_labels=None):
    iters = sorted(probes.keys())
    L = probes[iters[0]]["state_rms"].shape[1]
    if layer_labels is None:
        layer_labels = [f"l={l}" for l in range(L)]

    fig, axes = plt.subplots(1, len(iters), figsize=(5 * len(iters), 4.5),
                              sharey=True)
    if len(iters) == 1:
        axes = [axes]

    im = None
    for ax, it in zip(axes, iters):
        ratios = np.stack(
            [probes[it]["horizon_conv_ratio"][T] for T in DIAGNOSE_T_HORIZONS],
            axis=0,
        )
        # log-scale colormap on convergence ratio (lower = more settled)
        im = ax.imshow(ratios, aspect="auto", origin="lower", cmap="magma_r",
                       norm=plt.cm.colors.LogNorm(vmin=1e-4, vmax=1.0))
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(layer_labels, fontsize=7, rotation=45)
        ax.set_yticks(np.arange(len(DIAGNOSE_T_HORIZONS)))
        ax.set_yticklabels([f"T={T}" for T in DIAGNOSE_T_HORIZONS])
        ax.set_xlabel("layer")
        ax.set_title(f"iter {it}")
        for i, T in enumerate(DIAGNOSE_T_HORIZONS):
            for l in range(L):
                ax.text(l, i, f"{ratios[i, l]:.2g}",
                        ha="center", va="center", fontsize=6,
                        color="white" if ratios[i, l] > 0.05 else "black")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("‖z^l(T) − z^l(T_max)‖ / ‖z^l(T_max)‖   (log)")
    fig.suptitle("how converged is each layer at horizon T  (lower = more settled)",
                 fontsize=12)
    out = os.path.join(save_dir, "cross_T.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------- Main ----------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cfg_variant = _variant_kind_to_config_id(VARIANT)

    # Variant-specific overrides for ExperimentConfig.from_variant. The MLP
    # variant trains on MNIST by default; the ResNet-18 variant requires
    # CIFAR-10 (3×32×32) input.
    if VARIANT == "mlp":
        cfg = ExperimentConfig.from_variant(
            cfg_variant,
            depths=[DEPTH],
            act_fns=[ACT_FN],
            n_train_iters=N_TRAIN_ITERS_MAX,
            res_alpha=ALPHA,
            res_inference_T=INFERENCE_T_TRAIN,
            res_inference_dt=INFERENCE_DT,
            res_inference_method=INFERENCE_METHOD,
            res_highway_every_k=HIGHWAY_EVERY_K,
            res_forward_skip_every=FORWARD_SKIP_EVERY,
        )
    else:
        cfg = ExperimentConfig.from_variant(
            cfg_variant,
            dataset="CIFAR10",
            act_fns=[ACT_FN],
            n_train_iters=N_TRAIN_ITERS_MAX,
            res_resnet_alpha=ALPHA,
            res_resnet_inference_T=INFERENCE_T_TRAIN,
            res_resnet_inference_dt=INFERENCE_DT,
            res_resnet_inference_method=INFERENCE_METHOD,
            res_resnet_channels=RESNET_CHANNELS,
            res_resnet_blocks_per_stage=RESNET_BLOCKS_PER_STAGE,
            res_resnet_normalization=RESNET_NORMALIZATION,
            res_resnet_dyt_init_alpha=RESNET_DYT_INIT_ALPHA,
        )
    set_seed(cfg.seed)
    key = jr.PRNGKey(cfg.seed)

    variant = get_variant(cfg.variant)

    create_kwargs = _create_model_kwargs(cfg, VARIANT)
    bundle = variant.create_model(
        key,
        depth=DEPTH if VARIANT == "mlp" else None,
        width=cfg.width if VARIANT == "mlp" else None,
        act_fn=ACT_FN,
        input_dim=cfg.input_dim, output_dim=cfg.output_dim,
        inference_method=INFERENCE_METHOD,
        inference_b1=ADAM_B1,
        inference_b2=ADAM_B2,
        inference_eps=ADAM_EPS,
        **create_kwargs,
    )
    title_prefix = (
        "res-error-net" if VARIANT == "mlp" else "res-error-net-resnet18"
    )
    print(f"created {title_prefix}  L={bundle['depth']}  "
          f"act={ACT_FN}  S={bundle['highway_indices']}  "
          f"inference={INFERENCE_METHOD}  dt/lr={INFERENCE_DT}  "
          f"dataset={cfg.dataset}")

    vcfg = _resolve_cfg(cfg, VARIANT)
    print(f"\n=== training for {N_TRAIN_ITERS_MAX} iters with T={vcfg['T']} ===")
    snapshots, losses = train_with_snapshots(variant, bundle, cfg, vcfg)

    # Fixed eval batch (test loader, first batch — deterministic).
    _, test_loader = get_dataloaders(cfg.dataset, EVAL_BATCH_SIZE, use_zca=False)
    img_eval, lbl_eval = next(iter(test_loader))
    img_eval = img_eval.numpy()
    lbl_eval = lbl_eval.numpy()
    print(f"\nfixed eval batch  shape={img_eval.shape}  labels={lbl_eval.shape}")

    layer_labels = (
        variant.get_activity_labels(bundle)
        if hasattr(variant, "get_activity_labels")
        else [f"l={l}" for l in range(bundle["depth"])]
    )

    probes = {}
    print(f"\n=== probing inference for T={DIAGNOSE_T_MAX} steps at each snapshot ===")
    for it in sorted(snapshots.keys()):
        print(f"\n-- snapshot @ iter {it} --")
        probe = probe_inference(
            variant, snapshots[it], img_eval, lbl_eval,
            alpha=ALPHA, dt=INFERENCE_DT, T_max=DIAGNOSE_T_MAX,
            method=INFERENCE_METHOD,
        )
        probes[it] = probe

        F = probe["F"]
        # Indices may be > T_max if user shortened DIAGNOSE_T_MAX; guard them.
        def _F_at(t):
            return F[t] if t < len(F) else F[-1]
        print(f"   F(0)={F[0]:.4f}  "
              f"F({min(10, DIAGNOSE_T_MAX)})={_F_at(min(10, DIAGNOSE_T_MAX)):.4f}  "
              f"F({DIAGNOSE_T_MAX})={F[-1]:.4f}")
        for T in DIAGNOSE_T_HORIZONS:
            r = probe["horizon_conv_ratio"][T]
            print(f"   T={T:>4d}  per-layer conv-ratio "
                  f"(min/mean/max) = {r.min():.2e} / {r.mean():.2e} / {r.max():.2e}")

        plot_per_snapshot(probe, it, SAVE_DIR,
                          layer_labels=layer_labels,
                          title_prefix=title_prefix)

    plot_cross_iter(probes, SAVE_DIR, layer_labels=layer_labels)
    plot_cross_T(probes, SAVE_DIR, layer_labels=layer_labels)

    # Persist raw arrays for offline analysis.
    raw_path = os.path.join(SAVE_DIR, "probes.npz")
    flat = {}
    for it, p in probes.items():
        flat[f"iter{it}_F"]         = p["F"]
        flat[f"iter{it}_state_rms"] = p["state_rms"]
        flat[f"iter{it}_step_rms"]  = p["step_rms"]
        for T in DIAGNOSE_T_HORIZONS:
            flat[f"iter{it}_conv_ratio_T{T}"] = p["horizon_conv_ratio"][T]
    flat["train_loss"] = losses
    np.savez(raw_path, **flat)
    print(f"\nsaved raw arrays -> {raw_path}")
    print(f"all plots in    -> {SAVE_DIR}/")


if __name__ == "__main__":
    main()
