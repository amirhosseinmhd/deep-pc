"""All plotting functions for PC experiments."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.utils import ensure_dir, selected_layer_indices

# Paper-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

DEPTH_COLORS = {5: '#d62728', 10: '#1f77b4', 20: '#ff7f0e', 40: '#2ca02c'}
DEPTH_MARKERS = {5: 'v', 10: 'o', 20: 's', 40: '^'}


def _save_or_log(fig, save_path, log_to_wandb, wandb_key=None):
    """Save figure to disk or log to W&B (or both)."""
    if log_to_wandb:
        from common.wandb_logger import log_figure
        key = wandb_key or os.path.splitext(os.path.basename(save_path))[0]
        log_figure(key, fig)
    else:
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Performance curves
# ---------------------------------------------------------------------------
def plot_performance_curves(results_dict, title, save_path, log_to_wandb=False):
    """Plot test accuracy vs training iteration for multiple depths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, res in sorted(results_dict.items()):
        ax.plot(
            res["test_iters"], res["test_accs"],
            marker=DEPTH_MARKERS.get(depth, 'o'), markersize=5,
            color=DEPTH_COLORS.get(depth), linewidth=2,
            label=f"depth={depth}"
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_train_loss(results_dict, title, save_path, log_scale=True,
                    log_to_wandb=False):
    """Plot training loss vs iteration for multiple depths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, res in sorted(results_dict.items()):
        ax.plot(
            res["train_losses"], alpha=0.7,
            color=DEPTH_COLORS.get(depth), linewidth=1.5,
            label=f"depth={depth}"
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Train loss (MSE)")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Condition numbers
# ---------------------------------------------------------------------------
def plot_condition_numbers(results_dict, depths, title, save_path,
                           log_to_wandb=False):
    """Plot condition number vs depth for multiple configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':']
    for i, (label, conds) in enumerate(results_dict.items()):
        ax.plot(
            depths, conds,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            markersize=7, linewidth=2, label=label
        )
    ax.set_xlabel("Depth")
    ax.set_ylabel(r"$\kappa(\mathbf{H}_z)$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_condition_bar(results_dict, depths, title, save_path,
                       log_to_wandb=False):
    """Bar chart of condition numbers grouped by depth."""
    configs = list(results_dict.keys())
    n_configs = len(configs)
    x = np.arange(len(depths))
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        vals = results_dict[cfg]
        ax.bar(x + i * width, vals, width, label=cfg)

    ax.set_xlabel("Depth")
    ax.set_ylabel(r"$\kappa(\mathbf{H}_z)$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xticks(x + width * (n_configs - 1) / 2)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Weight update norms
# ---------------------------------------------------------------------------
def plot_weight_updates(update_norms, layer_labels, title, save_path,
                        log_to_wandb=False):
    """Plot weight update norms vs training iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(layer_labels)))
    for i, label in enumerate(layer_labels):
        ax.plot(
            update_norms[:, i], color=cmap[i],
            linewidth=1.5, alpha=0.8, label=label
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\|\Delta W_\ell\|_F$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_weight_updates_summary(summary_dict, depths, title, save_path,
                                log_to_wandb=False):
    """Plot mean weight update norm vs depth."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for act_fn, vals in summary_dict.items():
        ax.plot(
            depths, vals, 'o-', markersize=8, linewidth=2,
            label=act_fn
        )
    ax.set_xlabel("Depth")
    ax.set_ylabel(r"Mean $\|\Delta W\|_F$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Latent / activity norms
# ---------------------------------------------------------------------------
def plot_latent_norms_vs_layer(norms_dict, title, save_path, log_scale=True,
                               log_to_wandb=False):
    """Plot activity L2 norm vs layer index."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, norms in norms_dict.items():
        ax.plot(
            range(1, len(norms) + 1), norms,
            'o-', markersize=4, linewidth=1.5, label=label
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\|z_\ell\|_2$")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_latent_norms_vs_training(norms_array, layer_labels, title, save_path,
                                  sample_every=10, log_to_wandb=False):
    """Plot activity norms vs training iteration for selected layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(layer_labels)))
    iters = np.arange(norms_array.shape[0])
    sampled = iters[::sample_every]
    for i, (lidx, label) in enumerate(layer_labels):
        ax.plot(
            sampled, norms_array[sampled, lidx],
            color=cmap[i], linewidth=1.5, label=label
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\|z_\ell\|_2$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_latent_norms_across_depths(depth_norms_dict, title, save_path,
                                    log_to_wandb=False):
    """Plot init activity norms across layers for multiple depths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, norms in sorted(depth_norms_dict.items()):
        layers = np.linspace(0, 1, len(norms))
        ax.plot(
            layers, norms, 'o-', markersize=3, linewidth=1.5,
            color=DEPTH_COLORS.get(depth), label=f"depth={depth}"
        )
    ax.set_xlabel("Relative layer position")
    ax.set_ylabel(r"$\|z_\ell\|_2$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Gradient norms
# ---------------------------------------------------------------------------
def plot_grad_norms(grad_norms, layer_labels, title, save_path,
                    log_to_wandb=False):
    """Plot gradient norms vs training iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(layer_labels)))
    for i, label in enumerate(layer_labels):
        ax.plot(
            grad_norms[:, i], color=cmap[i],
            linewidth=1.5, alpha=0.8, label=label
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\|\nabla_{W_\ell} F\|_F$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Per-layer energy
# ---------------------------------------------------------------------------
def plot_energy_per_layer(energy_array, layer_labels, title, save_path,
                          log_to_wandb=False):
    """Plot per-layer PC energy vs training iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(layer_labels)))
    for i, label in enumerate(layer_labels):
        ax.plot(
            energy_array[:, i], color=cmap[i],
            linewidth=1.5, alpha=0.8, label=label
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Per-layer energy")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


# ---------------------------------------------------------------------------
# Orchestrator: generate all plots from a single results dict
# ---------------------------------------------------------------------------
def generate_all_plots(depth_results, cfg, act_fn, log_to_wandb=False):
    """Generate all applicable plots for a variant + activation function.

    Args:
        depth_results: {depth: metrics_dict} from train_and_record
        cfg: ExperimentConfig
        act_fn: activation function string
        log_to_wandb: if True, log figures to W&B instead of saving PNGs
    """
    plot_dir = ensure_dir(os.path.join(cfg.results_dir, "plots", act_fn))
    variant_name = cfg.variant

    # Performance
    plot_performance_curves(
        depth_results,
        f"{variant_name} — Test Accuracy ({act_fn})",
        os.path.join(plot_dir, "performance.png"),
        log_to_wandb=log_to_wandb,
    )
    plot_train_loss(
        depth_results,
        f"{variant_name} — Train Loss ({act_fn})",
        os.path.join(plot_dir, "train_loss.png"),
        log_to_wandb=log_to_wandb,
    )

    # Check what metrics are available from the first depth's results
    sample_res = next(iter(depth_results.values()))

    # Weight updates
    if "weight_update_norms" in sample_res:
        for depth, res in depth_results.items():
            norms = res["weight_update_norms"]
            n_layers = norms.shape[1]
            idxs = selected_layer_indices(n_layers)
            labels = [f"Layer {i+1}" for i in idxs]
            plot_weight_updates(
                norms[:, idxs], labels,
                f"{variant_name} — Weight Updates (d={depth}, {act_fn})",
                os.path.join(plot_dir, f"weight_updates_d{depth}.png"),
                log_to_wandb=log_to_wandb,
            )

        # Summary across depths
        depths = sorted(depth_results.keys())
        mean_updates = [
            float(np.mean(depth_results[d]["weight_update_norms"]))
            for d in depths
        ]
        plot_weight_updates_summary(
            {act_fn: mean_updates}, depths,
            f"{variant_name} — Mean Weight Update vs Depth",
            os.path.join(plot_dir, "weight_updates_summary.png"),
            log_to_wandb=log_to_wandb,
        )

    # Activity norms
    if "activity_norms_init" in sample_res:
        for depth, res in depth_results.items():
            init_norms = res["activity_norms_init"]
            post_norms = res["activity_norms_post"]

            # Norms vs layer at iteration 0
            plot_latent_norms_vs_layer(
                {
                    "Pre-inference (iter 0)": init_norms[0],
                    "Post-inference (iter 0)": post_norms[0],
                },
                f"{variant_name} — Activity Norms vs Layer (d={depth}, {act_fn})",
                os.path.join(plot_dir, f"latent_norms_layer_d{depth}.png"),
                log_to_wandb=log_to_wandb,
            )

            # Norms vs training for selected layers
            n_layers = init_norms.shape[1]
            idxs = selected_layer_indices(n_layers)
            layer_labels = [(i, f"Layer {i+1}") for i in idxs]
            plot_latent_norms_vs_training(
                post_norms, layer_labels,
                f"{variant_name} — Activity Norms vs Training (d={depth}, {act_fn})",
                os.path.join(plot_dir, f"latent_norms_training_d{depth}.png"),
                log_to_wandb=log_to_wandb,
            )

        # Across depths
        depths = sorted(depth_results.keys())
        depth_init_norms = {
            d: depth_results[d]["activity_norms_init"][0]
            for d in depths
        }
        plot_latent_norms_across_depths(
            depth_init_norms,
            f"{variant_name} — Init Norms Across Depths ({act_fn})",
            os.path.join(plot_dir, "latent_norms_across_depths.png"),
            log_to_wandb=log_to_wandb,
        )

    # Gradient norms
    if "grad_norms" in sample_res:
        for depth, res in depth_results.items():
            gnorms = res["grad_norms"]
            n_layers = gnorms.shape[1]
            idxs = selected_layer_indices(n_layers)
            labels = [f"Layer {i+1}" for i in idxs]
            plot_grad_norms(
                gnorms[:, idxs], labels,
                f"{variant_name} — Gradient Norms (d={depth}, {act_fn})",
                os.path.join(plot_dir, f"grad_norms_d{depth}.png"),
                log_to_wandb=log_to_wandb,
            )

    # Per-layer energy
    if "energy_per_layer" in sample_res:
        for depth, res in depth_results.items():
            energy = res["energy_per_layer"]
            n_layers = energy.shape[1]
            idxs = selected_layer_indices(n_layers)
            labels = [f"Layer {i+1}" for i in idxs]
            plot_energy_per_layer(
                energy[:, idxs], labels,
                f"{variant_name} — Per-Layer Energy (d={depth}, {act_fn})",
                os.path.join(plot_dir, f"energy_per_layer_d{depth}.png"),
                log_to_wandb=log_to_wandb,
            )
