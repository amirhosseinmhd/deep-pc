"""Plotting utilities for μPC experiments."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

DEPTH_COLORS = {10: '#1f77b4', 20: '#ff7f0e', 40: '#2ca02c'}
DEPTH_MARKERS = {10: 'o', 20: 's', 40: '^'}


# ---------------------------------------------------------------------------
# Experiment 1: Performance curves
# ---------------------------------------------------------------------------
def plot_performance_curves(results_dict, title, save_path):
    """Plot test accuracy vs training iteration for multiple depths.

    results_dict: {depth: {"test_accs": [...], "test_iters": [...]}}
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_train_loss(results_dict, title, save_path, log_scale=True):
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Experiment 2: Condition numbers
# ---------------------------------------------------------------------------
def plot_condition_numbers(results_dict, depths, title, save_path):
    """Plot condition number vs depth for multiple configurations.

    results_dict: {config_label: [cond_for_depth0, cond_for_depth1, ...]}
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_condition_bar(results_dict, depths, title, save_path):
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Experiment 3: Weight update norms
# ---------------------------------------------------------------------------
def plot_weight_updates(update_norms, layer_labels, title, save_path):
    """Plot weight update norms vs training iteration.

    update_norms: array of shape (n_iters, n_layers)
    layer_labels: list of layer label strings
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_weight_updates_summary(summary_dict, depths, title, save_path):
    """Plot mean weight update norm vs depth for different activations.

    summary_dict: {act_fn: [mean_update_depth0, mean_update_depth1, ...]}
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Experiment 4: Latent / activity norms
# ---------------------------------------------------------------------------
def plot_latent_norms_vs_layer(norms_dict, title, save_path, log_scale=True):
    """Plot activity L2 norm vs layer index.

    norms_dict: {label: array_of_norms_per_layer}
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latent_norms_vs_training(norms_array, layer_labels, title, save_path,
                                  sample_every=10):
    """Plot activity norms vs training iteration for selected layers.

    norms_array: shape (n_iters, n_layers_total)
    layer_labels: list of (layer_idx, label_str)
    """
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latent_norms_across_depths(depth_norms_dict, title, save_path):
    """Plot init activity norms across layers for multiple depths.

    depth_norms_dict: {depth: norms_per_layer_array}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, norms in sorted(depth_norms_dict.items()):
        layers = np.linspace(0, 1, len(norms))  # normalise x-axis
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
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")
