"""Cross-variant comparison plots."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.utils import ensure_dir

# Variant display settings
VARIANT_COLORS = {
    "baseline": '#1f77b4',
    "resnet": '#ff7f0e',
    "bf": '#2ca02c',
    "bf_v2": '#d62728',
    "dyt": '#9467bd',
    "dyt_v2": '#8c564b',
}

VARIANT_LABELS = {
    "baseline": "Vanilla MLP",
    "resnet": "SP ResNet",
    "bf": "BF ResNet",
    "bf_v2": "BF ResNet v2",
    "dyt": "DyT ResNet",
    "dyt_v2": "DyT ResNet v2",
}


def _save_or_log(fig, save_path, log_to_wandb, wandb_key=None):
    """Save figure to disk or log to W&B."""
    if log_to_wandb:
        from common.wandb_logger import log_figure
        key = wandb_key or os.path.splitext(os.path.basename(save_path))[0]
        log_figure(key, fig)
    else:
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_cross_variant_performance(all_results, depth, act_fn, save_path,
                                   log_to_wandb=False):
    """Plot test accuracy for all variants at a given depth.

    Args:
        all_results: {variant_name: metrics_dict}
        depth: the depth being compared
        act_fn: activation function
        save_path: where to save the plot
        log_to_wandb: if True, log to W&B instead of saving PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant_name, res in all_results.items():
        label = VARIANT_LABELS.get(variant_name, variant_name)
        color = VARIANT_COLORS.get(variant_name)
        ax.plot(
            res["test_iters"], res["test_accs"],
            'o-', markersize=4, linewidth=2,
            color=color, label=label,
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Cross-Variant Comparison — depth={depth}, {act_fn}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_log(fig, save_path, log_to_wandb)


def plot_cross_variant_final_accuracy(all_results, depths, act_fn, save_path,
                                      log_to_wandb=False):
    """Bar chart of final test accuracy across variants and depths.

    Args:
        all_results: {variant_name: {depth: metrics_dict}}
        depths: list of depths
        act_fn: activation function
        save_path: where to save
        log_to_wandb: if True, log to W&B instead of saving PNG
    """
    variants = list(all_results.keys())
    n_variants = len(variants)
    x = np.arange(len(depths))
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, v in enumerate(variants):
        final_accs = []
        for d in depths:
            if d in all_results[v] and len(all_results[v][d]["test_accs"]) > 0:
                final_accs.append(all_results[v][d]["test_accs"][-1])
            else:
                final_accs.append(0)
        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v)
        ax.bar(x + i * width, final_accs, width, label=label, color=color)

    ax.set_xlabel("Depth")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title(f"Final Accuracy Comparison ({act_fn})")
    ax.set_xticks(x + width * (n_variants - 1) / 2)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    _save_or_log(fig, save_path, log_to_wandb)
