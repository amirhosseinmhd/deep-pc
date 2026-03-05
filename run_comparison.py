#!/usr/bin/env python
"""Generate cross-variant comparison plots from saved results.

Loads .npy files from results/<variant>/<act_fn>_d<depth>/ directories
and generates comparison plots.

Usage:
    python run_comparison.py
    python run_comparison.py --variant resnet bf dyt --depths 10 20
    python run_comparison.py --no-wandb   # save PNGs locally instead
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import numpy as np

from config import ExperimentConfig, ALL_VARIANTS, RESULTS_DIR
from common.utils import ensure_dir
from plotting.comparison import (
    plot_cross_variant_performance,
    plot_cross_variant_final_accuracy,
)


def load_results(variant_name, act_fn, depth):
    """Load saved metrics for a variant/act_fn/depth combination."""
    results_dir = os.path.join(RESULTS_DIR, variant_name, f"{act_fn}_d{depth}")
    if not os.path.exists(results_dir):
        return None

    res = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".npy"):
            key = fname[:-4]  # strip .npy
            res[key] = np.load(os.path.join(results_dir, fname), allow_pickle=True)
    return res if res else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-variant comparison plots"
    )
    parser.add_argument(
        "--variant", choices=ALL_VARIANTS, nargs="+",
        default=ALL_VARIANTS,
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--act-fns", nargs="+", default=["relu"])
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (save PNGs locally)",
    )
    parser.add_argument("--wandb-project", type=str, default="pcn-experiments")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    plot_dir = ensure_dir(os.path.join(RESULTS_DIR, "comparison"))

    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="cross_variant_comparison",
            group="comparison",
            tags=["comparison"],
            config={
                "variants": args.variant,
                "depths": args.depths,
                "act_fns": args.act_fns,
            },
            reinit=True,
        )

    for act_fn in args.act_fns:
        # Per-depth comparison
        for depth in args.depths:
            per_depth_results = {}
            for v in args.variant:
                res = load_results(v, act_fn, depth)
                if res is not None:
                    per_depth_results[v] = res

            if len(per_depth_results) >= 2:
                plot_cross_variant_performance(
                    per_depth_results, depth, act_fn,
                    os.path.join(plot_dir, f"perf_comparison_{act_fn}_d{depth}.png"),
                    log_to_wandb=use_wandb,
                )

        # Final accuracy bar chart
        all_results = {}
        for v in args.variant:
            all_results[v] = {}
            for depth in args.depths:
                res = load_results(v, act_fn, depth)
                if res is not None:
                    all_results[v][depth] = res

        has_data = any(len(v) > 0 for v in all_results.values())
        if has_data:
            plot_cross_variant_final_accuracy(
                all_results, args.depths, act_fn,
                os.path.join(plot_dir, f"final_accuracy_{act_fn}.png"),
                log_to_wandb=use_wandb,
            )

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
