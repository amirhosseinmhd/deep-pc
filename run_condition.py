#!/usr/bin/env python
"""Run condition number experiments (separate from training — expensive).

Usage:
    python run_condition.py --variant resnet bf dyt
    python run_condition.py --variant resnet --depths 5 10 20
    python run_condition.py --no-wandb   # save PNGs locally instead
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse

from config import ExperimentConfig, ALL_VARIANTS
from variants import get_variant
from training.condition import run_condition_number
from plotting.plots import plot_condition_numbers
from common.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run condition number experiments"
    )
    parser.add_argument(
        "--variant", choices=ALL_VARIANTS, nargs="+",
        default=["resnet"],
    )
    parser.add_argument("--depths", type=int, nargs="+", default=None)
    parser.add_argument("--act-fns", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (save PNGs locally)",
    )
    parser.add_argument("--wandb-project", type=str, default="pcn-experiments")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="condition_number",
            group="condition",
            tags=["condition-number"],
            config={
                "variants": args.variant,
                "depths": args.depths,
            },
            reinit=True,
        )

    # Collect results across variants for comparison plot
    all_cond_results = {}

    for variant_name in args.variant:
        overrides = {}
        if args.depths:
            overrides["depths"] = args.depths
        if args.act_fns:
            overrides["act_fns"] = args.act_fns
        if args.seed is not None:
            overrides["seed"] = args.seed

        cfg = ExperimentConfig.from_variant(variant_name, **overrides)
        variant = get_variant(variant_name)

        print(f"\n=== Condition Number: {variant.name} ===")
        results = run_condition_number(variant, cfg)

        for act_fn, conds in results.items():
            label = f"{variant.name} ({act_fn})"
            all_cond_results[label] = conds

            # Log individual condition numbers to W&B
            if use_wandb:
                for i, (depth, cond) in enumerate(zip(cfg.depths, conds)):
                    wandb.log({
                        f"condition_number/{label}": cond,
                        "depth": depth,
                    })

    # Cross-variant comparison plot
    if len(args.variant) > 1:
        plot_dir = ensure_dir(os.path.join("results", "comparison"))
        plot_condition_numbers(
            all_cond_results,
            cfg.depths,
            "Condition Number Comparison",
            os.path.join(plot_dir, "condition_comparison.png"),
            log_to_wandb=use_wandb,
        )

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
