#!/usr/bin/env python
"""Run training experiments for one or more variants.

All metrics (performance, weight updates, activity norms, gradient norms,
per-layer energy) are collected in a single training pass.

Usage:
    python run_training.py --variant resnet bf dyt --depths 10 20 --act-fns relu
    python run_training.py --variant resnet --depths 10 --n-iters 200
    python run_training.py --variant resnet --no-wandb   # disable W&B
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import numpy as np
import jax.random as jr

from config import ExperimentConfig, ALL_VARIANTS
from variants import get_variant
from training.trainer import train_and_record
from common.data import set_seed
from common.utils import ensure_dir
from plotting.plots import generate_all_plots


def run_single(cfg):
    """Run training for all depths x act_fns for one variant."""
    variant = get_variant(cfg.variant)
    use_wandb = cfg.use_wandb

    if use_wandb:
        from common.wandb_logger import init_wandb, finish_wandb

    for act_fn in cfg.act_fns:
        depth_results = {}
        for depth in cfg.depths:
            print(f"\n=== {variant.name} | {act_fn} | depth={depth} ===")
            set_seed(cfg.seed)
            key = jr.PRNGKey(cfg.seed)

            if use_wandb:
                init_wandb(cfg, depth, act_fn)

            model = variant.create_model(
                key, depth=depth, width=cfg.width, act_fn=act_fn,
                init_alpha=cfg.init_alpha,
            )

            res = train_and_record(
                variant=variant,
                model=model,
                depth=depth,
                seed=cfg.seed,
                activity_lr=cfg.activity_lr,
                param_lr=cfg.param_lr,
                batch_size=cfg.batch_size,
                n_train_iters=cfg.n_train_iters,
                test_every=cfg.test_every,
                act_fn=act_fn,
                track_weight_updates=cfg.track_weight_updates,
                track_activity_norms=cfg.track_activity_norms,
                track_grad_norms=cfg.track_grad_norms,
                track_layer_energy=cfg.track_layer_energy,
                inference_multiplier=cfg.inference_multiplier,
                use_wandb=use_wandb,
            )
            depth_results[depth] = res

            # Save all raw metrics (.npy — cheap, useful for offline analysis)
            save_dir = ensure_dir(
                os.path.join(cfg.results_dir, f"{act_fn}_d{depth}")
            )
            for key_name, value in res.items():
                np.save(os.path.join(save_dir, f"{key_name}.npy"), value)
            print(f"  Results saved to: {save_dir}")

            if use_wandb:
                finish_wandb()

        # Generate plots: log to W&B as images or save PNGs locally
        if use_wandb:
            # Start a summary run for cross-depth plots
            from common.wandb_logger import init_wandb as _init, finish_wandb as _fin
            import wandb
            from dataclasses import asdict
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"{cfg.variant}_{act_fn}_summary",
                group=cfg.variant,
                tags=[act_fn, "summary", cfg.variant],
                config=asdict(cfg),
                reinit=True,
            )
            generate_all_plots(depth_results, cfg, act_fn, log_to_wandb=True)
            wandb.finish()
        else:
            generate_all_plots(depth_results, cfg, act_fn)


def main():
    parser = argparse.ArgumentParser(
        description="Run unified PC training experiments"
    )
    parser.add_argument(
        "--variant", choices=ALL_VARIANTS, nargs="+",
        default=["resnet"],
        help="Which variant(s) to train",
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=None,
        help="Network depths to test (default: [5, 10, 20, 40])",
    )
    parser.add_argument(
        "--act-fns", nargs="+", default=None,
        help="Activation functions (default: ['relu'])",
    )
    parser.add_argument(
        "--n-iters", type=int, default=None,
        help="Number of training iterations",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--activity-lr", type=float, default=None)
    parser.add_argument("--param-lr", type=float, default=None)
    parser.add_argument(
        "--no-weight-updates", action="store_true",
        help="Disable weight update tracking",
    )
    parser.add_argument(
        "--no-activity-norms", action="store_true",
        help="Disable activity norm tracking",
    )
    parser.add_argument(
        "--no-grad-norms", action="store_true",
        help="Disable gradient norm tracking",
    )
    parser.add_argument(
        "--no-layer-energy", action="store_true",
        help="Disable per-layer energy tracking",
    )
    parser.add_argument(
        "--inference-multiplier", type=float, default=None,
        help="Multiply inference steps by this factor (default: 1, i.e. depth steps)",
    )
    # W&B arguments
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="W&B project name (default: pcn-experiments)",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity (team or username)",
    )
    args = parser.parse_args()

    for variant_name in args.variant:
        overrides = {}
        if args.depths:
            overrides["depths"] = args.depths
        if args.act_fns:
            overrides["act_fns"] = args.act_fns
        if args.n_iters:
            overrides["n_train_iters"] = args.n_iters
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        if args.activity_lr:
            overrides["activity_lr"] = args.activity_lr
        if args.param_lr:
            overrides["param_lr"] = args.param_lr
        if args.no_weight_updates:
            overrides["track_weight_updates"] = False
        if args.no_activity_norms:
            overrides["track_activity_norms"] = False
        if args.no_grad_norms:
            overrides["track_grad_norms"] = False
        if args.no_layer_energy:
            overrides["track_layer_energy"] = False
        if args.inference_multiplier:
            overrides["inference_multiplier"] = args.inference_multiplier
        if args.no_wandb:
            overrides["use_wandb"] = False
        if args.wandb_project:
            overrides["wandb_project"] = args.wandb_project
        if args.wandb_entity:
            overrides["wandb_entity"] = args.wandb_entity

        cfg = ExperimentConfig.from_variant(variant_name, **overrides)
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"Depths: {cfg.depths}")
        print(f"Activity LR: {cfg.activity_lr}, Param LR: {cfg.param_lr}")
        print(f"Iterations: {cfg.n_train_iters}")
        print(f"Results dir: {cfg.results_dir}")
        print(f"W&B: {'enabled' if cfg.use_wandb else 'disabled'}")
        print(f"{'='*60}")
        run_single(cfg)


if __name__ == "__main__":
    main()
