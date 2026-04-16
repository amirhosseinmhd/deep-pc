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

from config import ExperimentConfig, ALL_VARIANTS, VARIANT_REC_LRA, VARIANT_CNN_REC_LRA
from variants import get_variant
from training.trainer import train_and_record
from training.rec_lra_trainer import train_rec_lra
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

            extra_create_kwargs = {}
            if cfg.variant == VARIANT_CNN_REC_LRA:
                extra_create_kwargs = dict(
                    cnn_channels=cfg.cnn_channels,
                    cnn_fc_width=cfg.cnn_fc_width,
                    n_fc_hidden=cfg.n_fc_hidden,
                    kernel_size=cfg.kernel_size,
                    input_shape=(3, 32, 32),
                )

            model = variant.create_model(
                key, depth=depth, width=cfg.width, act_fn=act_fn,
                input_dim=cfg.input_dim, output_dim=cfg.output_dim,
                init_alpha=cfg.init_alpha,
                activity_noise=cfg.activity_noise,
                forward_skip_every=cfg.forward_skip_every,
                error_skip_every=cfg.error_skip_every,
                **extra_create_kwargs,
            )

            if cfg.variant in (VARIANT_REC_LRA, VARIANT_CNN_REC_LRA):
                res = train_rec_lra(
                    variant=variant,
                    model=model,
                    depth=depth,
                    seed=cfg.seed,
                    param_lr=cfg.param_lr,
                    e_lr=cfg.e_lr,
                    batch_size=cfg.batch_size,
                    n_train_iters=cfg.n_train_iters,
                    test_every=cfg.test_every,
                    act_fn=act_fn,
                    dataset=cfg.dataset,
                    beta=cfg.beta,
                    gamma_E=cfg.gamma_E,
                    rec_lra_optim=cfg.rec_lra_optim,
                    rec_lra_loss=cfg.rec_lra_loss,
                    rec_lra_e_update=cfg.rec_lra_e_update,
                    track_weight_updates=cfg.track_weight_updates,
                    track_activity_norms=cfg.track_activity_norms,
                    track_grad_norms=cfg.track_grad_norms,
                    track_layer_energy=cfg.track_layer_energy,
                    use_wandb=use_wandb,
                )
            else:
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
                    dataset=cfg.dataset,
                    track_weight_updates=cfg.track_weight_updates,
                    track_activity_norms=cfg.track_activity_norms,
                    track_grad_norms=cfg.track_grad_norms,
                    track_layer_energy=cfg.track_layer_energy,
                    inference_multiplier=cfg.inference_multiplier,
                    activity_init=cfg.activity_init,
                    param_optim_type=cfg.param_optim_type,
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
        "--dataset", choices=["MNIST", "CIFAR10"], default=None,
        help="Dataset to train on (default: MNIST)",
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
    parser.add_argument(
        "--activity-init", choices=["ffwd", "zeros"], default=None,
        help="Activity initialization: 'ffwd' (forward pass) or 'zeros' (default: ffwd)",
    )
    parser.add_argument(
        "--param-optim", choices=["adam", "sgd"], default=None,
        help="Parameter optimizer: 'adam' or 'sgd' (default: adam)",
    )
    parser.add_argument(
        "--activity-noise", type=float, default=None,
        help="Noise scale for activity init (dyt_v2 experiment, default: 0.0)",
    )
    # rec-LRA arguments
    parser.add_argument(
        "--forward-skip-every", type=int, default=None,
        help="Forward skip connection interval (rec-LRA, default: 2)",
    )
    parser.add_argument(
        "--error-skip-every", type=int, default=None,
        help="Error skip connection interval (rec-LRA, default: 2)",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Target nudging strength (rec-LRA, default: 0.1)",
    )
    parser.add_argument(
        "--gamma-e", type=float, default=None,
        help="E matrix learning rate scale in Hebbian rule (rec-LRA, default: 0.01)",
    )
    parser.add_argument(
        "--e-lr", type=float, default=None,
        help="Learning rate for E matrices (rec-LRA, default: 1e-3)",
    )
    parser.add_argument(
        "--rec-lra-optim", choices=["sgd", "adam"], default=None,
        help="Optimizer for rec-LRA Hebbian updates (default: sgd)",
    )
    parser.add_argument(
        "--rec-lra-loss", choices=["mse", "ce"], default=None,
        help="Loss function for rec-LRA (default: mse)",
    )
    parser.add_argument(
        "--rec-lra-e-update", choices=["hebbian", "grad"], default=None,
        help="E update rule: 'hebbian' (Eq.6, default) or 'grad' (rLRA-dx, true gradient)",
    )
    # CNN-rec-LRA arguments
    parser.add_argument(
        "--cnn-channels", type=int, nargs="+", default=None,
        help="Output channels per conv layer for CNN-rLRA (default: [32, 64, 128])",
    )
    parser.add_argument(
        "--cnn-fc-width", type=int, default=None,
        help="FC hidden layer width for CNN-rLRA (default: 256)",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=None,
        help="Conv kernel size for CNN-rLRA (default: 3)",
    )
    parser.add_argument(
        "--n-fc-hidden", type=int, default=None,
        help="Number of hidden FC layers for CNN-rLRA (default: 1)",
    )
    parser.add_argument(
        "--test-every", type=int, default=None,
        help="Evaluate on test set every N iterations (default: 1)",
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
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="Custom W&B run name (default: variant_actfn_dDEPTH)",
    )
    args = parser.parse_args()

    for variant_name in args.variant:
        overrides = {}
        if args.dataset:
            overrides["dataset"] = args.dataset
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
        if args.activity_init:
            overrides["activity_init"] = args.activity_init
        if args.param_optim:
            overrides["param_optim_type"] = args.param_optim
        if args.activity_noise is not None:
            overrides["activity_noise"] = args.activity_noise
        if args.forward_skip_every is not None:
            overrides["forward_skip_every"] = args.forward_skip_every
        if args.error_skip_every is not None:
            overrides["error_skip_every"] = args.error_skip_every
        if args.beta is not None:
            overrides["beta"] = args.beta
        if args.gamma_e is not None:
            overrides["gamma_E"] = args.gamma_e
        if args.e_lr is not None:
            overrides["e_lr"] = args.e_lr
        if args.rec_lra_optim is not None:
            overrides["rec_lra_optim"] = args.rec_lra_optim
        if args.rec_lra_loss is not None:
            overrides["rec_lra_loss"] = args.rec_lra_loss
        if args.rec_lra_e_update is not None:
            overrides["rec_lra_e_update"] = args.rec_lra_e_update
        if args.cnn_channels:
            overrides["cnn_channels"] = args.cnn_channels
        if args.cnn_fc_width:
            overrides["cnn_fc_width"] = args.cnn_fc_width
        if args.kernel_size:
            overrides["kernel_size"] = args.kernel_size
        if args.n_fc_hidden is not None:
            overrides["n_fc_hidden"] = args.n_fc_hidden
        if args.test_every is not None:
            overrides["test_every"] = args.test_every
        if args.no_wandb:
            overrides["use_wandb"] = False
        if args.wandb_project:
            overrides["wandb_project"] = args.wandb_project
        if args.wandb_entity:
            overrides["wandb_entity"] = args.wandb_entity
        if args.wandb_run_name:
            overrides["wandb_run_name"] = args.wandb_run_name

        cfg = ExperimentConfig.from_variant(variant_name, **overrides)
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"Dataset: {cfg.dataset} (input_dim={cfg.input_dim})")
        print(f"Depths: {cfg.depths}")
        print(f"Activity LR: {cfg.activity_lr}, Param LR: {cfg.param_lr}")
        print(f"Iterations: {cfg.n_train_iters}")
        print(f"Results dir: {cfg.results_dir}")
        print(f"W&B: {'enabled' if cfg.use_wandb else 'disabled'}")
        print(f"{'='*60}")
        run_single(cfg)


if __name__ == "__main__":
    main()
