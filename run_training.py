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
import common.jax_setup  # Configure JAX platform before importing jax.

import argparse
import numpy as np
import jax
import jax.random as jr

from config import (
    ExperimentConfig, ALL_VARIANTS,
    VARIANT_REC_LRA, VARIANT_CNN_REC_LRA, VARIANT_RES_ERROR_NET,
    VARIANT_RES_ERROR_NET_RESNET18,
)
from variants import get_variant
from training.trainer import train_and_record
from training.rec_lra_trainer import train_rec_lra
from training.res_error_net_trainer import train_res_error_net
from common.data import set_seed
from common.utils import ensure_dir
from plotting.plots import generate_all_plots


def report_jax_backend():
    """Print the active JAX backend and selected devices."""
    devices = jax.devices()
    backend = jax.default_backend()
    device_summary = ", ".join(str(device) for device in devices)
    using_gpu = any(
        getattr(device, "platform", "") in {"gpu", "cuda"}
        for device in devices
    )

    print(f"JAX backend: {backend}")
    print(f"JAX devices: {device_summary}")
    if os.environ.get("JAX_PLATFORMS"):
        print(f"JAX_PLATFORMS: {os.environ['JAX_PLATFORMS']}")
    if not using_gpu:
        print("WARNING: JAX did not select a GPU; this run is using CPU.")


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
            if cfg.variant in (VARIANT_REC_LRA, VARIANT_CNN_REC_LRA):
                extra_create_kwargs = dict(
                    alpha_e_skip=cfg.alpha_e_skip,
                    alpha_e_adj=cfg.alpha_e_adj,
                )
            if cfg.variant == VARIANT_CNN_REC_LRA:
                extra_create_kwargs.update(dict(
                    cnn_channels=cfg.cnn_channels,
                    cnn_fc_width=cfg.cnn_fc_width,
                    n_fc_hidden=cfg.n_fc_hidden,
                    kernel_size=cfg.kernel_size,
                    input_shape=(3, 32, 32),
                    use_layer_norm=cfg.use_layer_norm,
                ))
            if cfg.variant == VARIANT_RES_ERROR_NET:
                extra_create_kwargs = dict(
                    highway_every_k=cfg.res_highway_every_k,
                    v_init_scale=cfg.res_v_init_scale,
                    res_init_scheme=cfg.res_init_scheme,
                    inference_method=cfg.res_inference_method,
                    s_mode=cfg.res_highway_s_mode,
                    res_param_type=cfg.res_param_type,
                )
            if cfg.variant == VARIANT_RES_ERROR_NET_RESNET18:
                extra_create_kwargs = dict(
                    input_shape=(3, 32, 32),
                    resnet_channels=cfg.res_resnet_channels,
                    blocks_per_stage=cfg.res_resnet_blocks_per_stage,
                    normalization=cfg.res_resnet_normalization,
                    dyt_init_alpha=cfg.res_resnet_dyt_init_alpha,
                    highway_include_stem=cfg.res_resnet_highway_include_stem,
                    v_init_scale=cfg.res_v_init_scale,
                    inference_method=cfg.res_resnet_inference_method,
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

            if cfg.variant in (VARIANT_RES_ERROR_NET, VARIANT_RES_ERROR_NET_RESNET18):
                # CNN variant uses its own T default (each inference step is
                # ~100× costlier than the MLP variant).
                eff_T = (
                    cfg.res_resnet_inference_T
                    if cfg.variant == VARIANT_RES_ERROR_NET_RESNET18
                    else cfg.res_inference_T
                )
                res = train_res_error_net(
                    variant=variant,
                    bundle=model,
                    depth=depth,
                    seed=cfg.seed,
                    param_lr=cfg.param_lr,
                    v_lr=cfg.res_v_lr,
                    batch_size=cfg.batch_size,
                    n_train_iters=cfg.n_train_iters,
                    test_every=cfg.test_every,
                    act_fn=act_fn,
                    dataset=cfg.dataset,
                    alpha=cfg.res_alpha,
                    inference_T=eff_T,
                    inference_dt=cfg.res_inference_dt,
                    v_update_rule=cfg.res_v_update_rule,
                    optim_type=cfg.res_optim,
                    loss_type=cfg.res_loss,
                    reproject_c=cfg.reproject_c,
                    input_noise_sigma=cfg.input_noise_sigma,
                    weight_decay=cfg.weight_decay,
                    v_reg=cfg.res_v_reg,
                    use_zca=cfg.use_zca,
                    track_weight_updates=cfg.track_weight_updates,
                    track_activity_norms=cfg.track_activity_norms,
                    track_grad_norms=cfg.track_grad_norms,
                    track_layer_energy=cfg.track_layer_energy,
                    use_wandb=use_wandb,
                    alpha_schedule=cfg.res_alpha_schedule,
                    alpha_min=cfg.res_alpha_min,
                    freeze_v=cfg.res_v_frozen,
                )
            elif cfg.variant in (VARIANT_REC_LRA, VARIANT_CNN_REC_LRA):
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
                    reproject_c=cfg.reproject_c,
                    input_noise_sigma=cfg.input_noise_sigma,
                    weight_decay=cfg.weight_decay,
                    use_zca=cfg.use_zca,
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
        "--dataset", choices=["MNIST", "FashionMNIST", "CIFAR10"], default=None,
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
        "--rec-lra-optim", choices=["sgd", "adam", "adamw"], default=None,
        help="Optimizer for rec-LRA Hebbian updates (default: adamw)",
    )
    parser.add_argument(
        "--reproject-c", type=float, default=None,
        help="Gaussian-ball radius for update re-projection. 0 disables (default: 1.0)",
    )
    parser.add_argument(
        "--input-noise-sigma", type=float, default=None,
        help="Stdev of Gaussian noise added to inputs. 0 disables (default: 0.1)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--alpha-e-skip", type=float, default=None,
        help="α for skip-error endpoints (paper: 0.19; 1.0 = pure MSE)",
    )
    parser.add_argument(
        "--alpha-e-adj", type=float, default=None,
        help="α for adjacent-error endpoints (paper: 0.24; 1.0 = pure MSE)",
    )
    parser.add_argument(
        "--no-layer-norm", action="store_true",
        help="Disable LayerNorm in CNN (default: enabled)",
    )
    parser.add_argument(
        "--use-zca", action="store_true",
        help="Enable GCN+ZCA preprocessing on CIFAR-10 (default: disabled — hurts in raw-MSE setup)",
    )
    parser.add_argument(
        "--rec-lra-loss", choices=["mse", "ce"], default=None,
        help="Loss function for rec-LRA (default: mse)",
    )
    parser.add_argument(
        "--rec-lra-e-update", choices=["hebbian", "grad"], default=None,
        help="E update rule: 'hebbian' (Eq.6, default) or 'grad' (rLRA-dx, true gradient)",
    )
    # res-error-net arguments
    parser.add_argument(
        "--res-highway-every-k", type=int, default=None,
        help="Stride of V_{L->i} highways (res-error-net, default: 2)",
    )
    parser.add_argument(
        "--res-alpha", type=float, default=None,
        help="Global coupling α for V highways (res-error-net, default: 0.1)",
    )
    parser.add_argument(
        "--res-inference-T", type=int, default=None,
        help="Number of inference (Euler) steps per batch (res-error-net, default: 20)",
    )
    parser.add_argument(
        "--res-inference-dt", type=float, default=None,
        help="Euler step size for z dynamics (res-error-net, default: 0.1)",
    )
    parser.add_argument(
        "--res-v-lr", type=float, default=None,
        help="Learning rate for V highway matrices (res-error-net, default: 1e-4)",
    )
    parser.add_argument(
        "--res-v-update-rule", choices=["energy", "state"], default=None,
        help="V update rule: 'energy' (α·e^i·e^L^T, derived) or 'state' (α·z^i·e^L^T)",
    )
    parser.add_argument(
        "--res-v-init-scale", type=float, default=None,
        help="Init scale for V matrices (res-error-net, default: 0.01)",
    )
    parser.add_argument(
        "--res-optim", choices=["sgd", "adam", "adamw"], default=None,
        help="Optimizer for res-error-net (default: adamw)",
    )
    parser.add_argument(
        "--res-loss", choices=["mse", "ce"], default=None,
        help="Loss for res-error-net reporting (default: mse)",
    )
    parser.add_argument(
        "--res-init-scheme", choices=["jpc_default", "unit_gaussian", "kaiming"], default=None,
        help="Weight init for res-error-net. 'jpc_default' = 1/√fan_in (stable), "
             "'unit_gaussian' = rec-LRA paper style (needs ZCA / small dt)",
    )
    parser.add_argument(
        "--res-v-reg", type=float, default=None,
        help="L2 penalty ρ on V highway matrices (res-error-net, default: 0.0). "
             "Adds ρ·V to ΔV so F_aug is bounded below in V.",
    )
    # Ablation knobs (res-error-net)
    parser.add_argument(
        "--res-alpha-schedule",
        choices=["fixed", "linear", "cosine"], default=None,
        help="Per-iter α schedule for the highway coupling (default: fixed).",
    )
    parser.add_argument(
        "--res-alpha-min", type=float, default=None,
        help="Endpoint α for linear/cosine decay (default: 0.0).",
    )
    parser.add_argument(
        "--res-v-frozen", action="store_true",
        help="DFA-style: freeze V_{L→i} at init, never update.",
    )
    parser.add_argument(
        "--res-highway-s-mode",
        choices=["stride", "dense", "sparse", "random"], default=None,
        help="How to choose S (the set of layers receiving V highways): "
             "'stride' uses --res-highway-every-k (default), 'dense' = all "
             "hidden layers, 'sparse' = only the layer just below output, "
             "'random' = sample |S|≈depth/k layers at init.",
    )
    # res-error-net-resnet18 arguments
    parser.add_argument(
        "--res-resnet-channels", type=int, nargs="+", default=None,
        help="Channels per stage for ResNet-18: [stem, s1, s2, s3, s4] "
             "(default: [64, 64, 128, 256, 512])",
    )
    parser.add_argument(
        "--res-resnet-blocks-per-stage", type=int, default=None,
        help="Basic blocks per stage (default: 2 → ResNet-18)",
    )
    parser.add_argument(
        "--res-resnet-normalization", choices=["dyt", "none"], default=None,
        help="Per-block normalization: 'dyt' (DyT) or 'none' (default: dyt)",
    )
    parser.add_argument(
        "--res-resnet-dyt-init-alpha", type=float, default=None,
        help="Initial α for DyT layers (default: 0.5, matches variants/dyt.py)",
    )
    parser.add_argument(
        "--res-resnet-no-stem-highway", action="store_true",
        help="Exclude the stem output from the set of highway activities",
    )
    parser.add_argument(
        "--res-resnet-inference-T", type=int, default=None,
        help="Inference steps for ResNet-18 variant (default: 15; the MLP "
             "variant uses 50 but each CNN step is ~100× more expensive)",
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
        if args.reproject_c is not None:
            overrides["reproject_c"] = args.reproject_c
        if args.input_noise_sigma is not None:
            overrides["input_noise_sigma"] = args.input_noise_sigma
        if args.weight_decay is not None:
            overrides["weight_decay"] = args.weight_decay
        if args.alpha_e_skip is not None:
            overrides["alpha_e_skip"] = args.alpha_e_skip
        if args.alpha_e_adj is not None:
            overrides["alpha_e_adj"] = args.alpha_e_adj
        if args.no_layer_norm:
            overrides["use_layer_norm"] = False
        if args.use_zca:
            overrides["use_zca"] = True
        if args.res_highway_every_k is not None:
            overrides["res_highway_every_k"] = args.res_highway_every_k
        if args.res_alpha is not None:
            overrides["res_alpha"] = args.res_alpha
        if args.res_inference_T is not None:
            overrides["res_inference_T"] = args.res_inference_T
        if args.res_inference_dt is not None:
            overrides["res_inference_dt"] = args.res_inference_dt
        if args.res_v_lr is not None:
            overrides["res_v_lr"] = args.res_v_lr
        if args.res_v_update_rule is not None:
            overrides["res_v_update_rule"] = args.res_v_update_rule
        if args.res_v_init_scale is not None:
            overrides["res_v_init_scale"] = args.res_v_init_scale
        if args.res_optim is not None:
            overrides["res_optim"] = args.res_optim
        if args.res_loss is not None:
            overrides["res_loss"] = args.res_loss
        if args.res_init_scheme is not None:
            overrides["res_init_scheme"] = args.res_init_scheme
        if args.res_v_reg is not None:
            overrides["res_v_reg"] = args.res_v_reg
        if args.res_alpha_schedule is not None:
            overrides["res_alpha_schedule"] = args.res_alpha_schedule
        if args.res_alpha_min is not None:
            overrides["res_alpha_min"] = args.res_alpha_min
        if args.res_v_frozen:
            overrides["res_v_frozen"] = True
        if args.res_highway_s_mode is not None:
            overrides["res_highway_s_mode"] = args.res_highway_s_mode
        if args.res_resnet_channels:
            overrides["res_resnet_channels"] = args.res_resnet_channels
        if args.res_resnet_blocks_per_stage is not None:
            overrides["res_resnet_blocks_per_stage"] = args.res_resnet_blocks_per_stage
        if args.res_resnet_normalization is not None:
            overrides["res_resnet_normalization"] = args.res_resnet_normalization
        if args.res_resnet_dyt_init_alpha is not None:
            overrides["res_resnet_dyt_init_alpha"] = args.res_resnet_dyt_init_alpha
        if args.res_resnet_no_stem_highway:
            overrides["res_resnet_highway_include_stem"] = False
        if args.res_resnet_inference_T is not None:
            overrides["res_resnet_inference_T"] = args.res_resnet_inference_T
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
        report_jax_backend()
        print(f"{'='*60}")
        run_single(cfg)


if __name__ == "__main__":
    main()
