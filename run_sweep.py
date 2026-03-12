#!/usr/bin/env python
"""W&B Sweep over activity_lr and param_lr.

Usage:
    python run_sweep.py                        # launch new sweep (50 trials)
    python run_sweep.py --count 100            # custom trial count
    python run_sweep.py --sweep-id <ID>        # resume existing sweep
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import wandb
import jax.random as jr

from config import ExperimentConfig
from variants import get_variant
from training.trainer import train_and_record
from common.data import set_seed


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "test_acc",
        "goal": "maximize",
    },
    "parameters": {
        "activity_lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e1,
        },
        "param_lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e1,
        },
    },
}


def sweep_train():
    """Single sweep trial."""
    wandb.init()

    activity_lr = wandb.config.activity_lr
    param_lr = wandb.config.param_lr

    cfg = ExperimentConfig(
        activity_lr=activity_lr,
        param_lr=param_lr,
        n_train_iters=200,
        test_every=50,
        depths=[5],
        act_fns=["relu"],
        use_wandb=True,
    )

    variant = get_variant(cfg.variant)
    depth = cfg.depths[0]
    act_fn = cfg.act_fns[0]

    set_seed(cfg.seed)
    key = jr.PRNGKey(cfg.seed)

    wandb.define_metric("test_acc", step_metric="test_iter")

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
        track_weight_updates=False,
        track_activity_norms=False,
        track_grad_norms=False,
        track_layer_energy=False,
        inference_multiplier=cfg.inference_multiplier,
        activity_init=cfg.activity_init,
        use_wandb=True,
    )

    if len(res["test_accs"]) > 0:
        final_acc = float(res["test_accs"][-1])
        wandb.log({"final_test_acc": final_acc})
        wandb.summary["final_test_acc"] = final_acc

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run W&B sweep over LRs")
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of sweep trials (default: 50)",
    )
    parser.add_argument(
        "--project", type=str, default="pcn-sweeps",
        help="W&B project name (default: pcn-sweeps)",
    )
    parser.add_argument(
        "--sweep-id", type=str, default=None,
        help="Resume an existing sweep by ID",
    )
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)

    wandb.agent(sweep_id, function=sweep_train, count=args.count,
                project=args.project)


if __name__ == "__main__":
    main()
