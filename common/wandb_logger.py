"""Weights & Biases logger for PCN experiments.

Thin wrapper that handles init, per-step metric logging,
matplotlib figure logging, and cleanup.
"""

import wandb


def init_wandb(cfg, depth, act_fn):
    """Initialize a W&B run for a single (variant, act_fn, depth) combo.

    Args:
        cfg: ExperimentConfig instance
        depth: network depth
        act_fn: activation function name
    """
    from dataclasses import asdict

    run_name = cfg.wandb_run_name or f"{cfg.variant}_{act_fn}_d{depth}"
    config_dict = asdict(cfg)
    config_dict["depth"] = depth
    config_dict["act_fn"] = act_fn

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        group=cfg.variant,
        tags=[act_fn, f"depth-{depth}", cfg.variant],
        config=config_dict,
        reinit=True,
    )

    # Custom x-axis for test metrics
    wandb.define_metric("test_acc", step_metric="test_iter")
    # Inference-diagnostic scalars are logged at checkpoint cadence; group
    # them under inference_iter so they plot against training iter, not the
    # global wandb step (which jumps by `test_every` between checkpoints).
    wandb.define_metric("inference/*", step_metric="inference_iter")


def log_step_metrics(iter_num, metrics_dict):
    """Log a dict of metrics at a given training step.

    Args:
        iter_num: current iteration number
        metrics_dict: flat dict of metric_name -> value
    """
    metrics_dict["iter"] = iter_num
    wandb.log(metrics_dict, step=iter_num)


def log_figure(name, fig):
    """Log a matplotlib figure to W&B as an image, then close it.

    Args:
        name: key name for the image in W&B
        fig: matplotlib Figure object
    """
    wandb.log({name: wandb.Image(fig)})
    import matplotlib.pyplot as plt
    plt.close(fig)


def finish_wandb():
    """Finish the current W&B run."""
    wandb.finish()
