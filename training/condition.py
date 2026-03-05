"""Condition number experiment (separate from training — expensive)."""

import jax
import jax.random as jr
import numpy as np

from common.data import set_seed, get_mnist_loaders
from common.utils import ensure_dir


def run_condition_number(variant, cfg):
    """Compute condition number kappa(H_z) for a variant across depths.

    Args:
        variant: PCVariant instance
        cfg: ExperimentConfig

    Returns:
        dict mapping act_fn to list of condition numbers per depth
    """
    jax.config.update("jax_enable_x64", True)

    save_dir = ensure_dir(cfg.results_dir)
    set_seed(cfg.seed)

    train_loader, _ = get_mnist_loaders(1)
    x_sample, y_sample = next(iter(train_loader))
    x_sample, y_sample = x_sample.numpy(), y_sample.numpy()

    all_results = {}
    for act_fn in cfg.act_fns:
        cond_nums = []
        for depth in cfg.depths:
            print(
                f"  {act_fn} | {variant.name} | depth={depth} ... ",
                end="", flush=True,
            )
            key = jr.PRNGKey(cfg.seed)
            model = variant.create_model(
                key, depth=depth, width=cfg.cond_width, act_fn=act_fn,
                init_alpha=cfg.init_alpha,
            )

            try:
                cond, eigenvals = variant.compute_condition_number(
                    model, x_sample, y_sample
                )
                cond_nums.append(cond)
                print(f"kappa(H_z) = {cond:.2e}")
            except Exception as e:
                cond_nums.append(float('nan'))
                print(f"FAILED ({e})")

        cond_dir = ensure_dir(f"{save_dir}/condition")
        np.save(f"{cond_dir}/cond_nums_{act_fn}.npy", cond_nums)
        np.save(f"{cond_dir}/depths.npy", cfg.depths)
        all_results[act_fn] = cond_nums

    return all_results
