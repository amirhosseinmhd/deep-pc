"""Experiment 2: Condition number of the activity Hessian.

Computes κ(H_z) at initialisation for different depths, with/without ResNet,
with standard Gaussian and orthogonal initialisations, for {tanh, relu}.
Uses WIDTH=16 for tractable Hessian computation.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.random as jr
import numpy as np

from config import SEED, DEPTHS, ACT_FNS, INPUT_DIM
from common import (
    set_seed, create_model, compute_condition_number,
    init_weights_orthogonal, get_mnist_loaders, ensure_dir
)
from plot_results import plot_condition_numbers, plot_condition_bar


COND_WIDTH = 16  # small width for tractable Hessian


def main():
    jax.config.update("jax_enable_x64", True)

    save_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results", "exp2"))

    # Get a single data sample
    set_seed(SEED)
    train_loader, _ = get_mnist_loaders(1)
    x_sample, y_sample = next(iter(train_loader))
    x_sample, y_sample = x_sample.numpy(), y_sample.numpy()

    for act_fn in ACT_FNS:
        results = {}

        for use_skips in [False, True]:
            arch_name = "ResNet" if use_skips else "MLP"
            for init_type in ["standard_gauss", "orthogonal"]:
                config_label = f"{arch_name}, {init_type}"
                cond_nums = []

                for depth in DEPTHS:
                    print(f"  {act_fn} | {config_label} | depth={depth} ... ",
                          end="", flush=True)
                    key = jr.PRNGKey(SEED)

                    model, skip_model = create_model(
                        key, depth=depth, width=COND_WIDTH,
                        act_fn=act_fn, param_type="sp", use_skips=use_skips
                    )
                    if init_type == "orthogonal":
                        key_orth = jr.PRNGKey(SEED + 1)
                        model = init_weights_orthogonal(
                            key_orth, model, act_fn=act_fn
                        )

                    try:
                        cond, eigenvals = compute_condition_number(
                            model, skip_model, "sp", x_sample, y_sample
                        )
                        cond_nums.append(cond)
                        print(f"κ(H_z) = {cond:.2e}")
                    except Exception as e:
                        cond_nums.append(float('nan'))
                        print(f"FAILED ({e})")

                results[config_label] = cond_nums

        # Save
        np.save(os.path.join(save_dir, f"cond_nums_{act_fn}.npy"),
                {k: v for k, v in results.items()})

        # Plot lines
        plot_condition_numbers(
            results, DEPTHS,
            title=f"Condition Number vs Depth — {act_fn} (SP)",
            save_path=os.path.join(save_dir, f"condition_number_{act_fn}.png")
        )

        # Plot bars
        plot_condition_bar(
            results, DEPTHS,
            title=f"Condition Number — {act_fn} (SP)",
            save_path=os.path.join(save_dir, f"condition_bar_{act_fn}.png")
        )

    print(f"\nExperiment 2 complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
