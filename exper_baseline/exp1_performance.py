"""Experiment 1 (Baseline): Plain MLP (no skip) performance across depths.

Trains standard-parameterisation (SP) MLPs on MNIST for ~2 epochs
and records train loss + test accuracy for depths {10, 20, 40} x activations.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exper'))

import jax.random as jr
import numpy as np

from config import (
    SEED, DEPTHS, ACT_FNS, WIDTH, ACTIVITY_LR, PARAM_LR,
    BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
)
from common import set_seed, create_model, train_and_record, ensure_dir
from plot_results import plot_performance_curves, plot_train_loss

def main():
    save_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results", "exp1"))

    for act_fn in ACT_FNS:
        depth_results = {}
        for depth in DEPTHS:
            print(f"\n=== SP MLP (no skip) | act_fn={act_fn} | depth={depth} ===")
            set_seed(SEED)
            key = jr.PRNGKey(SEED)

            model, skip_model = create_model(
                key, depth=depth, width=WIDTH,
                act_fn=act_fn, param_type="sp", use_skips=False
            )

            res = train_and_record(
                seed=SEED, model=model, skip_model=skip_model,
                param_type="sp", activity_lr=ACTIVITY_LR,
                param_lr=PARAM_LR, batch_size=BATCH_SIZE,
                n_train_iters=N_TRAIN_ITERS, test_every=TEST_EVERY,
                act_fn=act_fn
            )
            depth_results[depth] = res

            np.save(os.path.join(save_dir, f"losses_{act_fn}_d{depth}.npy"),
                    res["train_losses"])
            np.save(os.path.join(save_dir, f"accs_{act_fn}_d{depth}.npy"),
                    res["test_accs"])
            np.save(os.path.join(save_dir, f"iters_{act_fn}_d{depth}.npy"),
                    res["test_iters"])

        plot_performance_curves(
            depth_results,
            title=f"SP MLP (no skip) Performance — {act_fn}",
            save_path=os.path.join(save_dir, f"performance_{act_fn}.png")
        )
        plot_train_loss(
            depth_results,
            title=f"SP MLP (no skip) Train Loss — {act_fn}",
            save_path=os.path.join(save_dir, f"train_loss_{act_fn}.png")
        )

    print(f"\nExperiment 1 (baseline) complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
