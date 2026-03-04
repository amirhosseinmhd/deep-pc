"""Experiment 3: Weight update magnitude during training.

Tracks ||W_after - W_before||_F per layer at each training step for
SP ResNet across depths {10, 20, 40} x {tanh, relu}.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.random as jr
import numpy as np

from config import (
    SEED, DEPTHS, ACT_FNS, WIDTH, ACTIVITY_LR, PARAM_LR,
    BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
)
from common import (
    set_seed, create_model, train_and_record,
    _selected_layer_indices, ensure_dir
)
from plot_results import plot_weight_updates, plot_weight_updates_summary


def main():
    save_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results", "exp3"))

    summary = {act_fn: [] for act_fn in ACT_FNS}

    for act_fn in ACT_FNS:
        for depth in DEPTHS:
            print(f"\n=== Weight Updates | act_fn={act_fn} | depth={depth} ===")
            set_seed(SEED)
            key = jr.PRNGKey(SEED)

            model, skip_model = create_model(
                key, depth=depth, width=WIDTH,
                act_fn=act_fn, param_type="sp", use_skips=True
            )

            res = train_and_record(
                seed=SEED, model=model, skip_model=skip_model,
                param_type="sp", activity_lr=ACTIVITY_LR,
                param_lr=PARAM_LR, batch_size=BATCH_SIZE,
                n_train_iters=N_TRAIN_ITERS, test_every=TEST_EVERY,
                act_fn=act_fn, track_weight_updates=True
            )

            update_norms = res["weight_update_norms"]  # (n_iters, n_layers)
            n_layers = update_norms.shape[1]

            # Select layers to plot
            layer_idxs = _selected_layer_indices(depth)
            layer_idxs = [i for i in layer_idxs if i < n_layers]
            selected = update_norms[:, layer_idxs]
            layer_labels = [f"layer {i+1}" for i in layer_idxs]

            # Save
            np.save(
                os.path.join(save_dir, f"update_norms_{act_fn}_d{depth}.npy"),
                update_norms
            )

            # Plot per-config
            plot_weight_updates(
                selected, layer_labels,
                title=f"Weight Update Norms — {act_fn}, depth={depth} (SP ResNet)",
                save_path=os.path.join(
                    save_dir, f"weight_updates_{act_fn}_d{depth}.png"
                )
            )

            # Mean update norm across all layers for summary
            mean_update = float(np.nanmean(update_norms))
            summary[act_fn].append(mean_update)

    # Summary plot
    plot_weight_updates_summary(
        summary, DEPTHS,
        title="Mean Weight Update Norm vs Depth (SP ResNet)",
        save_path=os.path.join(save_dir, "weight_updates_summary.png")
    )

    print(f"\nExperiment 3 complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
