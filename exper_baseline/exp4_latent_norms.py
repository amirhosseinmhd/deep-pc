"""Experiment 4 (Baseline): Norm of latent activities.

Tracks L2 norms of activities across layers and training iterations for
plain MLP (no skip) across depths {10, 20, 40} x activations.
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
from common import (
    set_seed, create_model, train_and_record,
    _selected_layer_indices, ensure_dir
)
from plot_results import (
    plot_latent_norms_vs_layer,
    plot_latent_norms_vs_training,
    plot_latent_norms_across_depths,
)


def main():
    save_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results", "exp4"))

    for act_fn in ACT_FNS:
        init_norms_by_depth = {}

        for depth in DEPTHS:
            print(f"\n=== Latent Norms | act_fn={act_fn} | depth={depth} ===")
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
                act_fn=act_fn, track_activity_norms=True
            )

            norms_init = res["activity_norms_init"]
            norms_post = res["activity_norms_post"]

            np.save(os.path.join(save_dir, f"norms_init_{act_fn}_d{depth}.npy"),
                    norms_init)
            np.save(os.path.join(save_dir, f"norms_post_{act_fn}_d{depth}.npy"),
                    norms_post)

            first_init = norms_init[0]
            init_norms_by_depth[depth] = first_init

            plot_latent_norms_vs_layer(
                {"init (pre-inference)": first_init,
                 "post-inference": norms_post[0]},
                title=f"Activity Norms vs Layer — {act_fn}, depth={depth} (MLP)",
                save_path=os.path.join(
                    save_dir, f"norms_vs_layer_{act_fn}_d{depth}.png"
                )
            )

            n_activities = norms_init.shape[1]
            layer_idxs = _selected_layer_indices(depth)
            act_layer_idxs = [i for i in layer_idxs if i < n_activities]
            layer_labels = [(i, f"layer {i+1}") for i in act_layer_idxs]

            plot_latent_norms_vs_training(
                norms_init, layer_labels,
                title=f"Init Activity Norms During Training — {act_fn}, depth={depth} (MLP)",
                save_path=os.path.join(
                    save_dir, f"norms_training_init_{act_fn}_d{depth}.png"
                ),
                sample_every=max(1, len(norms_init) // 200)
            )

            plot_latent_norms_vs_training(
                norms_post, layer_labels,
                title=f"Post-Inference Activity Norms — {act_fn}, depth={depth} (MLP)",
                save_path=os.path.join(
                    save_dir, f"norms_training_post_{act_fn}_d{depth}.png"
                ),
                sample_every=max(1, len(norms_post) // 200)
            )

        plot_latent_norms_across_depths(
            init_norms_by_depth,
            title=f"Init Activity Norms Across Depths — {act_fn} (MLP, no skip)",
            save_path=os.path.join(
                save_dir, f"norms_across_depths_{act_fn}.png"
            )
        )

    print(f"\nExperiment 4 (baseline) complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
