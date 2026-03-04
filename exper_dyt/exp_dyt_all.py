"""DyT (Dynamic Tanh) experiments: performance, condition number,
weight updates, and latent norms.

Runs the same experiments as exper_batchFreezing/exp_bf_all.py but with
DyT replacing BatchNorm, for direct comparison.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.random as jr
import numpy as np

from common_dyt import (
    FCResNetDyT, train_dyt_and_record, compute_condition_number_dyt,
    INPUT_DIM, OUTPUT_DIM, WIDTH, SEED,
    ACTIVITY_LR, PARAM_LR, BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
    DEPTHS, ACT_FNS, DYT_RESULTS_DIR,
    set_seed, get_mnist_loaders, ensure_dir, _selected_layer_indices,
)
PARAM_LR = 5e-03
ACTIVITY_LR = 1e-03
from plot_results import (
    plot_performance_curves, plot_train_loss,
    plot_condition_numbers, plot_condition_bar,
    plot_weight_updates, plot_weight_updates_summary,
    plot_latent_norms_vs_layer, plot_latent_norms_vs_training,
    plot_latent_norms_across_depths,
)


def create_dyt_model(key, depth, width=WIDTH, act_fn="tanh", init_alpha=0.5):
    return FCResNetDyT(
        key=key, in_dim=INPUT_DIM, width=width, depth=depth,
        out_dim=OUTPUT_DIM, act_fn=act_fn, init_alpha=init_alpha
    )


# ====================================================================
# Experiment 1: Performance
# ====================================================================
def run_exp1_performance():
    save_dir = ensure_dir(os.path.join(DYT_RESULTS_DIR, "exp1"))
    print("\n" + "="*60)
    print("DyT Experiment 1: Performance")
    print("="*60)

    for act_fn in ACT_FNS:
        depth_results = {}
        for depth in DEPTHS:
            print(f"\n--- DyT ResNet | {act_fn} | depth={depth} ---")
            set_seed(SEED)
            key = jr.PRNGKey(SEED)
            model = create_dyt_model(key, depth=depth, act_fn=act_fn)

            res = train_dyt_and_record(
                seed=SEED, model=model, depth=depth,
                activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
                batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
                test_every=TEST_EVERY, act_fn=act_fn
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
            title=f"DyT ResNet Performance — {act_fn}",
            save_path=os.path.join(save_dir, f"performance_{act_fn}.png")
        )
        plot_train_loss(
            depth_results,
            title=f"DyT ResNet Train Loss — {act_fn}",
            save_path=os.path.join(save_dir, f"train_loss_{act_fn}.png")
        )
    print(f"\nDyT Exp 1 done. Results in {save_dir}")


# ====================================================================
# Experiment 2: Condition numbers
# ====================================================================
def run_exp2_condition():
    COND_WIDTH = 16
    save_dir = ensure_dir(os.path.join(DYT_RESULTS_DIR, "exp2"))
    print("\n" + "="*60)
    print("DyT Experiment 2: Condition Numbers")
    print("="*60)

    jax.config.update("jax_enable_x64", True)
    set_seed(SEED)
    train_loader, _ = get_mnist_loaders(1)
    x_sample, y_sample = next(iter(train_loader))
    x_sample, y_sample = x_sample.numpy(), y_sample.numpy()

    for act_fn in ACT_FNS:
        results = {}
        config_label = "DyT ResNet"
        cond_nums = []
        for depth in DEPTHS:
            print(f"  {act_fn} | DyT | depth={depth} ... ", end="", flush=True)
            key = jr.PRNGKey(SEED)
            model = create_dyt_model(key, depth=depth, width=COND_WIDTH,
                                     act_fn=act_fn)
            try:
                cond, _ = compute_condition_number_dyt(model, x_sample, y_sample)
                cond_nums.append(cond)
                print(f"κ(H_z) = {cond:.2e}")
            except Exception as e:
                cond_nums.append(float('nan'))
                print(f"FAILED ({e})")

        results[config_label] = cond_nums

        plot_condition_numbers(
            results, DEPTHS,
            title=f"Condition Number — {act_fn} (DyT)",
            save_path=os.path.join(save_dir, f"condition_number_{act_fn}.png")
        )
        plot_condition_bar(
            results, DEPTHS,
            title=f"Condition Number — {act_fn} (DyT)",
            save_path=os.path.join(save_dir, f"condition_bar_{act_fn}.png")
        )
    print(f"\nDyT Exp 2 done. Results in {save_dir}")


# ====================================================================
# Experiment 3: Weight updates
# ====================================================================
def run_exp3_weight_updates():
    save_dir = ensure_dir(os.path.join(DYT_RESULTS_DIR, "exp3"))
    print("\n" + "="*60)
    print("DyT Experiment 3: Weight Updates")
    print("="*60)

    summary = {act_fn: [] for act_fn in ACT_FNS}

    for act_fn in ACT_FNS:
        for depth in DEPTHS:
            print(f"\n--- DyT Weight Updates | {act_fn} | depth={depth} ---")
            set_seed(SEED)
            key = jr.PRNGKey(SEED)
            model = create_dyt_model(key, depth=depth, act_fn=act_fn)

            res = train_dyt_and_record(
                seed=SEED, model=model, depth=depth,
                activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
                batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
                test_every=TEST_EVERY, act_fn=act_fn,
                track_weight_updates=True
            )

            update_norms = res["weight_update_norms"]
            n_layers_tracked = update_norms.shape[1]
            layer_idxs = _selected_layer_indices(depth)
            layer_idxs = [i for i in layer_idxs if i < n_layers_tracked]
            selected = update_norms[:, layer_idxs]
            layer_labels = [f"layer {i+1}" for i in layer_idxs]

            np.save(os.path.join(save_dir, f"update_norms_{act_fn}_d{depth}.npy"),
                    update_norms)

            plot_weight_updates(
                selected, layer_labels,
                title=f"DyT Weight Updates — {act_fn}, depth={depth}",
                save_path=os.path.join(save_dir, f"weight_updates_{act_fn}_d{depth}.png")
            )
            summary[act_fn].append(float(np.nanmean(update_norms)))

    plot_weight_updates_summary(
        summary, DEPTHS,
        title="DyT Mean Weight Update Norm vs Depth",
        save_path=os.path.join(save_dir, "weight_updates_summary.png")
    )
    print(f"\nDyT Exp 3 done. Results in {save_dir}")


# ====================================================================
# Experiment 4: Latent norms
# ====================================================================
def run_exp4_latent_norms():
    save_dir = ensure_dir(os.path.join(DYT_RESULTS_DIR, "exp4"))
    print("\n" + "="*60)
    print("DyT Experiment 4: Latent Norms")
    print("="*60)

    for act_fn in ACT_FNS:
        init_norms_by_depth = {}
        for depth in DEPTHS:
            print(f"\n--- DyT Latent Norms | {act_fn} | depth={depth} ---")
            set_seed(SEED)
            key = jr.PRNGKey(SEED)
            model = create_dyt_model(key, depth=depth, act_fn=act_fn)

            res = train_dyt_and_record(
                seed=SEED, model=model, depth=depth,
                activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
                batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
                test_every=TEST_EVERY, act_fn=act_fn,
                track_activity_norms=True
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
                title=f"DyT Activity Norms vs Layer — {act_fn}, depth={depth}",
                save_path=os.path.join(save_dir, f"norms_vs_layer_{act_fn}_d{depth}.png")
            )

            n_activities = norms_init.shape[1]
            act_layer_idxs = [i for i in _selected_layer_indices(depth) if i < n_activities]
            layer_labels = [(i, f"layer {i+1}") for i in act_layer_idxs]

            plot_latent_norms_vs_training(
                norms_init, layer_labels,
                title=f"DyT Init Norms During Training — {act_fn}, depth={depth}",
                save_path=os.path.join(save_dir, f"norms_training_init_{act_fn}_d{depth}.png"),
                sample_every=max(1, len(norms_init) // 200)
            )
            plot_latent_norms_vs_training(
                norms_post, layer_labels,
                title=f"DyT Post-Inference Norms — {act_fn}, depth={depth}",
                save_path=os.path.join(save_dir, f"norms_training_post_{act_fn}_d{depth}.png"),
                sample_every=max(1, len(norms_post) // 200)
            )

        plot_latent_norms_across_depths(
            init_norms_by_depth,
            title=f"DyT Init Norms Across Depths — {act_fn}",
            save_path=os.path.join(save_dir, f"norms_across_depths_{act_fn}.png")
        )
    print(f"\nDyT Exp 4 done. Results in {save_dir}")


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    run_exp1_performance()
    run_exp2_condition()
    run_exp3_weight_updates()
    run_exp4_latent_norms()
    print("\n" + "="*60)
    print("All DyT experiments complete!")
    print(f"Results in {DYT_RESULTS_DIR}")
    print("="*60)
