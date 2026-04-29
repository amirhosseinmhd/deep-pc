#!/usr/bin/env python
"""Compare per-layer weight update norms: no-noise vs noise vs external skip.

Runs the no-noise dyt_v2 experiment fresh (to avoid overwrite issue),
then loads existing results for dyt_v2 (with noise) and dyt_v3.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import jax.random as jr

from config import ExperimentConfig
from variants import get_variant
from training.trainer import train_and_record
from common.data import set_seed

DEPTH = 50
N_ITERS = 200
ACT_FN = "relu"

def run_no_noise():
    """Run dyt_v2 with zero noise and return results."""
    cfg = ExperimentConfig.from_variant("dyt_v2",
        depths=[DEPTH], act_fns=[ACT_FN], n_train_iters=N_ITERS,
        use_wandb=False, activity_noise=0.0)
    variant = get_variant("dyt_v2")
    set_seed(cfg.seed)
    key = jr.PRNGKey(cfg.seed)
    model = variant.create_model(
        key, depth=DEPTH, width=cfg.width, act_fn=ACT_FN,
        init_alpha=cfg.init_alpha, activity_noise=0.0,
    )
    return train_and_record(
        variant=variant, model=model, depth=DEPTH, seed=cfg.seed,
        activity_lr=cfg.activity_lr, param_lr=cfg.param_lr,
        batch_size=cfg.batch_size, n_train_iters=N_ITERS,
        test_every=cfg.test_every, act_fn=ACT_FN,
        track_weight_updates=True, track_activity_norms=True,
        track_grad_norms=True, track_layer_energy=True,
        inference_multiplier=cfg.inference_multiplier,
        activity_init=cfg.activity_init,
        param_optim_type=cfg.param_optim_type,
        use_wandb=False,
    )


def load_saved(variant_name):
    """Load weight_update_norms from saved results."""
    path = f"results/{variant_name}/relu_d{DEPTH}/weight_update_norms.npy"
    return np.load(path)


def main():
    print("=" * 70)
    print("Running dyt_v2 NO NOISE (depth=50, 200 iters)...")
    print("=" * 70)
    no_noise_res = run_no_noise()

    # Weight update norms: shape (n_iters, n_layers)
    no_noise_norms = np.array(no_noise_res["weight_update_norms"])

    print("\nLoading saved results...")
    with_noise_norms = load_saved("dyt_v2")  # from previous with-noise run
    v3_norms = load_saved("dyt_v3")

    # Average across all iterations to get per-layer mean
    no_noise_mean = np.mean(no_noise_norms, axis=0)
    with_noise_mean = np.mean(with_noise_norms, axis=0)
    v3_mean = np.mean(v3_norms, axis=0)

    n_layers = len(no_noise_mean)

    print(f"\n{'=' * 70}")
    print(f"Per-Layer Mean Weight Update Norms (depth={DEPTH})")
    print(f"{'=' * 70}")
    print(f"{'Layer':>6}  {'No Noise':>12}  {'Noise 2.4e-7':>12}  {'v3 (ext skip)':>13}")
    print("-" * 50)
    for i in range(n_layers):
        nn_val = no_noise_mean[i] if i < len(no_noise_mean) else 0
        wn_val = with_noise_mean[i] if i < len(with_noise_mean) else 0
        v3_val = v3_mean[i] if i < len(v3_mean) else 0
        marker = ""
        if nn_val < 1e-10:
            marker = " <-- DEAD"
        print(f"{i:>6}  {nn_val:>12.6e}  {wn_val:>12.6e}  {v3_val:>13.6e}{marker}")

    # Summary stats
    print(f"\n{'=' * 70}")
    print("Summary:")
    dead_layers_nn = np.sum(no_noise_mean < 1e-10)
    dead_layers_wn = np.sum(with_noise_mean < 1e-10)
    dead_layers_v3 = np.sum(v3_mean < 1e-10)
    print(f"  Dead layers (norm < 1e-10):")
    print(f"    No noise:      {dead_layers_nn}/{n_layers}")
    print(f"    Noise 2.4e-7:  {dead_layers_wn}/{n_layers}")
    print(f"    v3 (ext skip): {dead_layers_v3}/{n_layers}")

    print(f"\n  Mean norm across all layers:")
    print(f"    No noise:      {np.mean(no_noise_mean):.6e}")
    print(f"    Noise 2.4e-7:  {np.mean(with_noise_mean):.6e}")
    print(f"    v3 (ext skip): {np.mean(v3_mean):.6e}")

    print(f"\n  Final test accuracy:")
    print(f"    No noise:      {no_noise_res['test_accs'][-1]:.2f}%")
    nn_test = np.load(f"results/dyt_v2/relu_d{DEPTH}/test_accs.npy")
    v3_test = np.load(f"results/dyt_v3/relu_d{DEPTH}/test_accs.npy")
    print(f"    Noise 2.4e-7:  {nn_test[-1]:.2f}%")
    print(f"    v3 (ext skip): {v3_test[-1]:.2f}%")


if __name__ == "__main__":
    main()
