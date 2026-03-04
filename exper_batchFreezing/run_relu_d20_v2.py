"""Re-run relu depth=20 with BN-after-skip fix — all metrics."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.random as jr
import numpy as np
from common_bf import (
    FCResNetBN, train_bf_and_record, compute_condition_number_bf,
    ensure_dir, set_seed, get_mnist_loaders,
    INPUT_DIM, OUTPUT_DIM, WIDTH, SEED,
    ACTIVITY_LR, PARAM_LR, BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
    BF_RESULTS_DIR, _selected_layer_indices,
)
from plot_results import (
    plot_performance_curves, plot_train_loss,
    plot_condition_numbers, plot_condition_bar,
    plot_weight_updates, plot_weight_updates_summary,
    plot_latent_norms_vs_layer, plot_latent_norms_vs_training,
)
PARAM_LR = 5e-03
ACTIVITY_LR = 1e-03
save_dir = ensure_dir(os.path.join(BF_RESULTS_DIR, "exp4_v2_d40"))
act_fn, depth = "relu", 20

# ================================================================
# 1. Performance + Weight Updates + Activity Norms (single run)
# ================================================================
print(f"\n=== BF (BN-after-skip) | {act_fn} | depth={depth} ===")
set_seed(SEED)
key = jr.PRNGKey(SEED)
model = FCResNetBN(
    key=key, in_dim=INPUT_DIM, width=WIDTH, depth=depth,
    out_dim=OUTPUT_DIM, act_fn=act_fn, use_frozen=True
)

res = train_bf_and_record(
    seed=SEED, model=model, depth=depth,
    activity_lr=ACTIVITY_LR  , param_lr=PARAM_LR  ,
    batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
    test_every=TEST_EVERY, act_fn=act_fn,
    track_weight_updates=True,
    track_activity_norms=True,
)

# --- Performance plots ---
depth_results = {depth: res}
plot_performance_curves(
    depth_results,
    title=f"BF (BN-after-skip) Performance — {act_fn}",
    save_path=os.path.join(save_dir, f"performance_{act_fn}_d{depth}.png"),
)
plot_train_loss(
    depth_results,
    title=f"BF (BN-after-skip) Train Loss — {act_fn}",
    save_path=os.path.join(save_dir, f"train_loss_{act_fn}_d{depth}.png"),
)

# --- Weight update plots ---
update_norms = res["weight_update_norms"]
n_layers_tracked = update_norms.shape[1]
layer_idxs = _selected_layer_indices(depth)
layer_idxs = [i for i in layer_idxs if i < n_layers_tracked]
selected = update_norms[:, layer_idxs]
layer_labels_w = [f"layer {i+1}" for i in layer_idxs]

np.save(os.path.join(save_dir, f"update_norms_{act_fn}_d{depth}.npy"), update_norms)

plot_weight_updates(
    selected, layer_labels_w,
    title=f"BF (BN-after-skip) Weight Updates — {act_fn}, depth={depth}",
    save_path=os.path.join(save_dir, f"weight_updates_{act_fn}_d{depth}.png"),
)

# --- Activity norm plots ---
norms_init = res["activity_norms_init"]
norms_post = res["activity_norms_post"]

np.save(os.path.join(save_dir, f"norms_init_{act_fn}_d{depth}.npy"), norms_init)
np.save(os.path.join(save_dir, f"norms_post_{act_fn}_d{depth}.npy"), norms_post)

plot_latent_norms_vs_layer(
    {"init (pre-inference)": norms_init[0], "post-inference": norms_post[0]},
    title=f"BF (BN-after-skip) Norms vs Layer — {act_fn}, depth={depth}",
    save_path=os.path.join(save_dir, f"norms_vs_layer_{act_fn}_d{depth}.png"),
)

n_activities = norms_init.shape[1]
act_layer_idxs = [i for i in _selected_layer_indices(depth) if i < n_activities]
layer_labels_a = [(i, f"layer {i+1}") for i in act_layer_idxs]

plot_latent_norms_vs_training(
    norms_init, layer_labels_a,
    title=f"BF (BN-after-skip) Init Norms — {act_fn}, depth={depth}",
    save_path=os.path.join(save_dir, f"norms_training_init_{act_fn}_d{depth}.png"),
    sample_every=max(1, len(norms_init) // 200),
)
plot_latent_norms_vs_training(
    norms_post, layer_labels_a,
    title=f"BF (BN-after-skip) Post-Inference Norms — {act_fn}, depth={depth}",
    save_path=os.path.join(save_dir, f"norms_training_post_{act_fn}_d{depth}.png"),
    sample_every=max(1, len(norms_post) // 200),
)

# ================================================================
# 2. Condition Number
# ================================================================
print(f"\n=== Condition Number | {act_fn} | depth={depth} ===")
COND_WIDTH = 16
jax.config.update("jax_enable_x64", True)
set_seed(SEED)
train_loader, _ = get_mnist_loaders(1)
x_sample, y_sample = next(iter(train_loader))
x_sample, y_sample = x_sample.numpy(), y_sample.numpy()

key = jr.PRNGKey(SEED)
cond_model = FCResNetBN(
    key=key, in_dim=INPUT_DIM, width=COND_WIDTH, depth=depth,
    out_dim=OUTPUT_DIM, act_fn=act_fn, use_frozen=True
)
try:
    cond, _ = compute_condition_number_bf(cond_model, x_sample, y_sample)
    print(f"  κ(H_z) = {cond:.2e}")
    results = {"BF ResNet": [cond]}
    plot_condition_bar(
        results, [depth],
        title=f"Condition Number — {act_fn} (BF, BN-after-skip)",
        save_path=os.path.join(save_dir, f"condition_bar_{act_fn}_d{depth}.png"),
    )
except Exception as e:
    print(f"  Condition number FAILED: {e}")

print(f"\nDone! All results in {save_dir}")
