"""Diagnose per-layer energy dissipation during inference across variants.

Uses float64 for maximum precision. Runs inference up to 50000 iterations
and produces publication-quality plots showing per-layer error norms.

Compares: baseline (no skip), resnet (external skip), mupc (external skip + scaling),
          dyt_v2 (internal skip), dyt_v3 (external skip)
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import optax
import equinox as eqx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

import sys
sys.path.insert(0, os.path.dirname(__file__))
import jpc

from variants.dyt import FCResNetDyT
from variants.dyt_v3 import FCResNetDyT_v3
from variants.batchfreezing import FCResNetBN, _compute_batch_stats, freeze_batch_stats
from variants.batchfreezing_v2 import FCResNetBN_v2, _compute_batch_stats_v2, _freeze_batch_stats_v2

# ── Config ──
DEPTH = 50
WIDTH = 64
BATCH = 4
IN_DIM = 784
OUT_DIM = 10
SEED = 42
INFERENCE_LR = 0.01

# Checkpoints for recording (dense near start, sparse later)
CHECKPOINTS = sorted(set(
    list(range(0, 100, 1)) +
    list(range(100, 1000, 10)) +
    list(range(1000, 10001, 100))
))
MAX_ITERS = max(CHECKPOINTS)

# Save directory
SAVE_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(SAVE_DIR, exist_ok=True)


def make_fake_data(key):
    k1, k2 = jr.split(key)
    x = jr.normal(k1, (BATCH, IN_DIM))
    y_idx = jr.randint(k2, (BATCH,), 0, OUT_DIM)
    y = jax.nn.one_hot(y_idx, OUT_DIM)
    return x, y


def compute_per_layer_errors(model_layers, skip_model, activities, x, y,
                              param_type="sp"):
    """Compute ||ε_l||_rms at each layer. Returns list ordered [input, ..., output]."""
    from jpc._core._energies import _get_param_scalings

    skip_list = [None] * len(model_layers) if skip_model is None else skip_model

    scalings = _get_param_scalings(
        model=model_layers, input=x, skip_model=skip_list,
        param_type=param_type
    )

    n_hidden = len(model_layers) - 1
    errors = []

    # Layer 0: input layer error  (ε_0 = z_0 - f_0(x))
    e1 = activities[0] - scalings[0] * vmap(model_layers[0])(x)
    errors.append(float(jnp.sqrt(jnp.mean(e1 ** 2))))

    # Layers 1..L-2: hidden errors in order (near-input → near-output)
    for net_l in range(1, n_hidden):
        act_l = net_l
        err = (activities[act_l]
               - scalings[net_l] * vmap(model_layers[net_l])(activities[act_l - 1]))
        if skip_list[net_l] is not None:
            err -= vmap(skip_list[net_l])(activities[act_l - 1])
        errors.append(float(jnp.sqrt(jnp.mean(err ** 2))))

    # Layer L: output layer error  (ε_L = y - f_L(z_{L-1}))
    eL = y - scalings[-1] * vmap(model_layers[-1])(activities[-2])
    errors.append(float(jnp.sqrt(jnp.mean(eL ** 2))))

    # errors[0]=input layer, errors[-1]=output layer
    return errors


def run_inference(model_layers, skip_model, activities, x, y,
                  param_type="sp", name=""):
    """Run PC inference and record per-layer errors at all checkpoints."""
    optim = optax.sgd(INFERENCE_LR)
    opt_state = optim.init(activities)
    params = (model_layers, skip_model)

    checkpoint_set = set(CHECKPOINTS)
    results = {}
    log_interval = 5000

    for t in range(MAX_ITERS + 1):
        if t in checkpoint_set:
            errs = compute_per_layer_errors(
                model_layers, skip_model, activities, x, y, param_type
            )
            results[t] = errs

        if t > 0 and t % log_interval == 0:
            print(f"    [{name}] iter {t}/{MAX_ITERS}")

        if t < MAX_ITERS:
            out = jpc.update_pc_activities(
                params=params,
                activities=activities,
                optim=optim,
                opt_state=opt_state,
                output=y,
                input=x,
                param_type=param_type,
            )
            activities = out["activities"]
            opt_state = out["opt_state"]

    return results


def results_to_array(results, n_layers):
    """Convert results dict to (n_checkpoints, n_layers) array + time axis."""
    times = sorted(results.keys())
    arr = np.zeros((len(times), n_layers))
    for i, t in enumerate(times):
        arr[i, :] = results[t]
    return np.array(times), arr


# ══════════════════════════════════════════════════════════════════════
# Build all models
# ══════════════════════════════════════════════════════════════════════
key = jr.PRNGKey(SEED)
keys = jr.split(key, 9)
k_data = keys[0]
k_base, k_resnet, k_mupc = keys[1], keys[2], keys[3]
k_dyt, k_dytv3 = keys[4], keys[5]
k_bf1, k_bf2 = keys[6], keys[7]
x, y = make_fake_data(k_data)

print(f"Config: depth={DEPTH}, width={WIDTH}, batch={BATCH}, dtype={x.dtype}")
print(f"Inference LR: {INFERENCE_LR}, max iters: {MAX_ITERS}")
print(f"Number of checkpoints: {len(CHECKPOINTS)}")

variants = {}

# 1. Baseline
print("\n>>> Building Baseline (no skip)...")
baseline_model = jpc.make_mlp(
    key=k_base, input_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    output_dim=OUT_DIM, act_fn="relu", use_bias=False, param_type="sp",
)
baseline_acts = jpc.init_activities_with_ffwd(
    model=baseline_model, input=x, skip_model=None, param_type="sp",
)
variants["Baseline\n(no skip)"] = (baseline_model, None, baseline_acts, "sp")

# 2. ResNet
print(">>> Building ResNet (external skip)...")
resnet_model = jpc.make_mlp(
    key=k_resnet, input_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    output_dim=OUT_DIM, act_fn="relu", use_bias=False, param_type="sp",
)
resnet_skip = jpc.make_skip_model(DEPTH)
resnet_acts = jpc.init_activities_with_ffwd(
    model=resnet_model, input=x, skip_model=resnet_skip, param_type="sp",
)
variants["ResNet\n(ext. skip)"] = (resnet_model, resnet_skip, resnet_acts, "sp")

# 3. μPC
print(">>> Building μPC (external skip + mupc scaling)...")
mupc_model = jpc.make_mlp(
    key=k_mupc, input_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    output_dim=OUT_DIM, act_fn="relu", use_bias=False, param_type="mupc",
)
mupc_skip = jpc.make_skip_model(DEPTH)
mupc_acts = jpc.init_activities_with_ffwd(
    model=mupc_model, input=x, skip_model=mupc_skip, param_type="mupc",
)
variants["μPC\n(ext. skip)"] = (mupc_model, mupc_skip, mupc_acts, "mupc")

# 4. DyT v1 (internal skip: DyT wraps linear+skip)
print(">>> Building DyT v1 (internal skip)...")
dyt_model = FCResNetDyT(
    key=k_dyt, in_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    out_dim=OUT_DIM, act_fn="relu", init_alpha=0.5,
)
dyt_acts = []
h = x
for layer in dyt_model.layers:
    h = vmap(layer)(h)
    dyt_acts.append(h)
variants["DyT v1\n(int. skip)"] = (dyt_model.layers, None, dyt_acts, "sp")

# 5. DyT v3 (external skip)
print(">>> Building DyT v3 (external skip)...")
dytv3_model = FCResNetDyT_v3(
    key=k_dytv3, in_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    out_dim=OUT_DIM, act_fn="relu", init_alpha=0.5,
)
dytv3_skip = jpc.make_skip_model(DEPTH)
dytv3_acts = jpc.init_activities_with_ffwd(
    model=dytv3_model.layers, input=x, skip_model=dytv3_skip, param_type="sp",
)
variants["DyT v3\n(ext. skip)"] = (dytv3_model.layers, dytv3_skip, dytv3_acts, "sp")

# 6. BF v1 (BN wraps linear+skip, internal skip)
print(">>> Building BF v1 (internal skip)...")
bf1_model = FCResNetBN(
    key=k_bf1, in_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    out_dim=OUT_DIM, act_fn="relu",
)
bf1_stats, _ = _compute_batch_stats(bf1_model, x)
bf1_frozen = freeze_batch_stats(bf1_model, bf1_stats)
bf1_acts = []
h = x
for layer in bf1_frozen.layers:
    h = vmap(layer)(h)
    bf1_acts.append(h)
variants["BF v1\n(int. skip)"] = (bf1_frozen.layers, None, bf1_acts, "sp")

# 7. BF v2 (BN wraps linear only, skip after, internal skip)
print(">>> Building BF v2 (internal skip)...")
bf2_model = FCResNetBN_v2(
    key=k_bf2, in_dim=IN_DIM, width=WIDTH, depth=DEPTH,
    out_dim=OUT_DIM, act_fn="relu",
)
bf2_stats, _ = _compute_batch_stats_v2(bf2_model, x)
bf2_frozen = _freeze_batch_stats_v2(bf2_model, bf2_stats)
bf2_acts = []
h = x
for layer in bf2_frozen.layers:
    h = vmap(layer)(h)
    bf2_acts.append(h)
variants["BF v2\n(int. skip)"] = (bf2_frozen.layers, None, bf2_acts, "sp")


# ══════════════════════════════════════════════════════════════════════
# Run inference for all variants
# ══════════════════════════════════════════════════════════════════════
all_results = {}
for vname, (model, skip, acts, ptype) in variants.items():
    short_name = vname.replace('\n', ' ')
    print(f"\n>>> Running inference: {short_name}...")
    all_results[vname] = run_inference(model, skip, acts, x, y, ptype, short_name)
    print(f"    Done.")

n_layers = DEPTH  # number of layers (activities)


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Heatmaps — layer (y) vs iteration (x), color = log10(error)
# ══════════════════════════════════════════════════════════════════════
print("\n>>> Generating heatmap plots...")

n_var = len(all_results)
ncols = 4
nrows = (n_var + ncols - 1) // ncols  # ceil division

# Use gridspec: nrows of heatmaps + a thin row for the colorbar
fig = plt.figure(figsize=(4.2 * ncols, 5.5 * nrows + 0.6))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig,
                       height_ratios=[1] * nrows + [0.05],
                       hspace=0.35, wspace=0.25)

FLOOR = 1e-20
vmin, vmax = 1e-18, 1e1
im = None

variant_names = list(all_results.keys())
for idx, vname in enumerate(variant_names):
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])
    results = all_results[vname]
    times, err_arr = results_to_array(results, n_layers)
    err_arr_clamped = np.where(err_arr > 0, err_arr, FLOOR)

    im = ax.pcolormesh(
        times, np.arange(n_layers), err_arr_clamped.T,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap='inferno', shading='auto', rasterized=True
    )

    ax.set_xlabel("Inference iteration", fontsize=10)
    if col == 0:
        ax.set_ylabel("Layer index", fontsize=10)
    ax.set_title(vname, fontsize=10, fontweight='bold')
    ax.set_xscale('symlog', linthresh=100)
    ax.invert_yaxis()


# Hide unused subplot slots in the last row
for idx in range(n_var, nrows * ncols):
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])
    ax.set_visible(False)

# Colorbar spanning the bottom row
cbar_ax = fig.add_subplot(gs[-1, :])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label("RMS prediction error  ||ε_l||", fontsize=11)

fig.suptitle(
    f"Per-layer error during PC inference (depth={DEPTH}, width={WIDTH}, float64)",
    fontsize=14, fontweight='bold', y=1.01
)
path1 = os.path.join(SAVE_DIR, "energy_dissipation_heatmaps.png")
fig.savefig(path1, dpi=200, bbox_inches='tight')
print(f"    Saved: {path1}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Error profile across layers at fixed inference iterations
# ══════════════════════════════════════════════════════════════════════
print(">>> Generating layer-profile plots...")

snapshot_iters = [0, 100, 500, 1000, 2000, 5000, 10000]
# Filter to what we actually have
snapshot_iters = [t for t in snapshot_iters if t <= MAX_ITERS]

n_variants = len(all_results)
fig, axes = plt.subplots(len(snapshot_iters), 1,
                          figsize=(10, 3.0 * len(snapshot_iters)),
                          sharex=True)
if len(snapshot_iters) == 1:
    axes = [axes]

colors = {
    "Baseline\n(no skip)": '#1f77b4',
    "ResNet\n(ext. skip)": '#ff7f0e',
    "μPC\n(ext. skip)": '#2ca02c',
    "DyT v1\n(int. skip)": '#d62728',
    "DyT v3\n(ext. skip)": '#9467bd',
    "BF v1\n(int. skip)": '#8c564b',
    "BF v2\n(int. skip)": '#e377c2',
}

for row, T in enumerate(snapshot_iters):
    ax = axes[row]
    layers = np.arange(n_layers)

    for vname, results in all_results.items():
        # Find closest checkpoint
        available = sorted(results.keys())
        t_actual = min(available, key=lambda t: abs(t - T))
        errs = np.array(results[t_actual])
        # Replace 0 with NaN for log scale
        errs_plot = np.where(errs > 0, errs, np.nan)

        label = vname.replace('\n', ' ')
        ax.semilogy(layers, errs_plot, '-o', markersize=2, linewidth=1.5,
                     color=colors.get(vname, 'black'), label=label, alpha=0.85)

    ax.set_ylabel("||ε_l||  (RMS)", fontsize=10)
    ax.set_title(f"T = {T} inference iterations", fontsize=11, fontweight='bold')
    ax.set_ylim(1e-20, 1e1)
    ax.grid(True, alpha=0.3)
    ax.axhline(2.4e-7, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(n_layers - 1, 3e-7, 'float32 ε', fontsize=7, color='gray',
            ha='right', va='bottom')

    if row == 0:
        ax.legend(fontsize=8, ncol=n_variants, loc='upper center',
                  bbox_to_anchor=(0.5, 1.45))

axes[-1].set_xlabel("Layer index (0=input, 49=output)", fontsize=11)

fig.suptitle(
    f"Error profile across layers at different inference budgets\n"
    f"(depth={DEPTH}, width={WIDTH}, float64)",
    fontsize=13, fontweight='bold', y=1.02
)
fig.tight_layout()
path2 = os.path.join(SAVE_DIR, "energy_profile_by_iteration.png")
fig.savefig(path2, dpi=200, bbox_inches='tight')
print(f"    Saved: {path2}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Error at specific layers as a function of iteration
# ══════════════════════════════════════════════════════════════════════
print(">>> Generating per-layer convergence plots...")

track_layers = [0, 5, 10, 20, 30, 40, 48]
track_layers = [l for l in track_layers if l < n_layers]

fig, axes = plt.subplots(len(track_layers), 1,
                          figsize=(10, 2.5 * len(track_layers)),
                          sharex=True)

for row, layer_idx in enumerate(track_layers):
    ax = axes[row]

    for vname, results in all_results.items():
        times, err_arr = results_to_array(results, n_layers)
        vals = err_arr[:, layer_idx]
        vals_plot = np.where(vals > 0, vals, np.nan)

        label = vname.replace('\n', ' ')
        ax.semilogy(times, vals_plot, linewidth=1.5,
                     color=colors.get(vname, 'black'), label=label, alpha=0.85)

    layer_label = f"Layer {layer_idx}" if layer_idx < n_layers - 1 else "Output"
    ax.set_ylabel(f"||ε_{{{layer_idx}}}||", fontsize=10)
    ax.set_title(layer_label, fontsize=11, fontweight='bold')
    ax.set_ylim(1e-20, 1e1)
    ax.set_xscale('symlog', linthresh=100)
    ax.grid(True, alpha=0.3)
    ax.axhline(2.4e-7, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # Mark typical inference budgets
    for Tmark in [100, 200]:
        ax.axvline(Tmark, color='gray', linestyle='--', alpha=0.3)

    if row == 0:
        ax.legend(fontsize=8, ncol=n_variants, loc='upper center',
                  bbox_to_anchor=(0.5, 1.55))

axes[-1].set_xlabel("Inference iteration", fontsize=11)

fig.suptitle(
    f"Error convergence at selected layers\n"
    f"(depth={DEPTH}, width={WIDTH}, float64)",
    fontsize=13, fontweight='bold', y=1.01
)
fig.tight_layout()
path3 = os.path.join(SAVE_DIR, "energy_convergence_per_layer.png")
fig.savefig(path3, dpi=200, bbox_inches='tight')
print(f"    Saved: {path3}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  SUMMARY at T={MAX_ITERS}")
print(f"{'='*90}")
print(f"\n{'Variant':<22s} | {'L0':>12s} | {'L10':>12s} | {'L20':>12s} | {'L30':>12s} | {'L40':>12s} | {'Out':>12s}")
print("-" * 100)
for vname, results in all_results.items():
    errs = results[MAX_ITERS]
    label = vname.replace('\n', ' ')
    row = f"{label:<22s}"
    for l in [0, 10, 20, 30, 40, n_layers - 1]:
        v = errs[l]
        if v == 0.0:
            row += f" | {'0':>12s}"
        else:
            row += f" | {v:>12.4e}"
    print(row)

print(f"\nAll plots saved to: {SAVE_DIR}/")
print("Done.")
