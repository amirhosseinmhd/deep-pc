"""Experiment 3b (Baseline): Gradient norms & per-layer energy during training.

For depth=7 plain MLP (no skip), tracks:
  1. ||grad_W_ℓ||_F  per layer at every training step
  2. Per-layer PC energy at every training step
  3. Snapshots of energy across all layers at 10 evenly-spaced iterations
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exper'))

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_leaves
import equinox as eqx
import optax
import numpy as np
import jpc

from config import (
    SEED, ACT_FNS, WIDTH, ACTIVITY_LR, PARAM_LR,
    BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
)
from common import (
    set_seed, create_model, get_mnist_loaders, evaluate,
    _get_weight_list, ensure_dir
)

DEPTH = 7
N_SNAPSHOTS = 10


def _reorder_energies(energies):
    """Reorder pc_energy_fn(record_layers=True) output to match model layer order.

    pc_energy_fn returns: [output, hidden_1, ..., hidden_{L-2}, input]
    We reorder to:        [input, hidden_1, ..., hidden_{L-2}, output]
    so that layer indices align with model layers and gradient indices.
    """
    return np.concatenate([[energies[-1]], energies[1:-1], [energies[0]]])


def _grad_norms_per_layer(grads, act_fn):
    """Compute Frobenius norm of parameter gradients for each layer.

    grads is a tuple (model_grads, skip_model_grads).
    We extract per-layer weight gradient norms from model_grads.
    """
    model_grads = grads[0]  # list of layer grad pytrees
    norms = []
    for layer_grad in model_grads:
        layer_params = tree_leaves(layer_grad)
        weight_params = [p for p in layer_params
                         if isinstance(p, jnp.ndarray) and p.ndim >= 2]
        if weight_params:
            layer_norm = sum(
                float(jnp.linalg.norm(jnp.ravel(w))) for w in weight_params
            )
            norms.append(layer_norm)
        else:
            norms.append(0.0)
    return norms


def train_with_grad_energy(
    seed, model, skip_model, param_type,
    activity_lr, param_lr, batch_size,
    n_train_iters, test_every, act_fn, depth
):
    """Custom training loop that records gradient norms and per-layer energy."""
    set_seed(seed)

    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    train_loader, test_loader = get_mnist_loaders(batch_size)

    train_losses = []
    test_accs = []
    test_iters = []
    grad_norms_history = []     # (n_iters, n_layers)
    energy_history = []         # (n_iters, n_energy_layers)

    # Determine snapshot iterations (10 evenly spaced)
    snapshot_iters = np.linspace(
        0, n_train_iters - 1, N_SNAPSHOTS, dtype=int
    ).tolist()
    energy_snapshots = {}       # {iter_num: array of per-layer energies}

    data_iter = iter(train_loader)
    for iter_num in range(n_train_iters):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)

        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # Init activities via forward pass
        activities = jpc.init_activities_with_ffwd(
            model=model, input=img_batch,
            skip_model=skip_model, param_type=param_type
        )
        activity_opt_state = activity_optim.init(activities)
        train_loss = float(jpc.mse_loss(activities[-1], label_batch))
        train_losses.append(train_loss)

        # Inference
        for t in range(depth):
            result = jpc.update_pc_activities(
                params=(model, skip_model), activities=activities,
                optim=activity_optim, opt_state=activity_opt_state,
                output=label_batch, input=img_batch,
                param_type=param_type
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        # -- Record per-layer energy (after inference, before param update) --
        layer_energies = jpc.pc_energy_fn(
            params=(model, skip_model),
            activities=activities,
            y=label_batch,
            x=img_batch,
            param_type=param_type,
            record_layers=True
        )
        # Reorder from [output, hidden..., input] to [input, hidden..., output]
        # so layer indices match model layers and gradient indices
        layer_energies_np = _reorder_energies(np.array(layer_energies))
        energy_history.append(layer_energies_np)

        if iter_num in snapshot_iters:
            energy_snapshots[iter_num] = layer_energies_np.copy()

        # Learning (returns grads)
        result = jpc.update_pc_params(
            params=(model, skip_model), activities=activities,
            optim=param_optim, opt_state=param_opt_state,
            output=label_batch, input=img_batch,
            param_type=param_type
        )
        model = result["model"]
        skip_model = result["skip_model"]
        param_opt_state = result["opt_state"]
        grads = result["grads"]

        # -- Record gradient norms per layer --
        layer_grad_norms = _grad_norms_per_layer(grads, act_fn)
        grad_norms_history.append(layer_grad_norms)

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

        if ((iter_num + 1) % test_every) == 0:
            avg_acc = evaluate(model, skip_model, test_loader, param_type)
            test_accs.append(avg_acc)
            test_iters.append(iter_num + 1)
            print(f"  Iter {iter_num+1}, loss={train_loss:.4f}, "
                  f"test acc={avg_acc:.2f}")

    return {
        "train_losses": train_losses,
        "test_accs": test_accs,
        "test_iters": test_iters,
        "grad_norms": np.array(grad_norms_history),
        "energy_per_layer": np.array(energy_history),
        "energy_snapshots": energy_snapshots,
        "snapshot_iters": snapshot_iters,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def plot_grad_norms(grad_norms, title, save_path, sample_every=5):
    """Plot gradient norm per layer vs training iteration.

    grad_norms: (n_iters, n_layers)
    """
    n_iters, n_layers = grad_norms.shape
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, n_layers))
    iters = np.arange(n_iters)
    sampled = iters[::sample_every]

    for l in range(n_layers):
        ax.plot(
            sampled, grad_norms[sampled, l],
            color=cmap[l], linewidth=1.3, alpha=0.85,
            label=f"layer {l+1}"
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\|\nabla_{W_\ell} \mathcal{F}\|_F$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_energy_per_layer(energy, title, save_path, sample_every=5):
    """Plot per-layer energy vs training iteration.

    energy: (n_iters, n_energy_layers)
    """
    n_iters, n_layers = energy.shape
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.magma(np.linspace(0.15, 0.90, n_layers))
    iters = np.arange(n_iters)
    sampled = iters[::sample_every]

    for l in range(n_layers):
        ax.plot(
            sampled, energy[sampled, l],
            color=cmap[l], linewidth=1.3, alpha=0.85,
            label=f"layer {l+1}"
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\mathcal{E}_\ell$  (layer energy)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_energy_snapshots(snapshots, snapshot_iters, title, save_path):
    """Bar/line plot of energy across layers at 10 different training points.

    snapshots: {iter_num: 1-D array of per-layer energies}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(snapshot_iters)))

    for i, it in enumerate(sorted(snapshots.keys())):
        vals = snapshots[it]
        layers = np.arange(1, len(vals) + 1)
        ax.plot(
            layers, vals, 'o-', color=cmap[i],
            markersize=5, linewidth=1.5, alpha=0.85,
            label=f"iter {it}"
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$\mathcal{E}_\ell$  (layer energy)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    save_dir = ensure_dir(
        os.path.join(os.path.dirname(__file__), "results", "exp3b")
    )

    for act_fn in ACT_FNS:
        print(f"\n=== Grad & Energy Monitor | act_fn={act_fn} | depth={DEPTH} ===")
        set_seed(SEED)
        key = jr.PRNGKey(SEED)

        model, skip_model = create_model(
            key, depth=DEPTH, width=WIDTH,
            act_fn=act_fn, param_type="sp", use_skips=False
        )

        res = train_with_grad_energy(
            seed=SEED, model=model, skip_model=skip_model,
            param_type="sp", activity_lr=ACTIVITY_LR,
            param_lr=PARAM_LR, batch_size=BATCH_SIZE,
            n_train_iters=N_TRAIN_ITERS, test_every=TEST_EVERY,
            act_fn=act_fn, depth=DEPTH
        )

        # Save raw data
        np.save(
            os.path.join(save_dir, f"grad_norms_{act_fn}_d{DEPTH}.npy"),
            res["grad_norms"]
        )
        np.save(
            os.path.join(save_dir, f"energy_per_layer_{act_fn}_d{DEPTH}.npy"),
            res["energy_per_layer"]
        )
        np.savez(
            os.path.join(save_dir, f"energy_snapshots_{act_fn}_d{DEPTH}.npz"),
            **{f"iter_{k}": v for k, v in res["energy_snapshots"].items()},
            snapshot_iters=np.array(res["snapshot_iters"])
        )

        # Plot gradient norms per layer over training
        plot_grad_norms(
            res["grad_norms"],
            title=(f"Gradient Norm per Layer — {act_fn}, "
                   f"depth={DEPTH} (MLP, no skip)"),
            save_path=os.path.join(
                save_dir, f"grad_norms_{act_fn}_d{DEPTH}.png"
            )
        )

        # Plot energy per layer over training
        plot_energy_per_layer(
            res["energy_per_layer"],
            title=(f"Per-Layer Energy — {act_fn}, "
                   f"depth={DEPTH} (MLP, no skip)"),
            save_path=os.path.join(
                save_dir, f"energy_per_layer_{act_fn}_d{DEPTH}.png"
            )
        )

        # Plot energy snapshots across layers at 10 training points
        plot_energy_snapshots(
            res["energy_snapshots"],
            res["snapshot_iters"],
            title=(f"Energy Snapshots Across Layers — {act_fn}, "
                   f"depth={DEPTH} (MLP, no skip)"),
            save_path=os.path.join(
                save_dir, f"energy_snapshots_{act_fn}_d{DEPTH}.png"
            )
        )

    print(f"\nExperiment 3b (baseline) complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
