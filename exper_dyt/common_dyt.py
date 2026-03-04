"""Dynamic Tanh (DyT) for Predictive Coding Networks.

Replaces BatchNorm with DyT (Dynamic Tanh) from "Transformers without
Normalization" (Zhu et al., 2025).  DyT is a simple element-wise operation:

    DyT(x) = gamma * tanh(alpha * x) + beta

where alpha is a learnable scalar, gamma/beta are per-channel affine params.
Unlike BN, DyT needs no batch statistics, no freezing, no EMA, and no
train/eval mode distinction — it operates identically per sample.

Each layer can optionally disable DyT (pass-through identity) via the
`use_dyt` flag, useful for experimenting with partial DyT placement.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jpc

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import equinox as eqx
import equinox.nn as nn
import optax

import numpy as np
from typing import List, Callable

# Constants from config, utilities from common
from config import (
    INPUT_DIM, OUTPUT_DIM, WIDTH, SEED,
    ACTIVITY_LR, PARAM_LR, BATCH_SIZE, N_TRAIN_ITERS, TEST_EVERY,
    DEPTHS, ACT_FNS, RESULTS_DIR,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exper'))
from common import (
    set_seed, get_mnist_loaders, ensure_dir, _selected_layer_indices,
    unwrap_hessian_pytree,
)

DYT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ============================================================================
# DyT Layer (Dynamic Tanh — replaces BatchNorm)
# ============================================================================
class DyTLayer(eqx.Module):
    """Dynamic Tanh (DyT) — a drop-in replacement for normalization layers.

    DyT(x) = gamma * tanh(alpha * x) + beta

    - alpha: learnable scalar (one per DyT instance), controls squashing width
    - gamma: per-channel learnable scale (init 1)
    - beta:  per-channel learnable shift (init 0)
    - use_dyt: when False, acts as identity (pass-through), allowing
               per-layer toggling for ConvNet-like architectures.
    """
    alpha: jnp.ndarray         # learnable scalar
    gamma: jnp.ndarray         # per-channel scale
    beta: jnp.ndarray          # per-channel shift
    use_dyt: bool = eqx.static_field()

    def __init__(self, num_features, *, init_alpha=0.5, use_dyt=True):
        self.alpha = jnp.ones(1) * init_alpha
        self.gamma = jnp.ones(num_features)
        self.beta = jnp.zeros(num_features)
        self.use_dyt = use_dyt

    def __call__(self, x):
        # x shape: (features,)  [called under vmap per sample]
        if not self.use_dyt:
            return x
        return self.gamma * jnp.tanh(self.alpha * x) + self.beta


# ============================================================================
# Layer modules (jpc-compatible: each layer is a callable  x -> y)
# ============================================================================
class InputLayer(eqx.Module):
    """First layer: Linear (no activation, no DyT, no skip)."""
    linear: nn.Linear

    def __call__(self, x):
        return self.linear(x)


class HiddenLayerDyT(eqx.Module):
    """Hidden layer: act_fn -> Linear + skip -> DyT (DyT after skip)."""
    act_fn: Callable = eqx.static_field()
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.linear(h) + x   # linear + skip first
        return self.dyt(h)        # DyT squashes full output


class OutputLayer(eqx.Module):
    """Output layer: act_fn -> DyT -> Linear (no skip)."""
    act_fn: Callable = eqx.static_field()
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.dyt(h)
        return self.linear(h)


# ============================================================================
# Full model
# ============================================================================
class FCResNetDyT(eqx.Module):
    """Fully-connected ResNet with DyT at every hidden layer.

    Compatible with jpc: supports len(), indexing, and each layer is callable.
    Uses param_type="sp" and skip_model=None (skips are internal).

    Args:
        dyt_enabled_layers: controls which layers get DyT.
            - None or "all": DyT at every hidden + output layer (default)
            - list of ints: only those layer indices get DyT (0-indexed)
            - "every_n:K": DyT at every K-th layer (1-indexed)
        init_alpha: initial value for DyT's alpha parameter (default 0.5)
    """
    layers: List[eqx.Module]

    def __init__(self, *, key, in_dim, width, depth, out_dim,
                 act_fn="tanh", init_alpha=0.5, dyt_enabled_layers=None):
        act = jpc.get_act_fn(act_fn)
        keys = jr.split(key, depth)
        self.layers = []

        # Determine which layers get DyT enabled
        if dyt_enabled_layers is None or dyt_enabled_layers == "all":
            dyt_mask = {i: True for i in range(depth)}
        elif isinstance(dyt_enabled_layers, str) and dyt_enabled_layers.startswith("every_n:"):
            n = int(dyt_enabled_layers.split(":")[1])
            dyt_mask = {i: (i % n == 0) for i in range(depth)}
        else:
            dyt_mask = {i: (i in dyt_enabled_layers) for i in range(depth)}

        # Layer 0: input -> hidden (no act, no DyT, no skip)
        self.layers.append(InputLayer(
            linear=nn.Linear(in_dim, width, use_bias=False, key=keys[0])
        ))

        # Layers 1 .. depth-2: hidden -> hidden (act + DyT + linear + skip)
        for i in range(1, depth - 1):
            self.layers.append(HiddenLayerDyT(
                act_fn=act,
                dyt=DyTLayer(width, init_alpha=init_alpha,
                             use_dyt=dyt_mask.get(i, True)),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

        # Last layer: hidden -> output (act + DyT + linear, no skip)
        self.layers.append(OutputLayer(
            act_fn=act,
            dyt=DyTLayer(width, init_alpha=init_alpha,
                         use_dyt=dyt_mask.get(depth - 1, True)),
            linear=nn.Linear(width, out_dim, use_bias=False, key=keys[-1]),
        ))

    def __call__(self, x):
        for f in self.layers:
            x = f(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]


# ============================================================================
# Forward-pass activity init
# ============================================================================
def init_activities_dyt(model, x_batch):
    """Initialise activities via a forward pass.

    With DyT there are no batch statistics to compute or freeze —
    each layer operates element-wise per sample.
    """
    activities = []
    h = x_batch
    for layer in model.layers:
        h = vmap(layer)(h)
        activities.append(h)
    return activities


# ============================================================================
# Evaluation (DyT has no train/eval distinction)
# ============================================================================
def evaluate_dyt(model, test_loader):
    """Evaluate test accuracy. DyT is the same at train and test time."""
    avg_acc = 0.0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        preds = vmap(model)(img_batch)
        acc = float(jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)) * 100)
        avg_acc += acc
    return avg_acc / len(test_loader)


# ============================================================================
# DyT Training loop
# ============================================================================
def train_dyt_and_record(
    seed, model, depth,
    activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
    batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
    test_every=TEST_EVERY, act_fn="tanh",
    track_weight_updates=False,
    track_activity_norms=False,
):
    """Train a DyT-PCN and record metrics.

    The model is an FCResNetDyT instance. We use param_type="sp" and
    skip_model=None since ResNet skips are internal to the model.

    Compared to the BF training loop, this is simpler because DyT
    needs no batch-stats computation, freezing, or EMA updates.
    """
    set_seed(seed)

    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)

    param_opt_state = param_optim.init(
        (eqx.filter(model.layers, eqx.is_array), None)
    )

    train_loader, test_loader = get_mnist_loaders(batch_size)

    train_losses = []
    test_accs = []
    test_iters = []

    weight_update_norms = [] if track_weight_updates else None
    activity_norms_init = [] if track_activity_norms else None
    activity_norms_post = [] if track_activity_norms else None

    def _get_weights(m):
        """Extract weight arrays for update tracking."""
        from jax.tree_util import tree_leaves
        return [p for p in tree_leaves(m) if isinstance(p, jnp.ndarray) and p.ndim >= 2]

    data_iter = iter(train_loader)
    for iter_num in range(n_train_iters):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)

        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # ---- Step 1: Forward pass to init activities ----
        activities = init_activities_dyt(model, img_batch)

        train_loss = float(jpc.mse_loss(activities[-1], label_batch))
        train_losses.append(train_loss)

        # Record activity norms at init
        if track_activity_norms:
            norms = [float(jnp.linalg.norm(a, axis=1, ord=2).mean()) for a in activities]
            activity_norms_init.append(norms)

        # Snapshot weights for update tracking
        if track_weight_updates:
            old_weights = [jnp.array(w) for w in _get_weights(model)]

        # ---- Step 2: PC Inference (no frozen stats needed with DyT) ----
        activity_opt_state = activity_optim.init(activities)
        for t in range(depth):
            result = jpc.update_pc_activities(
                params=(model.layers, None),
                activities=activities,
                optim=activity_optim,
                opt_state=activity_opt_state,
                output=label_batch,
                input=img_batch,
                param_type="sp"
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        # Record post-inference norms
        if track_activity_norms:
            norms = [float(jnp.linalg.norm(a, axis=1, ord=2).mean()) for a in activities]
            activity_norms_post.append(norms)

        # ---- Step 3: Learning — update W and DyT affine params ----
        result = jpc.update_pc_params(
            params=(model.layers, None),
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=label_batch,
            input=img_batch,
            param_type="sp"
        )
        updated_layers = result["model"]
        param_opt_state = result["opt_state"]

        # Reconstruct the full model with updated layers
        model = eqx.tree_at(lambda m: m.layers, model, updated_layers)

        # (No EMA update needed — DyT has no running statistics)

        # Weight update tracking
        if track_weight_updates:
            new_weights = _get_weights(model)
            update_norms_iter = []
            for w_old, w_new in zip(old_weights, new_weights):
                update_norms_iter.append(
                    float(jnp.linalg.norm(jnp.ravel(w_new - w_old)))
                )
            weight_update_norms.append(update_norms_iter)

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

        if ((iter_num + 1) % test_every) == 0:
            avg_acc = evaluate_dyt(model, test_loader)
            test_accs.append(avg_acc)
            test_iters.append(iter_num + 1)
            print(f"  Iter {iter_num+1}, loss={train_loss:.4f}, "
                  f"test acc={avg_acc:.2f}")

    out = {
        "train_losses": train_losses,
        "test_accs": test_accs,
        "test_iters": test_iters,
    }
    if track_weight_updates:
        out["weight_update_norms"] = np.array(weight_update_norms)
    if track_activity_norms:
        out["activity_norms_init"] = np.array(activity_norms_init)
        out["activity_norms_post"] = np.array(activity_norms_post)
    return out


# ============================================================================
# Condition number for DyT model
# ============================================================================
def compute_condition_number_dyt(model, x, y):
    """Compute kappa(H_z) for a DyT model at init."""
    activities = init_activities_dyt(model, x)

    hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
        (model.layers, None), activities, y,
        x=x, param_type="sp"
    )
    H = unwrap_hessian_pytree(hessian_pytree, activities)
    eigenvals = jnp.linalg.eigvalsh(H)
    lam_max = jnp.abs(eigenvals[-1])
    lam_min = jnp.abs(eigenvals[0])
    cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
    return cond, eigenvals
