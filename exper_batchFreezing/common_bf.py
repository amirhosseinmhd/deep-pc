"""BatchNorm Freezing (BF) for Predictive Coding Networks.

Implements a custom FC-ResNet with BatchNorm at every hidden layer,
compatible with jpc's functional energy / update API.

Key idea (BF algorithm):
  1. Forward-pass init: compute per-layer batch statistics (mean, var)
  2. Freeze these stats into the model (stored as array fields)
  3. PC inference loop: activities updated with BN using frozen batch stats
  4. Learning step: update weights W and BN affine params (gamma, beta)
  5. Update EMA running stats once from the batch stats in step 1

Since jpc uses vmap(model[l])(x_single_sample), BN normalisation per-sample
uses the *pre-computed* batch-level mean/var stored in the model fields.
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

BF_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ============================================================================
# BatchNorm Layer (stateless — stores frozen stats as array fields)
# ============================================================================
class BatchNormLayer(eqx.Module):
    """BatchNorm that uses pre-computed (frozen) batch statistics.

    When use_frozen=True (set as a static field), normalisation uses
    frozen_mean / frozen_var which are plain array fields.
    When use_frozen=False, it uses running_mean / running_var (for test time).
    """
    weight: jnp.ndarray       # gamma (learnable scale)
    bias: jnp.ndarray         # beta  (learnable shift)
    running_mean: jnp.ndarray  # EMA mean  (updated once per batch in learning phase)
    running_var: jnp.ndarray   # EMA var   (updated once per batch in learning phase)
    frozen_mean: jnp.ndarray   # batch stats frozen for inference
    frozen_var: jnp.ndarray    # batch stats frozen for inference
    use_frozen: bool = eqx.static_field()
    momentum: float = eqx.static_field()
    eps: float = eqx.static_field()

    def __init__(self, num_features, *, momentum=0.1, eps=1e-5, use_frozen=True):
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.running_mean = jnp.zeros(num_features)
        self.running_var = jnp.ones(num_features)
        self.frozen_mean = jnp.zeros(num_features)
        self.frozen_var = jnp.ones(num_features)
        self.use_frozen = use_frozen
        self.momentum = momentum
        self.eps = eps

    def __call__(self, x):
        # x shape: (features,)  [called under vmap per sample]
        if self.use_frozen:
            mean, var = self.frozen_mean, self.frozen_var
        else:
            # test mode: use running stats
            mean, var = self.running_mean, self.running_var
        return self.weight * (x - mean) / jnp.sqrt(var + self.eps) + self.bias


# ============================================================================
# Layer modules (jpc-compatible: each layer is a callable  x -> y)
# ============================================================================
class InputLayer(eqx.Module):
    """First layer: Linear (no activation, no BN, no skip)."""
    linear: nn.Linear

    def __call__(self, x):
        return self.linear(x)


class HiddenLayerBN(eqx.Module):
    """Hidden layer: act_fn -> Linear + skip -> BN (BN after skip)."""
    act_fn: Callable = eqx.static_field()
    bn: BatchNormLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.linear(h) + x   # linear + skip first
        return self.bn(h)         # BN normalizes full output


class OutputLayer(eqx.Module):
    """Output layer: act_fn -> BN -> Linear (no skip)."""
    act_fn: Callable = eqx.static_field()
    bn: BatchNormLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.bn(h)
        return self.linear(h)


# ============================================================================
# Full model
# ============================================================================
class FCResNetBN(eqx.Module):
    """Fully-connected ResNet with BatchNorm at every hidden layer.

    Compatible with jpc: supports len(), indexing, and each layer is callable.
    Uses param_type="sp" and skip_model=None (skips are internal).
    """
    layers: List[eqx.Module]

    def __init__(self, *, key, in_dim, width, depth, out_dim,
                 act_fn="tanh", use_frozen=True):
        act = jpc.get_act_fn(act_fn)
        keys = jr.split(key, depth)
        self.layers = []

        # Layer 0: input -> hidden (no act, no BN, no skip)
        self.layers.append(InputLayer(
            linear=nn.Linear(in_dim, width, use_bias=False, key=keys[0])
        ))

        # Layers 1 .. depth-2: hidden -> hidden (act + BN + linear + skip)
        for i in range(1, depth - 1):
            self.layers.append(HiddenLayerBN(
                act_fn=act,
                bn=BatchNormLayer(width, use_frozen=use_frozen),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

        # Last layer: hidden -> output (act + BN + linear, no skip)
        self.layers.append(OutputLayer(
            act_fn=act,
            bn=BatchNormLayer(width, use_frozen=use_frozen),
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
# Helper: compute batch statistics per layer via a forward pass
# ============================================================================
def _compute_batch_stats(model, x_batch):
    """Run a forward pass over the batch and collect per-layer BN statistics.

    Returns a list of (mean, var) tuples — one per layer that has a BN.
    Layers without BN (InputLayer) get None.
    """
    stats = []
    h = x_batch  # (batch, features)
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            h = vmap(layer)(h)
            stats.append(None)
        elif isinstance(layer, HiddenLayerBN):
            # HiddenLayerBN: act -> linear + skip -> BN
            h_act = vmap(layer.act_fn)(h)
            h_pre_bn = vmap(layer.linear)(h_act) + h  # linear + skip
            mean = jnp.mean(h_pre_bn, axis=0)
            var = jnp.var(h_pre_bn, axis=0)
            stats.append((mean, var))
            h = layer.bn.weight * (h_pre_bn - mean) / jnp.sqrt(var + layer.bn.eps) + layer.bn.bias
        elif isinstance(layer, OutputLayer):
            # OutputLayer: act -> BN -> linear (no skip, BN before linear)
            h_act = vmap(layer.act_fn)(h)
            mean = jnp.mean(h_act, axis=0)
            var = jnp.var(h_act, axis=0)
            stats.append((mean, var))
            h_normed = layer.bn.weight * (h_act - mean) / jnp.sqrt(var + layer.bn.eps) + layer.bn.bias
            h = vmap(layer.linear)(h_normed)
        else:
            h = vmap(layer)(h)
            stats.append(None)
    return stats, h


def freeze_batch_stats(model, batch_stats):
    """Return a new model with frozen_mean/frozen_var set from batch_stats
    and use_frozen=True on all BN layers."""
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN, OutputLayer)):
            mean, var = bs
            new_bn = eqx.tree_at(
                lambda bn: (bn.frozen_mean, bn.frozen_var),
                layer.bn,
                (mean, var)
            )
            new_layer = eqx.tree_at(lambda l: l.bn, layer, new_bn)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


def update_ema_stats(model, batch_stats):
    """Update EMA running_mean/running_var from batch_stats (once per batch)."""
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN, OutputLayer)):
            mean, var = bs
            mom = layer.bn.momentum
            new_rm = (1 - mom) * layer.bn.running_mean + mom * mean
            new_rv = (1 - mom) * layer.bn.running_var + mom * var
            new_bn = eqx.tree_at(
                lambda bn: (bn.running_mean, bn.running_var),
                layer.bn,
                (new_rm, new_rv)
            )
            new_layer = eqx.tree_at(lambda l: l.bn, layer, new_bn)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


def set_eval_mode(model):
    """Set all BN layers to use running stats (for test time).

    Returns a new model with use_frozen=False on all BN layers.
    Since use_frozen is a static field we need to rebuild the layers.
    """
    new_layers = []
    for layer in model.layers:
        if isinstance(layer, (HiddenLayerBN, OutputLayer)):
            new_bn = BatchNormLayer.__new__(BatchNormLayer)
            object.__setattr__(new_bn, 'weight', layer.bn.weight)
            object.__setattr__(new_bn, 'bias', layer.bn.bias)
            object.__setattr__(new_bn, 'running_mean', layer.bn.running_mean)
            object.__setattr__(new_bn, 'running_var', layer.bn.running_var)
            object.__setattr__(new_bn, 'frozen_mean', layer.bn.frozen_mean)
            object.__setattr__(new_bn, 'frozen_var', layer.bn.frozen_var)
            object.__setattr__(new_bn, 'use_frozen', False)
            object.__setattr__(new_bn, 'momentum', layer.bn.momentum)
            object.__setattr__(new_bn, 'eps', layer.bn.eps)
            new_layer = eqx.tree_at(lambda l: l.bn, layer, new_bn)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


# ============================================================================
# Forward-pass activity init (custom, since jpc.init_activities_with_ffwd
# doesn't know about our BN)
# ============================================================================
def init_activities_bf(model, x_batch):
    """Initialise activities via a forward pass with proper batch-level BN.

    Also returns per-layer batch stats so they can be frozen.
    """
    batch_stats, _ = _compute_batch_stats(model, x_batch)

    # Now run the frozen model to get per-layer activities
    # jpc expects L activities (one per layer), including the output prediction
    frozen_model = freeze_batch_stats(model, batch_stats)
    activities = []
    h = x_batch
    for layer in frozen_model.layers:
        h = vmap(layer)(h)
        activities.append(h)
    return activities, batch_stats, frozen_model


# ============================================================================
# Evaluation (test time uses running stats)
# ============================================================================
def evaluate_bf(model, test_loader):
    """Evaluate test accuracy using running stats (eval mode)."""
    eval_model = set_eval_mode(model)
    avg_acc = 0.0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        # Forward pass with running stats
        preds = vmap(eval_model)(img_batch)
        acc = float(jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)) * 100)
        avg_acc += acc
    return avg_acc / len(test_loader)


# ============================================================================
# BF Training loop
# ============================================================================
def train_bf_and_record(
    seed, model, depth,
    activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
    batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
    test_every=TEST_EVERY, act_fn="tanh",
    track_weight_updates=False,
    track_activity_norms=False,
):
    """Train a BF-PCN and record metrics.

    The model is an FCResNetBN instance. We use param_type="sp" and
    skip_model=None since ResNet skips are internal to the model.
    """
    set_seed(seed)

    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)

    # jpc.update_pc_params expects params=(layers, skip_model) so the
    # optimizer state must match this tuple structure.
    # Running/frozen BN stats are also array leaves — the optimizer
    # will track them but we overwrite them manually each iteration,
    # so the extra state is harmless.
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

        # ---- Step 1: Forward pass to compute batch stats + init activities ----
        activities, batch_stats, frozen_model = init_activities_bf(model, img_batch)

        train_loss = float(jpc.mse_loss(activities[-1], label_batch))
        train_losses.append(train_loss)

        # Record activity norms at init
        if track_activity_norms:
            norms = [float(jnp.linalg.norm(a, axis=1, ord=2).mean()) for a in activities]
            activity_norms_init.append(norms)

        # Snapshot weights for update tracking
        if track_weight_updates:
            old_weights = [jnp.array(w) for w in _get_weights(model)]

        # ---- Step 2: PC Inference with frozen BN stats ----
        activity_opt_state = activity_optim.init(activities)
        for t in range(depth):
            result = jpc.update_pc_activities(
                params=(frozen_model.layers, None),
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

        # ---- Step 3: Learning — update W and BN affine params ----
        result = jpc.update_pc_params(
            params=(frozen_model.layers, None),
            activities=activities,
            optim=param_optim,
            opt_state=param_opt_state,
            output=label_batch,
            input=img_batch,
            param_type="sp"
        )
        updated_layers = result["model"]
        param_opt_state = result["opt_state"]

        # Reconstruct the full model with updated layers but original BN stats
        model = eqx.tree_at(lambda m: m.layers, model, updated_layers)

        # ---- Step 4: Update EMA running stats once ----
        model = update_ema_stats(model, batch_stats)

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
            avg_acc = evaluate_bf(model, test_loader)
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
# Condition number for BF model
# ============================================================================
def compute_condition_number_bf(model, x, y):
    """Compute kappa(H_z) for a BF model at init."""
    activities, batch_stats, frozen_model = init_activities_bf(model, x)

    hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
        (frozen_model.layers, None), activities, y,
        x=x, param_type="sp"
    )
    H = unwrap_hessian_pytree(hessian_pytree, activities)
    eigenvals = jnp.linalg.eigvalsh(H)
    lam_max = jnp.abs(eigenvals[-1])
    lam_min = jnp.abs(eigenvals[0])
    cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
    return cond, eigenvals
