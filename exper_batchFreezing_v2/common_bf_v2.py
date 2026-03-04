"""BatchNorm Freezing v2 (BF-v2) for Predictive Coding Networks.

Same as BF but with different BatchNorm placement in hidden layers:
  - BF  (original): x -> act_fn -> BN(linear(x) + skip)
  - BF-v2 (this):   x -> act_fn -> BN(linear(x)) + skip

BN wraps only the linear output; the skip connection is added AFTER
normalisation.  Everything else (freezing algorithm, EMA, output layer,
training loop) is identical to the original BF experiment.
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

BF_V2_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ============================================================================
# BatchNorm Layer (identical to original BF)
# ============================================================================
class BatchNormLayer(eqx.Module):
    """BatchNorm that uses pre-computed (frozen) batch statistics."""
    weight: jnp.ndarray       # gamma (learnable scale)
    bias: jnp.ndarray         # beta  (learnable shift)
    running_mean: jnp.ndarray
    running_var: jnp.ndarray
    frozen_mean: jnp.ndarray
    frozen_var: jnp.ndarray
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
        if self.use_frozen:
            mean, var = self.frozen_mean, self.frozen_var
        else:
            mean, var = self.running_mean, self.running_var
        return self.weight * (x - mean) / jnp.sqrt(var + self.eps) + self.bias


# ============================================================================
# Layer modules — v2 placement: BN wraps linear only, skip added after
# ============================================================================
class InputLayer(eqx.Module):
    """First layer: Linear (no activation, no BN, no skip)."""
    linear: nn.Linear

    def __call__(self, x):
        return self.linear(x)


class HiddenLayerBN(eqx.Module):
    """Hidden layer v2: act_fn -> BN(linear(x)) + skip.

    BN normalises the linear output BEFORE the skip connection is added.
    """
    act_fn: Callable = eqx.static_field()
    bn: BatchNormLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.bn(self.linear(h)) + x   # BN on linear only, then skip
        return h


class OutputLayer(eqx.Module):
    """Output layer: act_fn -> BN -> Linear (no skip, same as original)."""
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
class FCResNetBN_v2(eqx.Module):
    """FC-ResNet with BN wrapping linear only (skip added after BN)."""
    layers: List[eqx.Module]

    def __init__(self, *, key, in_dim, width, depth, out_dim,
                 act_fn="tanh", use_frozen=True):
        act = jpc.get_act_fn(act_fn)
        keys = jr.split(key, depth)
        self.layers = []

        # Layer 0: input -> hidden
        self.layers.append(InputLayer(
            linear=nn.Linear(in_dim, width, use_bias=False, key=keys[0])
        ))

        # Layers 1 .. depth-2: hidden -> hidden
        for i in range(1, depth - 1):
            self.layers.append(HiddenLayerBN(
                act_fn=act,
                bn=BatchNormLayer(width, use_frozen=use_frozen),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

        # Last layer: hidden -> output
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
# Helper: compute batch statistics — v2 placement
# ============================================================================
def _compute_batch_stats(model, x_batch):
    """Forward pass collecting per-layer BN statistics.

    v2 difference: for HiddenLayerBN, stats are computed on linear output
    only (before adding the skip).
    """
    stats = []
    h = x_batch  # (batch, features)
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            h = vmap(layer)(h)
            stats.append(None)
        elif isinstance(layer, HiddenLayerBN):
            # v2: act -> linear -> BN -> + skip
            h_act = vmap(layer.act_fn)(h)
            h_linear = vmap(layer.linear)(h_act)       # linear only
            mean = jnp.mean(h_linear, axis=0)
            var = jnp.var(h_linear, axis=0)
            stats.append((mean, var))
            h_normed = layer.bn.weight * (h_linear - mean) / jnp.sqrt(var + layer.bn.eps) + layer.bn.bias
            h = h_normed + h                            # skip added after BN
        elif isinstance(layer, OutputLayer):
            # Output layer unchanged: act -> BN -> linear
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
    """Return a new model with frozen_mean/frozen_var set from batch_stats."""
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
    """Update EMA running_mean/running_var from batch_stats."""
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
    """Set all BN layers to use running stats (for test time)."""
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
# Forward-pass activity init
# ============================================================================
def init_activities_bf_v2(model, x_batch):
    """Initialise activities via a forward pass with proper batch-level BN."""
    batch_stats, _ = _compute_batch_stats(model, x_batch)
    frozen_model = freeze_batch_stats(model, batch_stats)
    activities = []
    h = x_batch
    for layer in frozen_model.layers:
        h = vmap(layer)(h)
        activities.append(h)
    return activities, batch_stats, frozen_model


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_bf_v2(model, test_loader):
    """Evaluate test accuracy using running stats (eval mode)."""
    eval_model = set_eval_mode(model)
    avg_acc = 0.0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        preds = vmap(eval_model)(img_batch)
        acc = float(jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)) * 100)
        avg_acc += acc
    return avg_acc / len(test_loader)


# ============================================================================
# BF-v2 Training loop
# ============================================================================
def train_bf_v2_and_record(
    seed, model, depth,
    activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
    batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
    test_every=TEST_EVERY, act_fn="tanh",
    track_weight_updates=False,
    track_activity_norms=False,
):
    """Train a BF-v2 PCN and record metrics."""
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
        activities, batch_stats, frozen_model = init_activities_bf_v2(model, img_batch)

        train_loss = float(jpc.mse_loss(activities[-1], label_batch))
        train_losses.append(train_loss)

        if track_activity_norms:
            norms = [float(jnp.linalg.norm(a, axis=1, ord=2).mean()) for a in activities]
            activity_norms_init.append(norms)

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

        model = eqx.tree_at(lambda m: m.layers, model, updated_layers)

        # ---- Step 4: Update EMA running stats once ----
        model = update_ema_stats(model, batch_stats)

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
            avg_acc = evaluate_bf_v2(model, test_loader)
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
# Condition number
# ============================================================================
def compute_condition_number_bf_v2(model, x, y):
    """Compute kappa(H_z) for a BF-v2 model at init."""
    activities, batch_stats, frozen_model = init_activities_bf_v2(model, x)

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
