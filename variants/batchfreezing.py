"""BatchNorm Freezing (BF) variant — BN wraps (linear + skip)."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import jpc
import equinox as eqx
import equinox.nn as nn

from typing import List, Callable

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list
from common.hessian import unwrap_hessian_pytree


# ============================================================================
# BatchNorm Layer
# ============================================================================
class BatchNormLayer(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    running_mean: jnp.ndarray
    running_var: jnp.ndarray
    frozen_mean: jnp.ndarray
    frozen_var: jnp.ndarray
    use_frozen: bool = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

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
# Layer modules
# ============================================================================
class InputLayer(eqx.Module):
    linear: nn.Linear

    def __call__(self, x):
        return self.linear(x)


class HiddenLayerBN(eqx.Module):
    """Hidden layer: act_fn -> Linear + skip -> BN (BN after skip)."""
    act_fn: Callable = eqx.field(static=True)
    bn: BatchNormLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.linear(h) + x
        return self.bn(h)


class OutputLayer(eqx.Module):
    """Output layer: act_fn -> BN -> Linear (no skip)."""
    act_fn: Callable = eqx.field(static=True)
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
    layers: List[eqx.Module]

    def __init__(self, *, key, in_dim, width, depth, out_dim,
                 act_fn="tanh", use_frozen=True):
        act = jpc.get_act_fn(act_fn)
        keys = jr.split(key, depth)
        self.layers = []

        self.layers.append(InputLayer(
            linear=nn.Linear(in_dim, width, use_bias=False, key=keys[0])
        ))

        for i in range(1, depth - 1):
            self.layers.append(HiddenLayerBN(
                act_fn=act,
                bn=BatchNormLayer(width, use_frozen=use_frozen),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

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
# BN helpers
# ============================================================================
def _compute_batch_stats(model, x_batch):
    stats = []
    h = x_batch
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            h = vmap(layer)(h)
            stats.append(None)
        elif isinstance(layer, HiddenLayerBN):
            h_act = vmap(layer.act_fn)(h)
            h_pre_bn = vmap(layer.linear)(h_act) + h
            mean = jnp.mean(h_pre_bn, axis=0)
            var = jnp.var(h_pre_bn, axis=0)
            stats.append((mean, var))
            h = layer.bn.weight * (h_pre_bn - mean) / jnp.sqrt(var + layer.bn.eps) + layer.bn.bias
        elif isinstance(layer, OutputLayer):
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
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN, OutputLayer)):
            mean, var = bs
            new_bn = eqx.tree_at(
                lambda bn: (bn.frozen_mean, bn.frozen_var),
                layer.bn, (mean, var),
            )
            new_layer = eqx.tree_at(lambda l: l.bn, layer, new_bn)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


def update_ema_stats(model, batch_stats):
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN, OutputLayer)):
            mean, var = bs
            mom = layer.bn.momentum
            new_rm = (1 - mom) * layer.bn.running_mean + mom * mean
            new_rv = (1 - mom) * layer.bn.running_var + mom * var
            new_bn = eqx.tree_at(
                lambda bn: (bn.running_mean, bn.running_var),
                layer.bn, (new_rm, new_rv),
            )
            new_layer = eqx.tree_at(lambda l: l.bn, layer, new_bn)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


def set_eval_mode(model):
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
# Variant implementation
# ============================================================================
class BatchFreezingVariant:
    """BF v1: BN wraps (linear + skip)."""

    @property
    def name(self):
        return "BF ResNet"

    @property
    def has_batch_stats(self):
        return True

    def create_model(self, key, depth, width, act_fn, **kwargs):
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        return FCResNetBN(
            key=key, in_dim=input_dim, width=width, depth=depth,
            out_dim=output_dim, act_fn=act_fn,
        )

    def init_activities(self, model, x_batch):
        batch_stats, _ = _compute_batch_stats(model, x_batch)
        frozen_model = freeze_batch_stats(model, batch_stats)
        activities = []
        h = x_batch
        for layer in frozen_model.layers:
            h = vmap(layer)(h)
            activities.append(h)
        return activities, batch_stats, frozen_model

    def get_params_for_jpc(self, model):
        if isinstance(model, FCResNetBN):
            return (model.layers, None)
        # If frozen model passed via effective_model
        return (model.layers, None)

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, model):
        return (eqx.filter(model.layers, eqx.is_array), None)

    def post_learning_step(self, model, result, batch_stats):
        updated_layers = result["model"]
        model = eqx.tree_at(lambda m: m.layers, model, updated_layers)
        model = update_ema_stats(model, batch_stats)
        return model

    def evaluate(self, model, test_loader):
        eval_model = set_eval_mode(model)
        avg_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            preds = vmap(eval_model)(img_batch)
            acc = float(
                jnp.mean(
                    jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
                ) * 100
            )
            avg_acc += acc
        return avg_acc / len(test_loader)

    def get_weight_arrays(self, model):
        return get_weight_list(model)

    def compute_condition_number(self, model, x, y):
        activities, batch_stats, frozen_model = self.init_activities(model, x)
        hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
            (frozen_model.layers, None), activities, y,
            x=x, param_type="sp",
        )
        H = unwrap_hessian_pytree(hessian_pytree, activities)
        eigenvals = jnp.linalg.eigvalsh(H)
        lam_max = jnp.abs(eigenvals[-1])
        lam_min = jnp.abs(eigenvals[0])
        cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
        return cond, eigenvals
