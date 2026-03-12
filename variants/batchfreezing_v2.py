"""BatchNorm Freezing v2 — BN wraps linear only, skip added after."""

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


# Reuse BatchNormLayer from batchfreezing (identical)
from variants.batchfreezing import BatchNormLayer, InputLayer, OutputLayer


# ============================================================================
# v2 Hidden layer — BN wraps linear only, skip after
# ============================================================================
class HiddenLayerBN_v2(eqx.Module):
    """Hidden layer v2: act_fn -> BN(linear(x)) + skip."""
    act_fn: Callable = eqx.field(static=True)
    bn: BatchNormLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.bn(self.linear(h)) + x
        return h


# ============================================================================
# Full model
# ============================================================================
class FCResNetBN_v2(eqx.Module):
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
            self.layers.append(HiddenLayerBN_v2(
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
# BN helpers (v2 — different _compute_batch_stats)
# ============================================================================
def _compute_batch_stats_v2(model, x_batch):
    stats = []
    h = x_batch
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            h = vmap(layer)(h)
            stats.append(None)
        elif isinstance(layer, HiddenLayerBN_v2):
            h_act = vmap(layer.act_fn)(h)
            h_linear = vmap(layer.linear)(h_act)
            mean = jnp.mean(h_linear, axis=0)
            var = jnp.var(h_linear, axis=0)
            stats.append((mean, var))
            h_normed = layer.bn.weight * (h_linear - mean) / jnp.sqrt(var + layer.bn.eps) + layer.bn.bias
            h = h_normed + h
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


def _freeze_batch_stats_v2(model, batch_stats):
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN_v2, OutputLayer)):
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


def _update_ema_stats_v2(model, batch_stats):
    new_layers = []
    for layer, bs in zip(model.layers, batch_stats):
        if bs is not None and isinstance(layer, (HiddenLayerBN_v2, OutputLayer)):
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


def _set_eval_mode_v2(model):
    new_layers = []
    for layer in model.layers:
        if isinstance(layer, (HiddenLayerBN_v2, OutputLayer)):
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
class BatchFreezingV2Variant:
    """BF v2: BN wraps linear only, skip added after."""

    @property
    def name(self):
        return "BF ResNet v2"

    @property
    def has_batch_stats(self):
        return True

    def create_model(self, key, depth, width, act_fn, **kwargs):
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        return FCResNetBN_v2(
            key=key, in_dim=input_dim, width=width, depth=depth,
            out_dim=output_dim, act_fn=act_fn,
        )

    def init_activities(self, model, x_batch):
        batch_stats, _ = _compute_batch_stats_v2(model, x_batch)
        frozen_model = _freeze_batch_stats_v2(model, batch_stats)
        activities = []
        h = x_batch
        for layer in frozen_model.layers:
            h = vmap(layer)(h)
            activities.append(h)
        return activities, batch_stats, frozen_model

    def get_params_for_jpc(self, model):
        return (model.layers, None)

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, model):
        return (eqx.filter(model.layers, eqx.is_array), None)

    def post_learning_step(self, model, result, batch_stats):
        updated_layers = result["model"]
        model = eqx.tree_at(lambda m: m.layers, model, updated_layers)
        model = _update_ema_stats_v2(model, batch_stats)
        return model

    def evaluate(self, model, test_loader):
        eval_model = _set_eval_mode_v2(model)
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
