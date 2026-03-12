"""Dynamic Tanh (DyT) variant — DyT wraps (linear + skip)."""

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
# DyT Layer
# ============================================================================
class DyTLayer(eqx.Module):
    """DyT(x) = gamma * tanh(alpha * x) + beta"""
    alpha: jnp.ndarray
    gamma: jnp.ndarray
    beta: jnp.ndarray
    use_dyt: bool = eqx.field(static=True)

    def __init__(self, num_features, *, init_alpha=0.5, use_dyt=True):
        self.alpha = jnp.ones(1) * init_alpha
        self.gamma = jnp.ones(num_features)
        self.beta = jnp.zeros(num_features)
        self.use_dyt = use_dyt

    def __call__(self, x):
        if not self.use_dyt:
            return x
        return self.gamma * jnp.tanh(self.alpha * x) + self.beta


# ============================================================================
# Layer modules
# ============================================================================
class InputLayer(eqx.Module):
    linear: nn.Linear

    def __call__(self, x):
        return self.linear(x)


class HiddenLayerDyT(eqx.Module):
    """Hidden layer: act_fn -> Linear + skip -> DyT."""
    act_fn: Callable = eqx.field(static=True)
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.linear(h) + x
        return self.dyt(h)


class OutputLayer(eqx.Module):
    """Output layer: act_fn -> DyT -> Linear."""
    act_fn: Callable = eqx.field(static=True)
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
    layers: List[eqx.Module]

    def __init__(self, *, key, in_dim, width, depth, out_dim,
                 act_fn="tanh", init_alpha=0.5, dyt_enabled_layers=None):
        act = jpc.get_act_fn(act_fn)
        keys = jr.split(key, depth)
        self.layers = []

        if dyt_enabled_layers is None or dyt_enabled_layers == "all":
            dyt_mask = {i: True for i in range(depth)}
        elif isinstance(dyt_enabled_layers, str) and dyt_enabled_layers.startswith("every_n:"):
            n = int(dyt_enabled_layers.split(":")[1])
            dyt_mask = {i: (i % n == 0) for i in range(depth)}
        else:
            dyt_mask = {i: (i in dyt_enabled_layers) for i in range(depth)}

        self.layers.append(InputLayer(
            linear=nn.Linear(in_dim, width, use_bias=False, key=keys[0])
        ))

        for i in range(1, depth - 1):
            self.layers.append(HiddenLayerDyT(
                act_fn=act,
                dyt=DyTLayer(width, init_alpha=init_alpha,
                             use_dyt=dyt_mask.get(i, True)),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

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
# Variant implementation
# ============================================================================
class DyTVariant:
    """DyT v1: DyT wraps (linear + skip)."""

    @property
    def name(self):
        return "DyT ResNet"

    @property
    def has_batch_stats(self):
        return False

    def create_model(self, key, depth, width, act_fn, **kwargs):
        init_alpha = kwargs.get("init_alpha", 0.5)
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        return FCResNetDyT(
            key=key, in_dim=input_dim, width=width, depth=depth,
            out_dim=output_dim, act_fn=act_fn, init_alpha=init_alpha,
        )

    def init_activities(self, model, x_batch):
        activities = []
        h = x_batch
        for layer in model.layers:
            h = vmap(layer)(h)
            activities.append(h)
        return activities, None, model

    def get_params_for_jpc(self, model):
        return (model.layers, None)

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, model):
        return (eqx.filter(model.layers, eqx.is_array), None)

    def post_learning_step(self, model, result, batch_stats):
        updated_layers = result["model"]
        return eqx.tree_at(lambda m: m.layers, model, updated_layers)

    def evaluate(self, model, test_loader):
        avg_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            preds = vmap(model)(img_batch)
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
        activities, _, _ = self.init_activities(model, x)
        hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
            (model.layers, None), activities, y,
            x=x, param_type="sp",
        )
        H = unwrap_hessian_pytree(hessian_pytree, activities)
        eigenvals = jnp.linalg.eigvalsh(H)
        lam_max = jnp.abs(eigenvals[-1])
        lam_min = jnp.abs(eigenvals[0])
        cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
        return cond, eigenvals
