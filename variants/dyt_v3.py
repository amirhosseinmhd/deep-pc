"""Dynamic Tanh v3 (DyT-v3) — DyT wraps linear only, skip handled by jpc."""

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

# Reuse DyTLayer and InputLayer from dyt
from variants.dyt import DyTLayer, InputLayer


# ============================================================================
# v3 Output layer — act_fn -> Linear (no DyT)
# ============================================================================
class OutputLayerV3(eqx.Module):
    """Output layer v3: act_fn -> Linear, no DyT."""
    act_fn: Callable = eqx.field(static=True)
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        return self.linear(h)


# ============================================================================
# v3 Hidden layer — DyT wraps linear only, NO skip (skip handled by jpc)
# ============================================================================
class HiddenLayerDyT_v3(eqx.Module):
    """Hidden layer v3: act_fn -> DyT(linear(x)), no skip."""
    act_fn: Callable = eqx.field(static=True)
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.dyt(self.linear(h))
        return h


# ============================================================================
# Full model
# ============================================================================
class FCResNetDyT_v3(eqx.Module):
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
            self.layers.append(HiddenLayerDyT_v3(
                act_fn=act,
                dyt=DyTLayer(width, init_alpha=init_alpha,
                             use_dyt=dyt_mask.get(i, True)),
                linear=nn.Linear(width, width, use_bias=False, key=keys[i]),
            ))

        self.layers.append(OutputLayerV3(
            act_fn=act,
            linear=nn.Linear(width, out_dim, use_bias=False, key=keys[-1]),
        ))

    def __call__(self, x):
        """Forward pass (for evaluation). Manually applies skip connections."""
        for i, f in enumerate(self.layers):
            if 1 <= i <= len(self.layers) - 2:
                x = f(x) + x
            else:
                x = f(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]


# ============================================================================
# Variant implementation
# ============================================================================
class DyTV3Variant:
    """DyT v3: DyT wraps linear only, skip handled by jpc."""

    @property
    def name(self):
        return "DyT ResNet v3"

    @property
    def has_batch_stats(self):
        return False

    def create_model(self, key, depth, width, act_fn, **kwargs):
        init_alpha = kwargs.get("init_alpha", 0.5)
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        model = FCResNetDyT_v3(
            key=key, in_dim=input_dim, width=width, depth=depth,
            out_dim=output_dim, act_fn=act_fn, init_alpha=init_alpha,
        )
        skip_model = jpc.make_skip_model(depth)
        return {"model": model, "skip_model": skip_model}

    def init_activities(self, bundle, x_batch):
        activities = jpc.init_activities_with_ffwd(
            model=bundle["model"].layers, input=x_batch,
            skip_model=bundle["skip_model"], param_type="sp",
        )
        return activities, None, bundle

    def get_params_for_jpc(self, bundle):
        return (bundle["model"].layers, bundle["skip_model"])

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, bundle):
        return (eqx.filter(bundle["model"].layers, eqx.is_array),
                bundle["skip_model"])

    def post_learning_step(self, bundle, result, batch_stats):
        updated_layers = result["model"]
        updated_model = eqx.tree_at(
            lambda m: m.layers, bundle["model"], updated_layers
        )
        return {
            "model": updated_model,
            "skip_model": result["skip_model"],
        }

    def evaluate(self, bundle, test_loader):
        avg_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            _, test_acc = jpc.test_discriminative_pc(
                model=bundle["model"].layers, output=label_batch,
                input=img_batch, skip_model=bundle["skip_model"],
                param_type="sp",
            )
            avg_acc += float(test_acc)
        return avg_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        return get_weight_list(bundle["model"])

    def compute_condition_number(self, bundle, x, y):
        activities = jpc.init_activities_with_ffwd(
            model=bundle["model"].layers, input=x,
            skip_model=bundle["skip_model"], param_type="sp",
        )
        hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
            (bundle["model"].layers, bundle["skip_model"]), activities, y,
            x=x, param_type="sp",
        )
        H = unwrap_hessian_pytree(hessian_pytree, activities)
        eigenvals = jnp.linalg.eigvalsh(H)
        lam_max = jnp.abs(eigenvals[-1])
        lam_min = jnp.abs(eigenvals[0])
        cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
        return cond, eigenvals
