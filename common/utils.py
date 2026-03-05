"""General utilities for PC experiments."""

import os
import math

import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_leaves
import equinox as eqx


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def selected_layer_indices(depth):
    """Return indices for layers: 1, L/4, L/2, 3L/4, L."""
    L = depth
    idxs = sorted(set([
        0,
        max(0, int(L / 4) - 1),
        max(0, int(L / 2) - 1),
        max(0, int(L * 3 / 4) - 1),
        L - 1,
    ]))
    return idxs


def get_weight_list(model):
    """Return a flat list of 2D+ weight arrays from the model."""
    all_params = tree_leaves(model)
    return [p for p in all_params if isinstance(p, jnp.ndarray) and p.ndim >= 2]


def orthogonal_init(key, weight, gain=1.0):
    out_f, in_f = weight.shape
    shape = (max(out_f, in_f), min(out_f, in_f))
    M = jr.normal(key, shape=shape)
    Q, R = jnp.linalg.qr(M)
    Q *= jnp.sign(jnp.diag(R))
    if out_f < in_f:
        Q = Q.T
    return gain * Q[:out_f, :in_f]


def init_weights_orthogonal(key, model, act_fn="tanh"):
    """Re-initialise all weights with orthogonal init."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight for x in tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    weights = get_weights(model)
    gain = 1.05 if act_fn == "tanh" else 1.0
    subkeys = jr.split(key, len(weights))
    new_weights = [orthogonal_init(sk, w, gain) for sk, w in zip(subkeys, weights)]
    return eqx.tree_at(get_weights, model, new_weights)
