"""res-error-net-resnet18 — ResNet-18 backbone with residual error highways.

Each of the 10 PC activities (stem output + 8 basic-block outputs + classifier
output) gets a learnable full-rank V_{L→i} matrix that carries the classifier
error e^L directly into that activity. Forward path is a genuine ResNet-18;
backward path combines standard PC errors with the highway shortcuts, so the
output-error signal reaches the early conv layers without vanishing.

Augmented free energy:
    F = Σ_ℓ (1/2)·mean‖e^ℓ‖²
      + α · Σ_{i∈S} mean( flatten(z^i) · ( sg(e^L) @ V_i^T ) )

The highway factor is the *state* z^i (not the prediction error e^i), and
e^L is wrapped in stop_gradient so F_hw has no influence on z^{L-1} during
inference — the last layer only aligns to the clamped target via F_pc.

Normalization: Dynamic Tanh (DyT) — DyT(x) = γ ⊙ tanh(α·x) + β — used instead
of BatchNorm because DyT is stateless and per-example, so it stays consistent
across the T-step iterative inference loop. Selectable via
`normalization ∈ {"dyt", "none"}`.
"""

from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import numpy as np

import equinox as eqx
import equinox.nn as nn
import jpc

from config import OUTPUT_DIM as _DEFAULT_OUTPUT_DIM


# ----------------------------------------------------------------------
# Dynamic Tanh for conv features
# ----------------------------------------------------------------------

class DyTConv2d(eqx.Module):
    """DyT(x) = γ ⊙ tanh(α·x) + β, broadcast over spatial dims.

    α : scalar,  γ, β : (C,).  Input x : (C, H, W).
    """
    alpha: jnp.ndarray
    gamma: jnp.ndarray
    beta:  jnp.ndarray

    def __init__(self, num_channels: int, init_alpha: float = 0.5):
        self.alpha = jnp.ones(()) * init_alpha
        self.gamma = jnp.ones(num_channels)
        self.beta  = jnp.zeros(num_channels)

    def __call__(self, x):
        return (
            self.gamma[:, None, None] * jnp.tanh(self.alpha * x)
            + self.beta[:, None, None]
        )


def _make_norm(num_channels: int, normalization: str, init_alpha: float):
    if normalization == "dyt":
        return DyTConv2d(num_channels, init_alpha)
    if normalization == "none":
        return None
    raise ValueError(f"Unknown normalization: {normalization!r}")


# ----------------------------------------------------------------------
# Layer modules
# ----------------------------------------------------------------------

class Stem(eqx.Module):
    conv: nn.Conv2d
    norm: Optional[DyTConv2d]
    act_fn: Callable = eqx.field(static=True)

    def __call__(self, x):               # x: (C_in, H, W)
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)
        return self.act_fn(h)


class BasicBlock(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    skip_proj: Optional[nn.Conv2d]
    norm1: Optional[DyTConv2d]
    norm2: Optional[DyTConv2d]
    act_fn: Callable = eqx.field(static=True)

    def __call__(self, z_in):            # z_in: (C_in, H, W)
        h1 = self.conv1(z_in)
        if self.norm1 is not None:
            h1 = self.norm1(h1)
        z_mid = self.act_fn(h1)
        h2 = self.conv2(z_mid)
        if self.norm2 is not None:
            h2 = self.norm2(h2)
        skip = self.skip_proj(z_in) if self.skip_proj is not None else z_in
        return self.act_fn(h2 + skip)


class Head(eqx.Module):
    linear: nn.Linear

    def __call__(self, z_in):            # z_in: (C, H, W)
        pooled = jnp.mean(z_in, axis=(1, 2))   # (C,)
        return self.linear(pooled)


# ----------------------------------------------------------------------
# Module-level JIT helpers.  `variant` is static (empty hashable class) so
# filter_jit caches per call site; `bundle` is a dict whose array leaves are
# traced and static fields (ints, strings, Callable) stay at compile time.
# ----------------------------------------------------------------------

@eqx.filter_jit
def _jit_inference_scan(variant, bundle, z_init, x_batch, y_batch, alpha, dt, T):
    def body(z, _):
        new_z = variant.inference_step(bundle, z, x_batch, y_batch, alpha, dt)
        return new_z, None
    z_out, _ = jax.lax.scan(body, z_init, None, length=T)
    return z_out


@eqx.filter_jit
def _jit_inference_scan_adam(variant, bundle, z_init, x_batch, y_batch,
                             alpha, lr, T, b1, b2, eps):
    """T-step Adam-on-z scan. Per-coordinate adaptive step size on the
    inference dynamics — much less sensitive to `lr` than plain Euler.
    Shape-agnostic, so conv-shape z's (B, C, H, W) work identically."""
    L = bundle["depth"]
    z_free = [z_init[l] for l in range(L - 1)]
    m0 = [jnp.zeros_like(zl) for zl in z_free]
    v0 = [jnp.zeros_like(zl) for zl in z_free]
    t0 = jnp.zeros((), dtype=jnp.int32)

    def body(carry, _):
        zf, m, v, t = carry
        grads = jax.grad(
            lambda zfree: variant.free_energy_z(
                zfree, bundle, x_batch, y_batch, alpha
            )
        )(zf)
        t_new = t + 1
        m_new = [b1 * m[i] + (1.0 - b1) * grads[i] for i in range(len(zf))]
        v_new = [b2 * v[i] + (1.0 - b2) * grads[i] ** 2 for i in range(len(zf))]
        bc1 = 1.0 - b1 ** t_new
        bc2 = 1.0 - b2 ** t_new
        zf_new = [
            zf[i] - lr * (m_new[i] / bc1) / (jnp.sqrt(v_new[i] / bc2) + eps)
            for i in range(len(zf))
        ]
        return (zf_new, m_new, v_new, t_new), None

    (zf_final, _, _, _), _ = jax.lax.scan(
        body, (z_free, m0, v0, t0), None, length=T
    )
    return zf_final + [y_batch]


@eqx.filter_jit
def _jit_inference_step_adam(variant, bundle, z, m, v, t, x_batch, y_batch,
                             alpha, lr, b1, b2, eps):
    """One Adam-on-z step with externally maintained (m, v, t). Used by
    the eager `record_energy=True` path and by diagnostic loops."""
    L = bundle["depth"]
    zf = [z[l] for l in range(L - 1)]
    grads = jax.grad(
        lambda zfree: variant.free_energy_z(
            zfree, bundle, x_batch, y_batch, alpha
        )
    )(zf)
    t_new = t + 1
    m_new = [b1 * m[i] + (1.0 - b1) * grads[i] for i in range(len(zf))]
    v_new = [b2 * v[i] + (1.0 - b2) * grads[i] ** 2 for i in range(len(zf))]
    bc1 = 1.0 - b1 ** t_new
    bc2 = 1.0 - b2 ** t_new
    zf_new = [
        zf[i] - lr * (m_new[i] / bc1) / (jnp.sqrt(v_new[i] / bc2) + eps)
        for i in range(len(zf))
    ]
    return zf_new + [y_batch], m_new, v_new, t_new


@eqx.filter_jit
def _jit_forward_pass(variant, bundle, x_batch):
    return variant.forward_pass(bundle, x_batch)


@eqx.filter_jit
def _jit_compute_errors(variant, bundle, z, x_batch, y_batch):
    return variant.compute_errors(bundle, z, x_batch, y_batch)


@eqx.filter_jit
def _jit_compute_W_updates(variant, bundle, z, x_batch, y_batch, alpha):
    bundle = {**bundle, "alpha_for_w_grad": alpha}
    return variant.compute_W_updates(bundle, z, x_batch, y_batch)


@eqx.filter_jit
def _jit_compute_V_updates(variant, bundle, z, x_batch, y_batch, alpha, rule, v_reg):
    return variant.compute_V_updates(
        bundle, z, x_batch, y_batch, alpha, rule=rule, v_reg=v_reg
    )


# ----------------------------------------------------------------------
# Variant
# ----------------------------------------------------------------------

class ResErrorNetResNet18Variant:
    """ResNet-18 + V_{L→i} residual error highways at every block output."""

    @property
    def name(self):
        return "res-error-net-resnet18"

    @property
    def has_batch_stats(self):
        return False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def create_model(self, key, depth=None, width=None, act_fn="relu", **kwargs):
        del depth, width  # fixed ResNet-18 layout

        input_shape     = kwargs.get("input_shape", (3, 32, 32))
        output_dim      = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        channels        = kwargs.get("resnet_channels", [64, 64, 128, 256, 512])
        blocks_per_stage = kwargs.get("blocks_per_stage", 2)
        normalization   = kwargs.get("normalization", "dyt")
        dyt_init_alpha  = kwargs.get("dyt_init_alpha", 0.5)
        highway_include_stem = kwargs.get("highway_include_stem", True)
        v_init_scale    = kwargs.get("v_init_scale", 0.01)
        inference_method = kwargs.get("inference_method", "euler")
        inference_b1 = kwargs.get("inference_b1", 0.9)
        inference_b2 = kwargs.get("inference_b2", 0.999)
        inference_eps = kwargs.get("inference_eps", 1e-8)

        act_fn_callable = jpc.get_act_fn(act_fn)
        C_in, H, W = input_shape
        stem_C = channels[0]
        stage_channels = channels[1:]

        layers: List[eqx.Module] = []
        layer_shapes: List[tuple] = []

        # ---- stem ----
        key, sub = jr.split(key)
        stem_conv = nn.Conv2d(
            C_in, stem_C, 3, stride=1, padding=1, use_bias=False, key=sub,
        )
        stem_norm = _make_norm(stem_C, normalization, dyt_init_alpha)
        layers.append(Stem(conv=stem_conv, norm=stem_norm, act_fn=act_fn_callable))
        layer_shapes.append((stem_C, H, W))
        cur_C, cur_H, cur_W = stem_C, H, W

        # ---- residual blocks ----
        for stage_idx, out_C in enumerate(stage_channels):
            for b_idx in range(blocks_per_stage):
                # Stride 2 at first block of stages 2-4 (stage_idx > 0, b_idx == 0)
                stride = 2 if (stage_idx > 0 and b_idx == 0) else 1
                key, sub = jr.split(key)
                conv1 = nn.Conv2d(
                    cur_C, out_C, 3, stride=stride, padding=1,
                    use_bias=False, key=sub,
                )
                key, sub = jr.split(key)
                conv2 = nn.Conv2d(
                    out_C, out_C, 3, stride=1, padding=1,
                    use_bias=False, key=sub,
                )
                if stride != 1 or cur_C != out_C:
                    key, sub = jr.split(key)
                    skip_proj = nn.Conv2d(
                        cur_C, out_C, 1, stride=stride, padding=0,
                        use_bias=False, key=sub,
                    )
                else:
                    skip_proj = None

                norm1 = _make_norm(out_C, normalization, dyt_init_alpha)
                norm2 = _make_norm(out_C, normalization, dyt_init_alpha)

                layers.append(BasicBlock(
                    conv1=conv1, conv2=conv2, skip_proj=skip_proj,
                    norm1=norm1, norm2=norm2, act_fn=act_fn_callable,
                ))

                cur_H = cur_H // stride
                cur_W = cur_W // stride
                cur_C = out_C
                layer_shapes.append((cur_C, cur_H, cur_W))

        # ---- head ----
        key, sub = jr.split(key)
        head_linear = nn.Linear(cur_C, output_dim, use_bias=False, key=sub)
        layers.append(Head(linear=head_linear))
        layer_shapes.append((output_dim,))

        L = len(layers)

        # ---- highway indices ----
        start = 0 if highway_include_stem else 1
        S = list(range(start, L - 1))

        # ---- V matrices (full-rank, flattened-spatial × output_dim) ----
        V_list = []
        for i in S:
            D_i = int(np.prod(layer_shapes[i]))
            key, sub = jr.split(key)
            V_list.append(v_init_scale * jr.normal(sub, (D_i, output_dim)))

        return {
            "model": layers,
            "V_list": V_list,
            "highway_indices": S,
            "layer_shapes": layer_shapes,
            "act_fn": act_fn_callable,
            "act_fn_name": act_fn,
            "depth": L,
            "input_shape": input_shape,
            "output_dim": output_dim,
            "normalization": normalization,
            "dyt_init_alpha": dyt_init_alpha,
            "v_init_scale": v_init_scale,
            "blocks_per_stage": blocks_per_stage,
            "resnet_channels": list(channels),
            "inference_method": inference_method,
            "inference_b1": inference_b1,
            "inference_b2": inference_b2,
            "inference_eps": inference_eps,
        }

    # ------------------------------------------------------------------
    # Forward / predictions / errors
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_input(bundle, x_batch):
        B = x_batch.shape[0]
        return x_batch.reshape(B, *bundle["input_shape"])

    def forward_pass(self, bundle, x_batch):
        model = bundle["model"]
        L = bundle["depth"]

        z_prev = self._reshape_input(bundle, x_batch)
        z = [None] * L
        h = [None] * L
        for l in range(L):
            out = vmap(model[l])(z_prev)
            z[l] = out
            h[l] = out
            z_prev = out
        return z, h

    @staticmethod
    def _predictions(model, z_list, x_spatial, L):
        mu = [None] * L
        for l in range(L):
            prev = x_spatial if l == 0 else z_list[l - 1]
            mu[l] = vmap(model[l])(prev)
        return mu

    def compute_errors(self, bundle, z, x_batch, y_batch):
        model = bundle["model"]
        L = bundle["depth"]
        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z)
        z_list[L - 1] = y_batch
        mu = self._predictions(model, z_list, x_spatial, L)
        return [z_list[l] - mu[l] for l in range(L)]

    # ------------------------------------------------------------------
    # Free energy
    # ------------------------------------------------------------------

    @staticmethod
    def _F_pc(e_list):
        total = jnp.array(0.0)
        for el in e_list:
            flat = el.reshape(el.shape[0], -1)
            total = total + 0.5 * jnp.mean(jnp.sum(flat ** 2, axis=1))
        return total

    @staticmethod
    def _F_highway(z_list, e_list, V_list, S, alpha, L):
        if not S:
            return jnp.array(0.0)
        # stop_grad on e^L: highway no longer pulls z^{L-1} during inference;
        # z^{L-1} is shaped only by F_pc (i.e., aligns to the clamped target).
        e_L = jax.lax.stop_gradient(e_list[L - 1])            # (B, output_dim)
        total = jnp.array(0.0)
        for idx, i in enumerate(S):
            Vi = V_list[idx]                                  # (D_i, output_dim)
            z_i_flat = z_list[i].reshape(z_list[i].shape[0], -1)  # (B, D_i)
            shortcut = e_L @ Vi.T                             # (B, D_i)
            total = total + alpha * jnp.mean(jnp.sum(z_i_flat * shortcut, axis=1))
        return total

    def free_energy_z(self, z_free, bundle, x_batch, y_batch, alpha):
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]

        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z_free) + [y_batch]
        mu = self._predictions(model, z_list, x_spatial, L)
        e = [z_list[l] - mu[l] for l in range(L)]
        return self._F_pc(e) + self._F_highway(z_list, e, V_list, S, alpha, L)

    # ------------------------------------------------------------------
    # Inference dynamics
    # ------------------------------------------------------------------

    def inference_step(self, bundle, z, x_batch, y_batch, alpha, dt):
        L = bundle["depth"]
        z_free = [z[l] for l in range(L - 1)]

        def f(zf):
            return self.free_energy_z(zf, bundle, x_batch, y_batch, alpha)

        grads = jax.grad(f)(z_free)
        new_z_free = [z_free[i] - dt * grads[i] for i in range(L - 1)]
        return new_z_free + [y_batch]

    def run_inference(self, bundle, x_batch, y_batch, alpha, dt, T,
                      z_init=None, record_energy=False):
        L = bundle["depth"]
        method = bundle.get("inference_method", "euler")
        b1 = bundle.get("inference_b1", 0.9)
        b2 = bundle.get("inference_b2", 0.999)
        eps = bundle.get("inference_eps", 1e-8)
        if z_init is None:
            z_init, _ = self.forward_pass(bundle, x_batch)
        z = list(z_init)
        z[L - 1] = y_batch

        if not record_energy:
            if method == "adam":
                z = _jit_inference_scan_adam(
                    self, bundle, z, x_batch, y_batch, alpha, dt, T,
                    b1, b2, eps,
                )
            else:
                z = _jit_inference_scan(
                    self, bundle, z, x_batch, y_batch, alpha, dt, T
                )
            return z, None

        energies = []
        if method == "adam":
            m = [jnp.zeros_like(z[l]) for l in range(L - 1)]
            v = [jnp.zeros_like(z[l]) for l in range(L - 1)]
            t_count = jnp.zeros((), dtype=jnp.int32)
            for _ in range(T):
                z, m, v, t_count = _jit_inference_step_adam(
                    self, bundle, z, m, v, t_count,
                    x_batch, y_batch, alpha, dt, b1, b2, eps,
                )
                z_free = [z[l] for l in range(L - 1)]
                energies.append(float(
                    self.free_energy_z(z_free, bundle, x_batch, y_batch, alpha)
                ))
        else:
            for _ in range(T):
                z = self.inference_step(bundle, z, x_batch, y_batch, alpha, dt)
                z_free = [z[l] for l in range(L - 1)]
                energies.append(float(
                    self.free_energy_z(z_free, bundle, x_batch, y_batch, alpha)
                ))
        return z, energies

    # ------------------------------------------------------------------
    # Parameter updates
    # ------------------------------------------------------------------

    def compute_W_updates(self, bundle, z, x_batch, y_batch):
        """Autodiff ∂F_aug/∂model (all conv/linear weights + DyT α/γ/β).

        Returned as a flat list of array leaves matching the structure of
        `eqx.filter(model, eqx.is_array)` — this lets the existing trainer
        iterate `for dw in delta_W` to take norms and re-project.
        """
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        alpha = bundle.get("alpha_for_w_grad", None)

        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z)
        z_list[L - 1] = y_batch

        def aug_energy(m):
            mu = self._predictions(m, z_list, x_spatial, L)
            e = [z_list[l] - mu[l] for l in range(L)]
            f_pc = self._F_pc(e)
            if alpha is not None and S:
                return f_pc + self._F_highway(z_list, e, V_list, S, alpha, L)
            return f_pc

        grads_tree = eqx.filter_grad(aug_energy)(model)
        return jax.tree_util.tree_leaves(eqx.filter(grads_tree, eqx.is_array))

    def compute_V_updates(self, bundle, z, x_batch, y_batch, alpha,
                          rule="energy", v_reg=0.0):
        """ΔV_{L→i} with flattened block errors."""
        S = bundle["highway_indices"]
        L = bundle["depth"]
        V_list = bundle["V_list"]
        B = x_batch.shape[0]

        e = self.compute_errors(bundle, z, x_batch, y_batch)
        e_L = e[L - 1]

        delta_V = []
        for idx, i in enumerate(S):
            z_i_flat = z[i].reshape(z[i].shape[0], -1)
            if rule == "energy":
                # ∂F_hw_new/∂V_i = α · z^i (e^L)^T / B
                dV = alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            elif rule == "state":
                # Hebbian anti-gradient: grows V along z^i (e^L)^T
                dV = -alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            else:
                raise ValueError(f"Unknown v_update_rule: {rule!r}")
            if v_reg and v_reg > 0.0:
                dV = dV + v_reg * V_list[idx]
            delta_V.append(dV)
        return delta_V

    # ------------------------------------------------------------------
    # Fused update path (single JIT): one forward + one backward yields
    # e, ΔW, ΔV, and per-layer energies together.  Saves two full forward
    # passes per batch vs. the legacy triple-JIT path (one each in
    # compute_errors, compute_W_updates, compute_V_updates).
    # ------------------------------------------------------------------

    def compute_updates_fused(self, bundle, z, x_batch, y_batch, alpha,
                              rule="energy", v_reg=0.0):
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        B = x_batch.shape[0]

        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z)
        z_list[L - 1] = y_batch

        def aug_energy_with_e(m):
            mu = self._predictions(m, z_list, x_spatial, L)
            e = [z_list[l] - mu[l] for l in range(L)]
            f_pc = self._F_pc(e)
            hw = self._F_highway(z_list, e, V_list, S, alpha, L) if S else jnp.array(0.0)
            return f_pc + hw, e

        (_, e), grads_tree = eqx.filter_value_and_grad(
            aug_energy_with_e, has_aux=True
        )(model)
        delta_W = jax.tree_util.tree_leaves(eqx.filter(grads_tree, eqx.is_array))

        energies = jnp.stack([
            0.5 * jnp.mean(jnp.sum(el.reshape(el.shape[0], -1) ** 2, axis=1))
            for el in e
        ])

        e_L = e[L - 1]
        delta_V = []
        for idx, i in enumerate(S):
            z_i_flat = z[i].reshape(z[i].shape[0], -1)
            if rule == "energy":
                # ∂F_hw_new/∂V_i = α · z^i (e^L)^T / B
                dV = alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            elif rule == "state":
                # Hebbian anti-gradient: grows V along z^i (e^L)^T
                dV = -alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            else:
                raise ValueError(f"Unknown v_update_rule: {rule!r}")
            if v_reg and v_reg > 0.0:
                dV = dV + v_reg * V_list[idx]
            delta_V.append(dV)

        return e, delta_W, delta_V, energies

    # ------------------------------------------------------------------
    # Applying updates
    # ------------------------------------------------------------------

    @staticmethod
    def _grad_leaves_to_tree(delta_W_leaves, model):
        """Reconstruct a PyTree matching model's array-leaf structure from a flat list."""
        params = eqx.filter(model, eqx.is_array)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        return jax.tree_util.tree_unflatten(treedef, delta_W_leaves)

    def apply_optax_updates(self, bundle, delta_W, delta_V,
                            w_optim, w_opt_state, v_optim, v_opt_state):
        model = bundle["model"]
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]

        grads_tree = self._grad_leaves_to_tree(delta_W, model)
        params = eqx.filter(model, eqx.is_array)
        updates, w_opt_state = w_optim.update(grads_tree, w_opt_state, params)
        new_model = eqx.apply_updates(model, updates)

        for idx in range(len(S)):
            updates_v, v_opt_state[idx] = v_optim.update(
                delta_V[idx], v_opt_state[idx], V_list[idx]
            )
            V_list[idx] = V_list[idx] + updates_v

        return (
            {**bundle, "model": new_model, "V_list": V_list},
            w_opt_state, v_opt_state,
        )

    def apply_sgd_updates(self, bundle, delta_W, delta_V, lr_W, lr_V):
        model = bundle["model"]
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]

        grads_tree = self._grad_leaves_to_tree(delta_W, model)
        updates_tree = jax.tree_util.tree_map(lambda g: -lr_W * g, grads_tree)
        new_model = eqx.apply_updates(model, updates_tree)

        for idx in range(len(S)):
            V_list[idx] = V_list[idx] - lr_V * delta_V[idx]

        return {**bundle, "model": new_model, "V_list": V_list}

    # ------------------------------------------------------------------
    # Optim-state init
    # ------------------------------------------------------------------

    def init_w_optim_states(self, bundle, w_optim):
        params = eqx.filter(bundle["model"], eqx.is_array)
        return w_optim.init(params)

    def init_v_optim_states(self, bundle, v_optim):
        return [v_optim.init(V) for V in bundle["V_list"]]

    # ------------------------------------------------------------------
    # Evaluation / introspection
    # ------------------------------------------------------------------

    def evaluate(self, bundle, test_loader):
        avg_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch = img_batch.numpy()
            label_batch = label_batch.numpy()
            z, _ = _jit_forward_pass(self, bundle, img_batch)
            preds = z[-1]
            acc = float(jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100)
            avg_acc += acc
        return avg_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        params = eqx.filter(bundle["model"], eqx.is_array)
        return jax.tree_util.tree_leaves(params)

    def get_V_arrays(self, bundle):
        return list(bundle["V_list"])

    def get_weight_labels(self, bundle):
        """Human-readable name for each leaf returned by get_weight_arrays.

        Mirrors the Equinox field order: Stem(conv, norm), BasicBlock(conv1,
        conv2, skip_proj?, norm1, norm2), Head(linear). DyT contributes
        three leaves per norm module in the order (alpha, gamma, beta).
        """
        layers = bundle["model"]
        labels = []
        block_idx = 0
        for layer in layers:
            if isinstance(layer, Stem):
                labels.append("stem.conv")
                if layer.norm is not None:
                    labels += ["stem.norm.alpha", "stem.norm.gamma", "stem.norm.beta"]
            elif isinstance(layer, BasicBlock):
                p = f"block{block_idx}"
                labels.append(f"{p}.conv1")
                labels.append(f"{p}.conv2")
                if layer.skip_proj is not None:
                    labels.append(f"{p}.skip_proj")
                if layer.norm1 is not None:
                    labels += [f"{p}.norm1.alpha", f"{p}.norm1.gamma", f"{p}.norm1.beta"]
                if layer.norm2 is not None:
                    labels += [f"{p}.norm2.alpha", f"{p}.norm2.gamma", f"{p}.norm2.beta"]
                block_idx += 1
            elif isinstance(layer, Head):
                labels.append("head.linear")
            else:
                labels.append(f"unknown_{type(layer).__name__}")
        return labels

    def get_wandb_log_indices(self, bundle):
        """Subset of get_weight_arrays indices worth logging to wandb.

        One conv per architectural block: stem conv, conv1 of each BasicBlock,
        and the head linear. Keeps wandb uncluttered while still covering
        every residual stage the image traverses.
        """
        labels = self.get_weight_labels(bundle)
        return [
            i for i, name in enumerate(labels)
            if name == "stem.conv" or name.endswith(".conv1") or name == "head.linear"
        ]

    # ------------------------------------------------------------------
    # Inference diagnostic
    # ------------------------------------------------------------------

    def diagnose_inference(self, bundle, x_batch, y_batch, alpha, dt, T,
                           loss_type="mse"):
        """Eager T-step inference with per-step (F, loss, acc) trajectory.

        Returns dict with keys "energy", "loss", "acc", each a list of length
        T+1: index 0 is the post-clamp pre-inference state (z = feed-forward,
        z[L-1] := y), indices 1..T are after each inference step.

        loss/acc are computed on μ^L = head(z^{L-2}) — the model's predicted
        output given the current internal state — not on the clamped z[L-1]
        (which would trivially match y). This is the meaningful signal for
        whether inference is moving internal activities toward states that
        produce a correct readout.
        """
        L = bundle["depth"]
        method = bundle.get("inference_method", "euler")
        b1 = bundle.get("inference_b1", 0.9)
        b2 = bundle.get("inference_b2", 0.999)
        eps = bundle.get("inference_eps", 1e-8)

        z_init, _ = self.forward_pass(bundle, x_batch)
        z = list(z_init)
        z[L - 1] = y_batch

        head = bundle["model"][L - 1]

        def snap(z_state):
            z_free = [z_state[l] for l in range(L - 1)]
            F = float(self.free_energy_z(
                z_free, bundle, x_batch, y_batch, alpha
            ))
            mu_L = vmap(head)(z_state[L - 2])
            if loss_type == "ce":
                loss = float(jpc.cross_entropy_loss(mu_L, y_batch))
            else:
                loss = float(jpc.mse_loss(mu_L, y_batch))
            acc = float(jnp.mean(
                jnp.argmax(mu_L, axis=1) == jnp.argmax(y_batch, axis=1)
            ) * 100)
            return F, loss, acc

        energies, losses, accs = [], [], []
        F, ll, ac = snap(z)
        energies.append(F); losses.append(ll); accs.append(ac)

        if method == "adam":
            m = [jnp.zeros_like(z[l]) for l in range(L - 1)]
            v = [jnp.zeros_like(z[l]) for l in range(L - 1)]
            t_count = jnp.zeros((), dtype=jnp.int32)
            for _ in range(T):
                z, m, v, t_count = _jit_inference_step_adam(
                    self, bundle, z, m, v, t_count,
                    x_batch, y_batch, alpha, dt, b1, b2, eps,
                )
                F, ll, ac = snap(z)
                energies.append(F); losses.append(ll); accs.append(ac)
        else:
            for _ in range(T):
                z = self.inference_step(
                    bundle, z, x_batch, y_batch, alpha, dt
                )
                F, ll, ac = snap(z)
                energies.append(F); losses.append(ll); accs.append(ac)

        return {"energy": energies, "loss": losses, "acc": accs}

    def get_activity_labels(self, bundle):
        """One label per PC activity (same length as z / layer_energies)."""
        layers = bundle["model"]
        out, bi = [], 0
        for layer in layers:
            if isinstance(layer, Stem):
                out.append("stem")
            elif isinstance(layer, BasicBlock):
                out.append(f"block{bi}")
                bi += 1
            elif isinstance(layer, Head):
                out.append("head")
            else:
                out.append(f"layer{len(out)}")
        return out
