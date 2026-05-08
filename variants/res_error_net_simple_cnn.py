"""res-error-net-simple-cnn — small plain-CNN backbone with residual error highways.

A configurable, lightweight CNN variant of the residual-error-highway PC
algorithm. The forward path is a chain of `Conv → DyT → activation` blocks
(no in-block skip — the skip is purely on the *backward* error pathway via
V_{L→i}), terminated by a GAP + linear head. Depth/width are controlled by
`cnn_channels` and `cnn_strides`. Designed to:

  - hit 95%+ on MNIST with 3 conv blocks (stem at 28×28, two stride-2 blocks
    landing at 7×7) plus the head, and
  - scale to deeper CIFAR-10 configurations by extending the channel list.

Augmented free energy (identical to the ResNet-18 sibling):
    F = Σ_ℓ (1/2)·mean‖e^ℓ‖²
      + α · Σ_{i∈S} mean( flatten(z^i) · ( sg(e^L) @ V_iᵀ ) )

Layer modules and the DyT normalization are imported from
`variants.res_error_net_resnet18` so the two variants stay bit-identical at
the per-layer level (a Stem here = a Stem there).
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
from variants.res_error_net_resnet18 import (
    DyTConv2d, _make_norm, Stem, Head,
)


class PostActConvBlock(eqx.Module):
    """Conv → activation → DyT (post-activation normalization).

    Mirrors the configuration that unlocked deep MLP training in
    `res_error_net.py` with `--res-dyt-norm post`: applying DyT to the
    *output* of the activation rather than the pre-activation. For relu this
    bounds the post-block range to [β, β+γ] (vs. [0, ∞) for plain conv→norm),
    which materially helps gradient flow through the iterative inference loop.
    """
    conv: nn.Conv2d
    norm: Optional[DyTConv2d]
    act_fn: Callable = eqx.field(static=True)

    def __call__(self, x):                       # x: (C_in, H, W)
        h = self.conv(x)
        a = self.act_fn(h)
        if self.norm is not None:
            a = self.norm(a)
        return a


class FlattenHead(eqx.Module):
    """Flatten + Linear head.

    Default for this variant — under iterative PC inference, the strength of
    the gradient signal carried back from e^L to z^{L-2} scales like
    ‖W_head‖. With GAP+Linear that pull is divided by H·W (≈ 49 at 7×7), so
    z^{L-2} barely moves in T=20 inference steps. Flatten+Linear gives the
    full per-coord pull and lets the conv state actually settle to a useful
    fixed point during inference.
    """
    linear: nn.Linear

    def __call__(self, z_in):                    # z_in: (C, H, W)
        return self.linear(z_in.reshape(-1))


# ----------------------------------------------------------------------
# Module-level JIT helpers. Mirror the ResNet-18 sibling — `variant` is a
# stateless hashable class (treated as static by filter_jit), so each call
# site caches per (variant_type, bundle_static_fields).
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

class ResErrorNetSimpleCNNVariant:
    """Plain conv-stack backbone + V_{L→i} residual error highways."""

    @property
    def name(self):
        return "res-error-net-simple-cnn"

    @property
    def has_batch_stats(self):
        return False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def create_model(self, key, depth=None, width=None, act_fn="relu", **kwargs):
        del depth, width  # depth is determined by `cnn_channels`

        input_shape          = kwargs.get("input_shape", (1, 28, 28))
        output_dim           = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        cnn_channels         = list(kwargs.get("cnn_channels", [16, 32, 64]))
        cnn_strides          = kwargs.get("cnn_strides", None)
        kernel_size          = kwargs.get("kernel_size", 3)
        normalization        = kwargs.get("normalization", "dyt")
        dyt_init_alpha       = kwargs.get("dyt_init_alpha", 0.5)
        # "post" (conv → act → DyT) tracks the user-validated MLP recipe
        # (--res-dyt-norm post). "pre" (conv → DyT → act) reuses Stem and
        # matches the ResNet-18 sibling.
        dyt_position         = kwargs.get("dyt_position", "post")
        if dyt_position not in ("pre", "post"):
            raise ValueError(f"Unknown dyt_position: {dyt_position!r}")
        # "flatten" (default): full Linear(C·H·W → output_dim) — needed so the
        # backward pull from e^L into z^{L-2} is strong enough for T-step
        # inference to converge. "gap": classic GAP + Linear(C → output_dim);
        # has fewer parameters but the pull is divided by H·W and inference
        # struggles to move states in a small T.
        head_type            = kwargs.get("head_type", "flatten")
        if head_type not in ("flatten", "gap"):
            raise ValueError(f"Unknown head_type: {head_type!r}")
        # "mse" → top-layer energy is 0.5‖y - μ_L‖² (default).
        # "ce"  → top-layer energy is CE(softmax(μ_L), y); the highway
        # supervision signal e^L becomes (y - softmax(μ_L)) so V is updated
        # against the CE gradient direction. Inner-layer PC errors stay MSE.
        loss_type            = kwargs.get("loss_type", "mse")
        if loss_type not in ("mse", "ce"):
            raise ValueError(f"Unknown loss_type: {loss_type!r}")
        # "kaiming" (default for relu): sqrt(2/fan_in). Equinox's default
        # 1/sqrt(fan_in) under-scales for relu by sqrt(2), giving forward
        # activations that halve in std at every block (16→32→64 channels:
        # 0.17 → 0.04 → 0.009 in our test) — a vanishing cascade that holds
        # learning back. Kaiming preserves activation variance.
        # "default": Equinox's truncated-normal init (1/sqrt(fan_in)).
        init_scheme          = kwargs.get("init_scheme", "kaiming")
        if init_scheme not in ("default", "kaiming"):
            raise ValueError(f"Unknown init_scheme: {init_scheme!r}")
        highway_include_stem = kwargs.get("highway_include_stem", True)
        v_init_scale         = kwargs.get("v_init_scale", 0.01)
        inference_method     = kwargs.get("inference_method", "euler")
        inference_b1         = kwargs.get("inference_b1", 0.9)
        inference_b2         = kwargs.get("inference_b2", 0.999)
        inference_eps        = kwargs.get("inference_eps", 1e-8)

        n_blocks = len(cnn_channels)
        if cnn_strides is None:
            # Default: stem at full resolution, every subsequent block halves
            # the spatial dims (canonical CIFAR/MNIST layout).
            cnn_strides = [1] + [2] * (n_blocks - 1)
        else:
            cnn_strides = list(cnn_strides)
            if len(cnn_strides) != n_blocks:
                raise ValueError(
                    f"cnn_strides length {len(cnn_strides)} != cnn_channels "
                    f"length {n_blocks}"
                )

        pad = (kernel_size - 1) // 2
        act_fn_callable = jpc.get_act_fn(act_fn)

        C_in, H_in, W_in = input_shape
        layers: List[eqx.Module] = []
        layer_shapes: List[tuple] = []
        cur_C, cur_H, cur_W = C_in, H_in, W_in

        for out_C, stride in zip(cnn_channels, cnn_strides):
            key, sub = jr.split(key)
            conv = nn.Conv2d(
                cur_C, out_C, kernel_size,
                stride=stride, padding=pad,
                use_bias=False, key=sub,
            )
            if init_scheme == "kaiming":
                # Kaiming-He: N(0, 2/fan_in) for relu/gelu. fan_in for a
                # conv is C_in × kH × kW.
                fan_in = cur_C * kernel_size * kernel_size
                key, sub = jr.split(key)
                w = (2.0 / fan_in) ** 0.5 * jr.normal(sub, conv.weight.shape)
                conv = eqx.tree_at(lambda c: c.weight, conv, w)
            norm = _make_norm(out_C, normalization, dyt_init_alpha)
            if dyt_position == "post":
                layers.append(PostActConvBlock(
                    conv=conv, norm=norm, act_fn=act_fn_callable,
                ))
            else:
                # Stem = conv → norm → activation (pre-activation DyT).
                layers.append(Stem(
                    conv=conv, norm=norm, act_fn=act_fn_callable,
                ))

            cur_H = (cur_H + 2 * pad - kernel_size) // stride + 1
            cur_W = (cur_W + 2 * pad - kernel_size) // stride + 1
            cur_C = out_C
            layer_shapes.append((cur_C, cur_H, cur_W))

        # head
        key, sub = jr.split(key)
        if head_type == "flatten":
            flat_in = int(cur_C * cur_H * cur_W)
            head_linear = nn.Linear(flat_in, output_dim, use_bias=False, key=sub)
            layers.append(FlattenHead(linear=head_linear))
        else:  # "gap"
            head_linear = nn.Linear(cur_C, output_dim, use_bias=False, key=sub)
            layers.append(Head(linear=head_linear))
        layer_shapes.append((output_dim,))

        L = len(layers)

        # Highway indices: every conv block (and optionally the stem at l=0)
        start = 0 if highway_include_stem else 1
        S = list(range(start, L - 1))

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
            "dyt_position": dyt_position,
            "head_type": head_type,
            "init_scheme": init_scheme,
            "v_init_scale": v_init_scale,
            "cnn_channels": cnn_channels,
            "cnn_strides": cnn_strides,
            "kernel_size": kernel_size,
            "inference_method": inference_method,
            "inference_b1": inference_b1,
            "inference_b2": inference_b2,
            "inference_eps": inference_eps,
            "loss_type": loss_type,
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
    def _F_pc(e_list, mu_L=None, y=None, loss_type="mse"):
        """Augmented PC energy. Inner layers always use MSE (0.5‖e‖²).

        Top layer:
          mse → 0.5‖y - μ_L‖²
          ce  → CE(softmax(μ_L), y)
        """
        total = jnp.array(0.0)
        for el in e_list[:-1]:
            flat = el.reshape(el.shape[0], -1)
            total = total + 0.5 * jnp.mean(jnp.sum(flat ** 2, axis=1))
        if loss_type == "ce":
            return total + jpc.cross_entropy_loss(mu_L, y)
        last = e_list[-1].reshape(e_list[-1].shape[0], -1)
        return total + 0.5 * jnp.mean(jnp.sum(last ** 2, axis=1))

    @staticmethod
    def _highway_e_top(e_list, mu_L=None, y=None, loss_type="mse"):
        """Supervision error at the top, used by F_highway and ΔV.

          mse → e^L = y - μ_L  (the existing e[-1])
          ce  → e^L = y - softmax(μ_L)  (negative gradient of CE w.r.t. μ_L)
        """
        if loss_type == "ce":
            return y - jax.nn.softmax(mu_L, axis=-1)
        return e_list[-1]

    @staticmethod
    def _F_highway(z_list, e_top, V_list, S, alpha):
        if not S:
            return jnp.array(0.0)
        e_L = jax.lax.stop_gradient(e_top)                    # (B, output_dim)
        total = jnp.array(0.0)
        for idx, i in enumerate(S):
            Vi = V_list[idx]                                  # (D_i, output_dim)
            z_i_flat = z_list[i].reshape(z_list[i].shape[0], -1)
            shortcut = e_L @ Vi.T                             # (B, D_i)
            total = total + alpha * jnp.mean(jnp.sum(z_i_flat * shortcut, axis=1))
        return total

    def free_energy_z(self, z_free, bundle, x_batch, y_batch, alpha):
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        loss_type = bundle.get("loss_type", "mse")
        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z_free) + [y_batch]
        mu = self._predictions(model, z_list, x_spatial, L)
        e = [z_list[l] - mu[l] for l in range(L)]
        f_pc = self._F_pc(e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type)
        e_top = self._highway_e_top(
            e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type,
        )
        return f_pc + self._F_highway(z_list, e_top, V_list, S, alpha)

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
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        alpha = bundle.get("alpha_for_w_grad", None)
        loss_type = bundle.get("loss_type", "mse")

        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z)
        z_list[L - 1] = y_batch

        def aug_energy(m):
            mu = self._predictions(m, z_list, x_spatial, L)
            e = [z_list[l] - mu[l] for l in range(L)]
            f_pc = self._F_pc(e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type)
            if alpha is not None and S:
                e_top = self._highway_e_top(
                    e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type,
                )
                return f_pc + self._F_highway(z_list, e_top, V_list, S, alpha)
            return f_pc

        grads_tree = eqx.filter_grad(aug_energy)(model)
        return jax.tree_util.tree_leaves(eqx.filter(grads_tree, eqx.is_array))

    def compute_V_updates(self, bundle, z, x_batch, y_batch, alpha,
                          rule="energy", v_reg=0.0):
        S = bundle["highway_indices"]
        L = bundle["depth"]
        V_list = bundle["V_list"]
        B = x_batch.shape[0]
        loss_type = bundle.get("loss_type", "mse")

        if loss_type == "ce":
            model = bundle["model"]
            x_spatial = self._reshape_input(bundle, x_batch)
            z_list = list(z)
            z_list[L - 1] = y_batch
            mu = self._predictions(model, z_list, x_spatial, L)
            e = [z_list[l] - mu[l] for l in range(L)]
            e_L = self._highway_e_top(
                e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type,
            )
        else:
            e = self.compute_errors(bundle, z, x_batch, y_batch)
            e_L = e[L - 1]

        delta_V = []
        for idx, i in enumerate(S):
            z_i_flat = z[i].reshape(z[i].shape[0], -1)
            if rule == "energy":
                dV = alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            elif rule == "state":
                dV = -alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            else:
                raise ValueError(f"Unknown v_update_rule: {rule!r}")
            if v_reg and v_reg > 0.0:
                dV = dV + v_reg * V_list[idx]
            delta_V.append(dV)
        return delta_V

    # Fused: one forward+backward yields e, ΔW, ΔV, per-layer energies.
    def compute_updates_fused(self, bundle, z, x_batch, y_batch, alpha,
                              rule="energy", v_reg=0.0):
        model = bundle["model"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        B = x_batch.shape[0]
        loss_type = bundle.get("loss_type", "mse")

        x_spatial = self._reshape_input(bundle, x_batch)
        z_list = list(z)
        z_list[L - 1] = y_batch

        def aug_energy_with_e(m):
            mu = self._predictions(m, z_list, x_spatial, L)
            e = [z_list[l] - mu[l] for l in range(L)]
            f_pc = self._F_pc(e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type)
            if S:
                e_top = self._highway_e_top(
                    e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type,
                )
                hw = self._F_highway(z_list, e_top, V_list, S, alpha)
            else:
                hw = jnp.array(0.0)
            # Stash the highway-error so the V-update path uses the same e^L
            # as the W gradient (avoids a redundant forward under CE).
            return f_pc + hw, (e, mu)

        (_, (e, mu)), grads_tree = eqx.filter_value_and_grad(
            aug_energy_with_e, has_aux=True
        )(model)
        delta_W = jax.tree_util.tree_leaves(eqx.filter(grads_tree, eqx.is_array))

        energies = jnp.stack([
            0.5 * jnp.mean(jnp.sum(el.reshape(el.shape[0], -1) ** 2, axis=1))
            for el in e
        ])

        e_L = self._highway_e_top(
            e, mu_L=mu[L - 1], y=y_batch, loss_type=loss_type,
        )
        delta_V = []
        for idx, i in enumerate(S):
            z_i_flat = z[i].reshape(z[i].shape[0], -1)
            if rule == "energy":
                dV = alpha * jnp.einsum("bi,bj->ij", z_i_flat, e_L) / B
            elif rule == "state":
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

    def apply_optax_updates_w_only(self, bundle, delta_W, w_optim, w_opt_state):
        """W-only update path used when V is frozen (`--res-v-frozen`).

        Required by the trainer's `freeze_v` branch — leaves V_list (and its
        optimizer state) untouched. Per the v_nonlearnable report, on the MLP
        sibling at the default init scale, frozen V matches learnable V
        within seed jitter, so this path lets the CNN cut the per-step cost
        of computing and applying ΔV.
        """
        model = bundle["model"]
        grads_tree = self._grad_leaves_to_tree(delta_W, model)
        params = eqx.filter(model, eqx.is_array)
        updates, w_opt_state = w_optim.update(grads_tree, w_opt_state, params)
        new_model = eqx.apply_updates(model, updates)
        return {**bundle, "model": new_model}, w_opt_state

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
        """Per-leaf labels matching the order of get_weight_arrays.

        Block fields: (conv, norm.alpha, norm.gamma, norm.beta) when DyT is on,
        or just (conv,) when normalization='none'. Head: (linear,).
        """
        layers = bundle["model"]
        labels = []
        conv_idx = 0
        for layer in layers:
            if isinstance(layer, (Stem, PostActConvBlock)):
                p = f"conv{conv_idx}"
                labels.append(f"{p}.conv")
                if layer.norm is not None:
                    labels += [f"{p}.norm.alpha", f"{p}.norm.gamma", f"{p}.norm.beta"]
                conv_idx += 1
            elif isinstance(layer, (Head, FlattenHead)):
                labels.append("head.linear")
            else:
                labels.append(f"unknown_{type(layer).__name__}")
        return labels

    def get_wandb_log_indices(self, bundle):
        """One conv per architectural block + the head linear."""
        labels = self.get_weight_labels(bundle)
        return [
            i for i, name in enumerate(labels)
            if name.endswith(".conv") or name == "head.linear"
        ]

    def get_activity_labels(self, bundle):
        """One label per PC activity (same length as z / layer_energies)."""
        layers = bundle["model"]
        out, ci = [], 0
        for layer in layers:
            if isinstance(layer, (Stem, PostActConvBlock)):
                out.append(f"conv{ci}")
                ci += 1
            elif isinstance(layer, (Head, FlattenHead)):
                out.append("head")
            else:
                out.append(f"layer{len(out)}")
        return out

    # ------------------------------------------------------------------
    # Inference diagnostic
    # ------------------------------------------------------------------

    def diagnose_inference(self, bundle, x_batch, y_batch, alpha, dt, T,
                           loss_type="mse"):
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
