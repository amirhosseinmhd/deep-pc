"""res-error-net variant — deep iterative PC with residual error highways.

Extends standard predictive coding with learnable V_{L→i} matrices that carry
the output error e^L directly back to hidden layer i, analogous to ResNet
skip connections but on the backward/error pathway. Addresses the vanishing
error signal problem in deep PC networks.

Augmented free energy:
    F = Σ_ℓ (1/2)·mean‖e^ℓ‖²
      + α · Σ_{i∈S} mean( z^i · ( sg(e^L) @ V_{L→i}ᵀ ) )

The highway factor is the *state* z^i (not the local error e^i), and e^L is
wrapped in stop_gradient — so F_hw has no influence on z^{L-1} during
inference (the last free layer aligns to the clamped target via F_pc only)
and contributes 0 to ∂F/∂W.

Inference dynamics (Euler on dF/dz):
    ż^i = -∂F/∂z^i   for i ∈ {0, …, L-2};  z^{L-1} hard-clamped to y

V update rules (config flag):
    "energy": ΔV_{L→i} = +α z^i (e^L)ᵀ       (derived from ∂F_hw/∂V)
    "state" : ΔV_{L→i} = -α z^i (e^L)ᵀ       (Hebbian anti-gradient, stored
                                              negated so optax subtraction
                                              yields +α z^i (e^L)ᵀ growth)

W update: standard PC gradient on F_pc — F_hw contributes 0 to ∂F/∂W under
the new energy.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import equinox as eqx
import jpc

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list
from variants.dyt import DyTLayer


# Module-level JIT helpers. eqx.filter_jit auto-partitions args: jax arrays →
# traced (dynamic), everything else (Python ints/floats/strings/callables and
# lists/dicts of those) → static. ResErrorNetVariant is an empty, hashable
# class so `variant` is treated as static and the compile is cached per call
# site. Without these wrappers the inference loop runs ~40× eager dispatches
# per batch, dominating training time.

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
    """T-step Adam-on-z scan. Maintains per-element first/second moments of
    ∂F/∂z and applies the bias-corrected Adam update each step."""
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
    """One Adam-on-z step. Public so the diagnostic loop can drive it."""
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


class ResErrorNetVariant:
    """True iterative PC with residual error highways V_{L→i}."""

    @property
    def name(self):
        return "res-error-net"

    @property
    def has_batch_stats(self):
        return False

    def create_model(self, key, depth, width, act_fn, **kwargs):
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        highway_every_k = kwargs.get("highway_every_k", 2)
        s_mode = kwargs.get("s_mode", "stride")
        forward_skip_every = kwargs.get("forward_skip_every", 0)
        v_init_scale = kwargs.get("v_init_scale", 0.01)
        # "unit_gaussian" mirrors rec-LRA paper (p.16); "jpc_default" keeps
        # the 1/√fan_in scale which is stable without ZCA preprocessing.
        init_scheme = kwargs.get("res_init_scheme", "jpc_default")
        # "sp" or "mupc" (Innocenti et al. 2025). When "mupc", the forward
        # pass applies per-layer scalings (1/√D, 1/√N, …, 1/N) and weights
        # are re-initialised as N(0, 1) — overriding init_scheme.
        param_type = kwargs.get("res_param_type", "sp")
        # "euler" (plain gradient flow) or "adam" (per-coordinate adaptive
        # step size on z). Adam treats `dt` as a learning rate and is far
        # less sensitive to its value.
        inference_method = kwargs.get("inference_method", "euler")
        inference_b1 = kwargs.get("inference_b1", 0.9)
        inference_b2 = kwargs.get("inference_b2", 0.999)
        inference_eps = kwargs.get("inference_eps", 1e-8)
        # DyT (Dynamic Tanh) normalization. dyt_norm ∈ {"off","pre","post"}.
        # pre  → DyT applied to a layer's input (feature dim = prev width)
        # post → DyT applied after the activation (feature dim = this width)
        # dyt_layers ∈ {"hidden","all_internal"} or an explicit list of layer
        # indices; the output layer L-1 is always excluded.
        dyt_norm = kwargs.get("dyt_norm", "off")
        dyt_init_alpha = kwargs.get("dyt_init_alpha", 0.5)
        dyt_layers_spec = kwargs.get("dyt_layers", "hidden")
        if dyt_norm not in ("off", "pre", "post"):
            raise ValueError(f"Unknown dyt_norm: {dyt_norm!r}")

        key, subkey = jr.split(key)
        model = jpc.make_mlp(
            key=subkey, input_dim=input_dim, width=width, depth=depth,
            output_dim=output_dim, act_fn=act_fn, use_bias=False,
            param_type="sp",
        )
        act_fn_callable = jpc.get_act_fn(act_fn)

        if param_type == "mupc":
            # μPC: weights N(0, 1) on every layer (incl. output); scalings
            # in the forward pass do the variance work.
            model = self._reinit_unit_gaussian(model, key, include_output=True)
            key, _ = jr.split(key)
        elif init_scheme == "unit_gaussian":
            model = self._reinit_unit_gaussian(model, key)
            key, _ = jr.split(key)
        elif init_scheme == "kaiming":
            model = self._reinit_kaiming(model, key, act_fn)
            key, _ = jr.split(key)

        dims = [output_dim if (i + 1) == depth else width for i in range(depth)]
        output_idx = depth - 1
        candidates = list(range(1, output_idx))   # layers 1 .. depth-2

        if s_mode == "stride":
            # S = {output_idx - k, output_idx - 2k, …} ∩ [1, output_idx - 1]
            S = []
            for step in range(1, depth):
                i = output_idx - highway_every_k * step
                if 1 <= i <= output_idx - 1:
                    S.append(i)
                else:
                    break
        elif s_mode == "dense":
            S = list(candidates)
        elif s_mode == "sparse":
            S = candidates[-1:] if candidates else []
        elif s_mode == "random":
            # |S| matched to the stride default for fair comparison
            target_size = max(1, len(candidates) // highway_every_k)
            target_size = min(target_size, len(candidates))
            if candidates:
                key, sub = jr.split(key)
                perm = jr.permutation(sub, jnp.array(candidates))
                S = sorted(int(x) for x in perm[:target_size].tolist())
            else:
                S = []
        else:
            raise ValueError(f"Unknown s_mode: {s_mode!r}")
        S.sort()

        V_list = []
        for i in S:
            key, sub = jr.split(key)
            V_list.append(v_init_scale * jr.normal(sub, (dims[i], dims[output_idx])))

        # DyT layer construction. Built per active layer index; feature dim
        # depends on placement (pre = previous-layer width, post = this layer's
        # width). Output layer is always excluded — z[L-1] is hard-clamped to
        # y, and DyT'ing the prediction would distort the supervision signal.
        if dyt_norm == "off":
            dyt_indices = []
            dyt_list = []
        else:
            if dyt_layers_spec == "hidden":
                candidate_layers = list(range(1, depth - 1))
            elif dyt_layers_spec == "all_internal":
                candidate_layers = list(range(0, depth - 1))
            elif isinstance(dyt_layers_spec, (list, tuple)):
                candidate_layers = [int(i) for i in dyt_layers_spec
                                    if 0 <= int(i) <= depth - 2]
            else:
                raise ValueError(f"Unknown dyt_layers: {dyt_layers_spec!r}")
            dyt_indices = sorted(set(candidate_layers))
            dyt_list = []
            for li in dyt_indices:
                if dyt_norm == "pre":
                    feat = input_dim if li == 0 else width
                else:  # post
                    feat = output_dim if li == depth - 1 else width
                dyt_list.append(DyTLayer(feat, init_alpha=dyt_init_alpha))

        # Per-layer forward-pass scalings. Under "sp" all are 1.0 (no-op);
        # under "mupc" we use the recipe from Innocenti et al. 2025:
        #   input  = 1/√D, hidden = 1/√N (or 1/√(N·L) when forward skips
        #   are active), output = 1/N.
        if param_type == "mupc":
            D = input_dim
            N = width
            L = depth
            in_scale = D ** -0.5
            hid_scale = (N * L) ** -0.5 if forward_skip_every > 0 else N ** -0.5
            out_scale = 1.0 / N
            param_scalings = [in_scale] + [hid_scale] * (L - 2) + [out_scale]
        else:
            param_scalings = [1.0] * depth

        return {
            "model": model,
            "V_list": V_list,
            "highway_indices": S,
            "highway_every_k": highway_every_k,
            "s_mode": s_mode,
            "forward_skip_every": forward_skip_every,
            "v_init_scale": v_init_scale,
            "init_scheme": init_scheme,
            "param_type": param_type,
            "param_scalings": param_scalings,
            "act_fn": act_fn_callable,
            "act_fn_name": act_fn,
            "depth": depth,
            "dims": dims,
            "inference_method": inference_method,
            "inference_b1": inference_b1,
            "inference_b2": inference_b2,
            "inference_eps": inference_eps,
            "dyt_norm": dyt_norm,
            "dyt_indices": dyt_indices,
            "dyt_list": dyt_list,
        }

    @staticmethod
    def _reinit_kaiming(model, key, act_fn_name):
        """Proper variance-preserving init.

        relu/gelu → Kaiming: N(0, 2/n_in).
        tanh      → Xavier w/ gain 5/3: N(0, (5/3)²/n_in).
        Everything else defaults to Xavier: N(0, 1/n_in).
        """
        if act_fn_name in ("relu", "gelu"):
            gain_sq = 2.0
        elif act_fn_name == "tanh":
            gain_sq = 1.0
        else:
            gain_sq = 1.0

        new_layers = list(model)
        for l in range(len(new_layers)):
            seq = new_layers[l]
            linear = seq.layers[1]
            fan_in = linear.weight.shape[1]
            std = (gain_sq / fan_in) ** 0.5
            key, sub = jr.split(key)
            new_w = std * jr.normal(sub, linear.weight.shape)
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_w)
            new_layers[l] = eqx.tree_at(lambda s: s.layers[1], seq, new_linear)
        return new_layers

    @staticmethod
    def _reinit_unit_gaussian(model, key, include_output=False):
        new_layers = list(model)
        upper = len(new_layers) if include_output else len(new_layers) - 1
        for l in range(upper):
            seq = new_layers[l]
            linear = seq.layers[1]
            key, sub = jr.split(key)
            new_w = jr.normal(sub, linear.weight.shape)
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_w)
            new_layers[l] = eqx.tree_at(lambda s: s.layers[1], seq, new_linear)
        return new_layers

    # --- Forward / predictions / errors ---

    @staticmethod
    def _add_forward_skip(pred, z_hist, layer_idx, forward_skip_every):
        if forward_skip_every <= 0 or layer_idx < forward_skip_every:
            return pred
        src = z_hist[layer_idx - forward_skip_every]
        if src is None or src.shape[-1] != pred.shape[-1]:
            return pred
        return pred + src

    @staticmethod
    def _maybe_dyt(x, layer_idx, dyt_norm, dyt_indices, dyt_list, position):
        """Apply DyT[layer_idx] iff dyt_norm == position and layer_idx is active."""
        if dyt_norm != position or not dyt_indices:
            return x
        if layer_idx not in dyt_indices:
            return x
        idx = dyt_indices.index(layer_idx)
        return vmap(dyt_list[idx])(x)

    def forward_pass(self, bundle, x_batch):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        n = bundle.get("forward_skip_every", 0)
        L = bundle["depth"]
        scalings = bundle.get("param_scalings", [1.0] * L)
        dyt_norm = bundle.get("dyt_norm", "off")
        dyt_indices = bundle.get("dyt_indices", [])
        dyt_list = bundle.get("dyt_list", [])

        z = [None] * L
        h = [None] * L
        z_prev = x_batch
        for l in range(L):
            z_in = self._maybe_dyt(z_prev, l, dyt_norm, dyt_indices, dyt_list, "pre")
            h[l] = scalings[l] * vmap(model[l])(z_in)
            z_pred = vmap(act_fn)(h[l]) if l < L - 1 else h[l]
            z_pred = self._maybe_dyt(
                z_pred, l, dyt_norm, dyt_indices, dyt_list, "post"
            )
            z[l] = self._add_forward_skip(z_pred, z, l, n)
            z_prev = z[l]
        return z, h

    @staticmethod
    def _predictions_and_errors(
        model, z_list, x_batch, act_fn, L,
        forward_skip_every=0, scalings=None,
        dyt_norm="off", dyt_indices=None, dyt_list=None,
    ):
        if scalings is None:
            scalings = [1.0] * L
        if dyt_indices is None:
            dyt_indices = []
        if dyt_list is None:
            dyt_list = []
        mu = [None] * L
        h = [None] * L
        for l in range(L):
            prev = x_batch if l == 0 else z_list[l - 1]
            prev = ResErrorNetVariant._maybe_dyt(
                prev, l, dyt_norm, dyt_indices, dyt_list, "pre"
            )
            h[l] = scalings[l] * vmap(model[l])(prev)
            mu_pred = vmap(act_fn)(h[l]) if l < L - 1 else h[l]
            mu_pred = ResErrorNetVariant._maybe_dyt(
                mu_pred, l, dyt_norm, dyt_indices, dyt_list, "post"
            )
            mu[l] = ResErrorNetVariant._add_forward_skip(
                mu_pred, z_list, l, forward_skip_every
            )
        e = [z_list[l] - mu[l] for l in range(L)]
        return e, mu, h

    def compute_errors(self, bundle, z, x_batch, y_batch):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        n = bundle.get("forward_skip_every", 0)
        L = bundle["depth"]
        scalings = bundle.get("param_scalings", [1.0] * L)
        dyt_norm = bundle.get("dyt_norm", "off")
        dyt_indices = bundle.get("dyt_indices", [])
        dyt_list = bundle.get("dyt_list", [])
        z_list = list(z)
        z_list[L - 1] = y_batch
        e, _, _ = self._predictions_and_errors(
            model, z_list, x_batch, act_fn, L, n, scalings,
            dyt_norm, dyt_indices, dyt_list,
        )
        return e

    # --- Energy ---

    @staticmethod
    def _F_pc(e_list):
        return sum(0.5 * jnp.mean(jnp.sum(el ** 2, axis=1)) for el in e_list)

    @staticmethod
    def _F_highway(z_list, e_list, V_list, S, alpha, L):
        if not S:
            return jnp.array(0.0)
        # stop_grad on e^L: highway no longer pulls z^{L-1} during inference;
        # z^{L-1} is shaped only by F_pc (i.e., aligns to the clamped target).
        e_L = jax.lax.stop_gradient(e_list[L - 1])
        total = jnp.array(0.0)
        for idx, i in enumerate(S):
            Vi = V_list[idx]
            z_i = z_list[i]
            shortcut = e_L @ Vi.T
            total = total + alpha * jnp.mean(jnp.sum(z_i * shortcut, axis=1))
        return total

    def free_energy_z(self, z_free, bundle, x_batch, y_batch, alpha):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        n = bundle.get("forward_skip_every", 0)
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        scalings = bundle.get("param_scalings", [1.0] * L)
        dyt_norm = bundle.get("dyt_norm", "off")
        dyt_indices = bundle.get("dyt_indices", [])
        dyt_list = bundle.get("dyt_list", [])

        z_list = list(z_free) + [y_batch]
        e, _, _ = self._predictions_and_errors(
            model, z_list, x_batch, act_fn, L, n, scalings,
            dyt_norm, dyt_indices, dyt_list,
        )
        return self._F_pc(e) + self._F_highway(z_list, e, V_list, S, alpha, L)

    # --- Inference dynamics ---

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

        # Diagnostic path (used by diagnose_*.py scripts) — keep eager so we
        # can sync per-step free-energy without breaking trace.
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
    # Inference diagnostic
    # ------------------------------------------------------------------

    def diagnose_inference(self, bundle, x_batch, y_batch, alpha, dt, T,
                           loss_type="mse"):
        """Eager T-step inference with per-step (F, loss, acc) trajectory.

        Returns dict with keys "energy", "loss", "acc", each a list of length
        T+1: index 0 is the post-clamp pre-inference state (z = feed-forward,
        z[L-1] := y), indices 1..T are after each inference step.

        loss/acc are computed on μ^L = forward(z^{L-2}) — the model's
        predicted output given the current internal state — not on the
        clamped z[L-1] (which would trivially match y).
        """
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        n = bundle.get("forward_skip_every", 0)
        L = bundle["depth"]
        method = bundle.get("inference_method", "euler")
        b1 = bundle.get("inference_b1", 0.9)
        b2 = bundle.get("inference_b2", 0.999)
        eps = bundle.get("inference_eps", 1e-8)

        z_init, _ = self.forward_pass(bundle, x_batch)
        z = list(z_init)
        z[L - 1] = y_batch

        def snap(z_state):
            z_free = [z_state[l] for l in range(L - 1)]
            F = float(self.free_energy_z(
                z_free, bundle, x_batch, y_batch, alpha
            ))
            _, mu, _ = self._predictions_and_errors(
                model, list(z_state), x_batch, act_fn, L, n,
                bundle.get("param_scalings", [1.0] * L),
            )
            mu_L = mu[L - 1]
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

    # --- Parameter updates ---

    def compute_W_updates(self, bundle, z, x_batch, y_batch):
        """∂F_aug/∂W^l (and ∂F_aug/∂DyT params) via autodiff.

        Under the new highway energy F_hw = α·Σ z^i·(sg(e^L) @ V_iᵀ), z is
        constant w.r.t. model params and stop_gradient blocks flow through
        e^L, so F_hw contributes 0 to ∂F/∂W. The W update therefore
        reduces to ∂F_pc/∂W (we still go through the augmented closure for
        symmetry with free_energy_z and to keep the API uniform).

        When DyT is enabled, gradients also flow into the DyT layers' (α, γ,
        β) parameters. Returns `(delta_W, delta_dyt)` — `delta_dyt` is an
        empty list when `dyt_norm == "off"`.
        """
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]
        alpha = bundle.get("alpha_for_w_grad", None)

        scalings = bundle.get("param_scalings", [1.0] * L)
        dyt_norm = bundle.get("dyt_norm", "off")
        dyt_indices = bundle.get("dyt_indices", [])
        dyt_list = bundle.get("dyt_list", [])
        z_list = list(z)
        z_list[L - 1] = y_batch

        # eqx.filter_grad differentiates the first positional arg by default,
        # so we pack (model, dyt_list) into a tuple to get gradients on both.
        def aug_energy(params):
            model_, dyt_list_ = params
            e, _, _ = self._predictions_and_errors(
                model_, z_list, x_batch, act_fn, L,
                scalings=scalings,
                dyt_norm=dyt_norm, dyt_indices=dyt_indices,
                dyt_list=dyt_list_,
            )
            f = self._F_pc(e)
            if alpha is not None and S:
                f = f + self._F_highway(z_list, e, V_list, S, alpha, L)
            return f

        grads_model, grads_dyt = eqx.filter_grad(aug_energy)((model, dyt_list))

        delta_W = []
        for l in range(L):
            seq_grad = grads_model[l]
            linear_grad = seq_grad.layers[1]
            delta_W.append(linear_grad.weight)
        return delta_W, list(grads_dyt)

    def compute_V_updates(self, bundle, z, x_batch, y_batch, alpha,
                          rule="energy", v_reg=0.0):
        """ΔV_{L→i}. Sign convention: treat as 'gradient', i.e. V ← V - lr·ΔV.

        energy: ΔV = +α z^i (e^L)ᵀ + ρ·V     (∂F_hw_new/∂V + ∂F_reg/∂V)
        state : ΔV = -α z^i (e^L)ᵀ + ρ·V     (Hebbian anti-gradient: subtracting yields growth)

        Under the new highway energy F_hw = α·Σ z^i·(sg(e^L) @ V_iᵀ), the two
        rules differ only in sign — same convention as the resnet18 variant.

        v_reg (ρ) adds an L2 penalty (ρ/2)·‖V‖² to the augmented free energy,
        bounding F below in V and preventing unbounded growth of the highway
        term.
        """
        S = bundle["highway_indices"]
        L = bundle["depth"]
        V_list = bundle["V_list"]
        batch_size = x_batch.shape[0]

        e = self.compute_errors(bundle, z, x_batch, y_batch)
        e_L = e[L - 1]

        delta_V = []
        for idx, i in enumerate(S):
            if rule == "energy":
                # ∂F_hw_new/∂V_i = α · z^i (e^L)^T / B
                dV = alpha * jnp.einsum("bi,bj->ij", z[i], e_L) / batch_size
            elif rule == "state":
                # Hebbian anti-gradient: grows V along z^i (e^L)^T
                dV = -alpha * jnp.einsum("bi,bj->ij", z[i], e_L) / batch_size
            else:
                raise ValueError(f"Unknown v_update_rule: {rule!r}")
            if v_reg and v_reg > 0.0:
                dV = dV + v_reg * V_list[idx]
            delta_V.append(dV)
        return delta_V

    # --- Apply updates ---

    def apply_optax_updates(self, bundle, delta_W, delta_V,
                            w_optim, w_opt_state, v_optim, v_opt_state,
                            delta_dyt=None, dyt_opt_state=None):
        model = list(bundle["model"])
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]
        dyt_list = list(bundle.get("dyt_list", []))
        if delta_dyt is None:
            delta_dyt = []
        if dyt_opt_state is None:
            dyt_opt_state = []

        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]
            updates, w_opt_state[l] = w_optim.update(
                delta_W[l], w_opt_state[l], linear.weight
            )
            new_weight = linear.weight + updates
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(lambda seq: seq.layers[1], layer, new_linear)

        for idx in range(len(S)):
            updates, v_opt_state[idx] = v_optim.update(
                delta_V[idx], v_opt_state[idx], V_list[idx]
            )
            V_list[idx] = V_list[idx] + updates

        for idx in range(len(dyt_list)):
            params = eqx.filter(dyt_list[idx], eqx.is_array)
            updates, dyt_opt_state[idx] = w_optim.update(
                delta_dyt[idx], dyt_opt_state[idx], params
            )
            dyt_list[idx] = eqx.apply_updates(dyt_list[idx], updates)

        return (
            {**bundle, "model": model, "V_list": V_list, "dyt_list": dyt_list},
            w_opt_state, v_opt_state, dyt_opt_state,
        )

    def apply_optax_updates_w_only(self, bundle, delta_W, w_optim, w_opt_state,
                                   delta_dyt=None, dyt_opt_state=None):
        model = list(bundle["model"])
        dyt_list = list(bundle.get("dyt_list", []))
        if delta_dyt is None:
            delta_dyt = []
        if dyt_opt_state is None:
            dyt_opt_state = []

        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]
            updates, w_opt_state[l] = w_optim.update(
                delta_W[l], w_opt_state[l], linear.weight
            )
            new_weight = linear.weight + updates
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(lambda seq: seq.layers[1], layer, new_linear)

        for idx in range(len(dyt_list)):
            params = eqx.filter(dyt_list[idx], eqx.is_array)
            updates, dyt_opt_state[idx] = w_optim.update(
                delta_dyt[idx], dyt_opt_state[idx], params
            )
            dyt_list[idx] = eqx.apply_updates(dyt_list[idx], updates)

        return (
            {**bundle, "model": model, "dyt_list": dyt_list},
            w_opt_state, dyt_opt_state,
        )

    def apply_sgd_updates(self, bundle, delta_W, delta_V, lr_W, lr_V,
                         delta_dyt=None):
        model = list(bundle["model"])
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]
        dyt_list = list(bundle.get("dyt_list", []))
        if delta_dyt is None:
            delta_dyt = []

        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]
            new_weight = linear.weight - lr_W * delta_W[l]
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(lambda seq: seq.layers[1], layer, new_linear)

        for idx in range(len(S)):
            V_list[idx] = V_list[idx] - lr_V * delta_V[idx]

        for idx in range(len(dyt_list)):
            scaled = jax.tree_util.tree_map(
                lambda g: -lr_W * g if eqx.is_array(g) else g,
                delta_dyt[idx],
            )
            dyt_list[idx] = eqx.apply_updates(dyt_list[idx], scaled)

        return {**bundle, "model": model, "V_list": V_list, "dyt_list": dyt_list}

    # --- Optimizer state init ---

    def init_w_optim_states(self, bundle, w_optim):
        states = []
        for layer in bundle["model"]:
            linear = layer.layers[1]
            states.append(w_optim.init(linear.weight))
        return states

    def init_v_optim_states(self, bundle, v_optim):
        return [v_optim.init(V) for V in bundle["V_list"]]

    def init_dyt_optim_states(self, bundle, dyt_optim):
        """Per-DyT-layer optimizer state. Empty list when DyT is off — the
        trainer then loops over zero entries, yielding a no-op."""
        return [
            dyt_optim.init(eqx.filter(dyt, eqx.is_array))
            for dyt in bundle.get("dyt_list", [])
        ]

    # --- Evaluation / introspection ---

    def evaluate(self, bundle, test_loader):
        avg_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch = img_batch.numpy()
            label_batch = label_batch.numpy()
            z, _ = self.forward_pass(bundle, img_batch)
            preds = z[-1]
            acc = float(jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100)
            avg_acc += acc
        return avg_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        return get_weight_list(bundle["model"])

    def get_V_arrays(self, bundle):
        return list(bundle["V_list"])

    def get_weight_labels(self, bundle):
        return [f"layer{i+1}" for i in range(len(self.get_weight_arrays(bundle)))]

    def get_wandb_log_indices(self, bundle):
        return list(range(len(self.get_weight_arrays(bundle))))

    def get_activity_labels(self, bundle):
        L = bundle["depth"] if "depth" in bundle else len(self.get_weight_arrays(bundle))
        return [f"layer{i+1}" for i in range(L)]
