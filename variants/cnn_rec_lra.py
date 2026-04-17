"""CNN rec-LRA variant — convolutional Recursive Local Representation Alignment.

Extends rec-LRA (Ororbia & Mali, AAAI 2023) to convolutional networks for
CIFAR-10.  The algorithm is identical to the MLP variant (single forward pass +
backward target sweep + Hebbian updates) with two adaptations:

  1. Convolutional layers: weight updates use JAX VJP so the Hebbian rule
     ΔW = e_l ⊛ z_prev is computed correctly without manual cross-correlation.

  2. Channel-only E matrices for conv layers: E[l] has shape (C_l, C_source)
     instead of (C_l*H*W, C_src*H*W).  The correction signal d[l] is computed
     from spatially-averaged errors and then broadcast back to the full spatial
     shape, keeping memory tractable.

Architecture (default, CIFAR-10, 9 layers):
  Conv(3→32,   3×3, stride=1, pad=1), act   → [B, 32,  32, 32]   layer 0
  Conv(32→64,  3×3, stride=2, pad=1), act   → [B, 64,  16, 16]   layer 1
  Conv(64→64,  3×3, stride=1, pad=1), act   → [B, 64,  16, 16]   layer 2
  Conv(64→128, 3×3, stride=2, pad=1), act   → [B, 128,  8,  8]   layer 3
  Conv(128→128,3×3, stride=1, pad=1), act   → [B, 128,  8,  8]   layer 4
  Conv(128→256,3×3, stride=2, pad=1), act   → [B, 256,  4,  4]   layer 5
  Conv(256→256,3×3, stride=1, pad=1), act   → [B, 256,  4,  4]   layer 6
  Flatten                                   → [B, 4096]
  Linear(4096→512), act                     → [B, 512]            layer 7
  Linear(512→10)                            → [B, 10]             layer 8 (output)

Striding rule: stride=2 at odd-indexed conv layers (1, 3, 5, …), stride=1
elsewhere.  This gives progressive downsampling (VGG-style) and lets networks
with many conv layers remain valid even at 9+ total layers.

The `n_fc_hidden` parameter controls how many hidden FC layers sit between
Flatten and the output layer (default 1).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import equinox as eqx
import equinox.nn as nn
import numpy as np

import jpc

from config import OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from variants.rec_lra_common import alpha_mix_error


def _adj_source_dim(l, n_conv, layer_shapes, flat_dims):
    """Effective scalar dimension of e[l] when used as ADJACENT-E source.

    Adjacent E matrices remain channel-only for conv layers (full-rank
    would explode to e.g. 16k×16k). For FC layers it is the flat dim.
    """
    if l < n_conv:
        return int(layer_shapes[l][0])   # C_l
    return int(flat_dims[l])             # D_l


def _adj_e_source(e_l, l, n_conv):
    """Adjacent error source — channel-averaged for conv, raw for FC."""
    if l < n_conv:
        return e_l.mean(axis=(2, 3))     # [B, C_l]
    return e_l                           # [B, D_l]


class CNNRecLRAVariant:
    """rec-LRA applied to a simple CNN for CIFAR-10."""

    @property
    def name(self):
        return "CNN-rec-LRA"

    @property
    def has_batch_stats(self):
        return False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def create_model(self, key, depth, width, act_fn, **kwargs):
        """Build the CNN bundle.

        kwargs:
            cnn_channels       : list of output channels per conv layer
            cnn_fc_width       : hidden FC layer width (default 512)
            n_fc_hidden        : number of hidden FC layers (default 1)
            kernel_size        : conv kernel size (default 3)
            input_shape        : (C, H, W) (default (3, 32, 32))
            output_dim         : number of classes (default 10)
            forward_skip_every : default 0
            error_skip_every   : default 2
            alpha_e_skip       : α for skip-error endpoints (paper: 0.19)
            alpha_e_adj        : α for adjacent-error endpoints (paper: 0.24)
            use_layer_norm     : LayerNorm after each conv & FC hidden (default True)

        Striding rule: stride=2 at odd-indexed conv layers, stride=1 at even.
        """
        cnn_channels       = kwargs.get("cnn_channels",       [32, 64, 64, 128, 128, 256, 256])
        cnn_fc_width       = kwargs.get("cnn_fc_width",       512)
        n_fc_hidden        = kwargs.get("n_fc_hidden",        1)
        kernel_size        = kwargs.get("kernel_size",        3)
        input_shape        = kwargs.get("input_shape",        (3, 32, 32))
        output_dim         = kwargs.get("output_dim",         _DEFAULT_OUTPUT_DIM)
        forward_skip_every = kwargs.get("forward_skip_every", 0)
        error_skip_every   = kwargs.get("error_skip_every",   2)
        alpha_e_skip       = kwargs.get("alpha_e_skip",       0.19)
        alpha_e_adj        = kwargs.get("alpha_e_adj",        0.24)
        use_layer_norm     = kwargs.get("use_layer_norm",     True)

        C_in, H_in, W_in = input_shape
        act_fn_callable   = jpc.get_act_fn(act_fn)
        pad               = (kernel_size - 1) // 2

        layers        = []
        layer_shapes  = []
        layer_norms   = []   # parallel to layers; None where no LN
        cur_C, cur_H, cur_W = C_in, H_in, W_in

        # --- conv layers ---
        for i, out_C in enumerate(cnn_channels):
            stride = 2 if (i % 2 == 1) else 1
            key, sub = jr.split(key)
            conv = nn.Conv2d(
                cur_C, out_C, kernel_size,
                stride=stride, padding=pad,
                use_bias=False, key=sub,
            )
            # Paper p.16: weights init from unit Gaussian.
            key, sub = jr.split(key)
            new_w = jr.normal(sub, conv.weight.shape)
            conv = eqx.tree_at(lambda c: c.weight, conv, new_w)
            layers.append(conv)

            cur_H = (cur_H + 2 * pad - kernel_size) // stride + 1
            cur_W = (cur_W + 2 * pad - kernel_size) // stride + 1
            layer_shapes.append((out_C, cur_H, cur_W))

            if use_layer_norm:
                ln = nn.LayerNorm(shape=(out_C, cur_H, cur_W),
                                  use_weight=False, use_bias=False)
                layer_norms.append(ln)
            else:
                layer_norms.append(None)

            cur_C = out_C

        n_conv   = len(cnn_channels)
        flat_dim = int(cur_C * cur_H * cur_W)

        # --- hidden FC layers ---
        fc_in = flat_dim
        for _ in range(n_fc_hidden):
            key, sub = jr.split(key)
            lin = nn.Linear(fc_in, cnn_fc_width, use_bias=False, key=sub)
            key, sub = jr.split(key)
            new_w = jr.normal(sub, lin.weight.shape)
            lin = eqx.tree_at(lambda l_: l_.weight, lin, new_w)
            layers.append(lin)
            layer_shapes.append((cnn_fc_width,))
            if use_layer_norm:
                layer_norms.append(nn.LayerNorm(shape=(cnn_fc_width,),
                                                use_weight=False, use_bias=False))
            else:
                layer_norms.append(None)
            fc_in = cnn_fc_width

        # --- output FC layer ---
        # Output stays at the equinox default (1/√fan_in) since we train
        # with MSE on one-hot targets rather than softmax+CE; unit-Gaussian
        # at the read-out would push outputs to ~√fan_in magnitude.
        key, sub = jr.split(key)
        out_lin = nn.Linear(fc_in, output_dim, use_bias=False, key=sub)
        layers.append(out_lin)
        layer_shapes.append((output_dim,))
        layer_norms.append(None)   # no LN on output

        total_depth = len(layers)
        flat_dims   = [int(np.prod(s)) for s in layer_shapes]

        # --- E matrices ---
        # Skip-from-output  (l % m == 0): full FC of shape (D_l, output_dim)
        #                                  – paper-spec dimensionality.
        # Adjacent          (otherwise) : channel-only for conv to keep
        #                                  memory tractable; full FC for FC.
        m = error_skip_every
        E = [None] * total_depth
        e_is_full = [False] * total_depth   # tracks E layout per layer
        for l in range(1, total_depth - 1):
            is_skip = (m > 0 and l % m == 0)
            key, sub = jr.split(key)
            if is_skip:
                tgt_dim = int(flat_dims[l])
                src_dim = int(output_dim)
                E[l] = jr.normal(sub, (tgt_dim, src_dim))
                e_is_full[l] = True
            else:
                tgt_dim = _adj_source_dim(l, n_conv, layer_shapes, flat_dims)
                src_dim = _adj_source_dim(l + 1, n_conv, layer_shapes, flat_dims)
                E[l] = jr.normal(sub, (tgt_dim, src_dim))
                e_is_full[l] = (l >= n_conv)   # full only for FC adjacent

        return {
            "model":              layers,
            "layer_norms":        layer_norms,
            "E":                  E,
            "e_is_full":          e_is_full,
            "act_fn":             act_fn_callable,
            "depth":              total_depth,
            "n_conv":             n_conv,
            "n_fc_hidden":        n_fc_hidden,
            "layer_shapes":       layer_shapes,
            "flat_dims":          flat_dims,
            "input_shape":        input_shape,
            "forward_skip_every": forward_skip_every,
            "error_skip_every":   error_skip_every,
            "alpha_e_skip":       alpha_e_skip,
            "alpha_e_adj":        alpha_e_adj,
            "use_layer_norm":     use_layer_norm,
        }

    # ------------------------------------------------------------------
    # Forward pass  (RUNMODEL)
    # ------------------------------------------------------------------

    def forward_pass(self, bundle, x_batch):
        """Single forward pass.

        h[l] is the *post-LayerNorm pre-activation* signal — i.e. the value
        used for target generation in the backward sweep. The Hebbian W
        update still uses the raw pre-LN preactivation, since that is what
        the conv weight produces.
        """
        layers       = bundle["model"]
        layer_norms  = bundle.get("layer_norms", [None] * bundle["depth"])
        act_fn       = bundle["act_fn"]
        n_conv       = bundle["n_conv"]
        n            = bundle["forward_skip_every"]
        L            = bundle["depth"]
        input_shape  = bundle["input_shape"]

        B = x_batch.shape[0]
        z_prev = x_batch.reshape(B, *input_shape)

        z = [None] * L
        h = [None] * L

        for l in range(L):
            if l == n_conv:
                z_prev = z_prev.reshape(B, -1)

            pre = vmap(layers[l])(z_prev)

            # Forward skip (dimension-guarded)
            if n > 0 and l >= n and l % n == 0:
                src = z[l - n]
                if src is not None and src.shape == pre.shape:
                    pre = pre + src

            # LayerNorm on hidden layers; output layer left alone.
            if layer_norms[l] is not None and l < L - 1:
                pre = vmap(layer_norms[l])(pre)

            h[l] = pre

            if l < L - 1:
                z[l] = vmap(act_fn)(pre)
            else:
                z[l] = pre

            z_prev = z[l]

        return z, h

    # ------------------------------------------------------------------
    # Target & error computation  (CALCERRRUNITS)
    # ------------------------------------------------------------------

    def compute_targets_and_errors(self, bundle, z, h, y_batch, beta):
        """Backward sweep: compute per-layer targets and errors.

        Skip-from-output E entries are full-rank (D_l, output_dim); the
        displacement is reshaped to the spatial layout of the layer.
        Adjacent E entries are channel-only for conv (broadcast spatially)
        and full for FC, matching create_model.
        """
        E_matrices   = bundle["E"]
        e_is_full    = bundle["e_is_full"]
        m            = bundle["error_skip_every"]
        act_fn       = bundle["act_fn"]
        L            = bundle["depth"]
        n_conv       = bundle["n_conv"]
        layer_shapes = bundle["layer_shapes"]
        alpha_skip   = bundle.get("alpha_e_skip", 0.19)
        alpha_adj    = bundle.get("alpha_e_adj", 0.24)
        B            = z[0].shape[0]

        e = [None] * L
        d = [None] * L

        # Output is treated as a skip endpoint
        e[L - 1] = alpha_mix_error(z[L - 1], y_batch, alpha_skip)

        for l in range(L - 2, -1, -1):
            is_skip = (m > 0 and l % m == 0)

            if l == 0:
                d[l] = jnp.zeros_like(z[l])
            elif E_matrices[l] is not None:
                if is_skip:
                    e_src = e[L - 1]                       # [B, output_dim]
                else:
                    e_src = _adj_e_source(e[l + 1], l + 1, n_conv)

                d_vec = e_src @ E_matrices[l].T            # [B, tgt_dim]

                if e_is_full[l] and l < n_conv:
                    # Full-rank skip-from-output to a conv layer: reshape
                    # the (D_l,) displacement back to (C_l, H_l, W_l).
                    C_l, H_l, W_l = layer_shapes[l]
                    d[l] = d_vec.reshape(B, C_l, H_l, W_l)
                elif l < n_conv:
                    # Channel-only adjacent: broadcast spatially.
                    C_l, H_l, W_l = layer_shapes[l]
                    d[l] = jnp.broadcast_to(
                        d_vec.reshape(B, C_l, 1, 1), (B, C_l, H_l, W_l)
                    )
                else:
                    d[l] = d_vec                            # FC layer
            else:
                d[l] = jnp.zeros_like(z[l])

            if l < L - 1:
                y_l = vmap(act_fn)(h[l] - beta * d[l])
            else:
                y_l = h[l] - beta * d[l]

            alpha_l = alpha_skip if is_skip else alpha_adj
            e[l] = alpha_mix_error(z[l], y_l, alpha_l)

        return e, d

    # ------------------------------------------------------------------
    # Hebbian weight updates  (COMPUTEUPDATES)
    # ------------------------------------------------------------------

    def compute_hebbian_updates(self, bundle, z, e, d, x_batch):
        """Compute Hebbian deltas for all W and E matrices."""
        layers       = bundle["model"]
        E_matrices   = bundle["E"]
        m            = bundle["error_skip_every"]
        gamma        = bundle.get("gamma_E", 0.01)
        L            = bundle["depth"]
        n_conv       = bundle["n_conv"]
        input_shape  = bundle["input_shape"]
        B            = x_batch.shape[0]

        x_spatial = x_batch.reshape(B, *input_shape)   # [B, C, H, W]

        delta_W = []
        delta_E = [None] * L

        # --- W updates ---
        for l in range(L):
            # Determine pre-synaptic activations entering layer l
            if l == 0:
                z_prev = x_spatial          # [B, C_in, H, W], no extra activation
            elif l == n_conv:
                # First FC layer: input is last conv output (spatial), flatten
                z_prev = z[l - 1].reshape(B, -1)
            else:
                z_prev = z[l - 1]           # already post-activation

            layer = layers[l]

            if l < n_conv:
                # Conv layer: Hebbian = ∂/∂W [<conv(z_prev,W), e_l>]
                e_l = e[l]                  # [B, C_out, H_out, W_out]

                def _weighted_sum(W, _z_prev=z_prev, _e_l=e_l, _layer=layer):
                    new_conv = eqx.tree_at(lambda c: c.weight, _layer, W)
                    out = vmap(new_conv)(_z_prev)
                    return jnp.sum(out * _e_l)

                dW = jax.grad(_weighted_sum)(layer.weight) / B
            else:
                # FC layer: standard outer-product Hebbian
                e_l = e[l]                  # [B, D_out_l]
                dW = jnp.einsum('bi,bj->ij', e_l, z_prev) / B

            delta_W.append(dW)

        # --- E updates (Hebbian Eq. 6): ΔE = -γ * d ⊗ e_src ---
        # The dimensionality of d depends on the E layout: full-rank skip
        # uses flattened d[l]; channel-only adjacent uses spatial average.
        e_is_full = bundle["e_is_full"]
        for l in range(2, L - 1):
            if E_matrices[l] is None:
                continue
            is_skip = (m > 0 and l % m == 0)

            if e_is_full[l]:
                # Full layout: d[l] is (B, ...spatial); flatten target side
                d_flat = d[l].reshape(B, -1)          # [B, D_l]
            else:
                d_flat = d[l].mean(axis=(2, 3)) if l < n_conv else d[l]

            if is_skip:
                e_src = e[L - 1]                       # [B, output_dim]
            else:
                e_src = _adj_e_source(e[l + 1], l + 1, n_conv)

            delta_E[l] = -gamma * jnp.einsum('bi,bj->ij', d_flat, e_src) / B

        return delta_W, delta_E

    # ------------------------------------------------------------------
    # Gradient-based E updates  (rLRA-dx variant)
    # ------------------------------------------------------------------

    def compute_grad_E_updates(self, bundle, z, h, y_batch, beta):
        """True gradient ∂D/∂E via JAX autodiff (biologically implausible).

        D = Σ_l ||e_l||² summed over all layers.
        Only E matrices are differentiated; W stays Hebbian.
        """
        E_matrices   = bundle["E"]
        e_is_full    = bundle["e_is_full"]
        m            = bundle["error_skip_every"]
        act_fn       = bundle["act_fn"]
        L            = bundle["depth"]
        n_conv       = bundle["n_conv"]
        layer_shapes = bundle["layer_shapes"]
        alpha_skip   = bundle.get("alpha_e_skip", 0.19)
        alpha_adj    = bundle.get("alpha_e_adj", 0.24)
        B            = z[0].shape[0]

        active_indices = [l for l in range(L) if E_matrices[l] is not None]
        E_active       = [E_matrices[l] for l in active_indices]

        if not E_active:
            return [None] * L

        def discrepancy_fn(E_active_list):
            E_full = [None] * L
            for idx, l in enumerate(active_indices):
                E_full[l] = E_active_list[idx]

            e  = [None] * L
            e[L - 1] = alpha_mix_error(z[L - 1], y_batch, alpha_skip)

            for l in range(L - 2, -1, -1):
                is_skip = (m > 0 and l % m == 0)
                if l == 0:
                    d_l = jnp.zeros_like(z[l])
                elif E_full[l] is not None:
                    if is_skip:
                        e_src = e[L - 1]
                    else:
                        e_src = _adj_e_source(e[l + 1], l + 1, n_conv)

                    d_vec = e_src @ E_full[l].T

                    if e_is_full[l] and l < n_conv:
                        C_l, H_l, W_l = layer_shapes[l]
                        d_l = d_vec.reshape(B, C_l, H_l, W_l)
                    elif l < n_conv:
                        C_l, H_l, W_l = layer_shapes[l]
                        d_l = jnp.broadcast_to(
                            d_vec.reshape(B, C_l, 1, 1),
                            (B, C_l, H_l, W_l),
                        )
                    else:
                        d_l = d_vec
                else:
                    d_l = jnp.zeros_like(z[l])

                if l < L - 1:
                    y_l = vmap(act_fn)(h[l] - beta * d_l)
                else:
                    y_l = h[l] - beta * d_l

                alpha_l = alpha_skip if is_skip else alpha_adj
                e[l] = alpha_mix_error(z[l], y_l, alpha_l)

            D = sum(
                jnp.mean(jnp.sum(e[l].reshape(B, -1) ** 2, axis=1))
                for l in range(L) if e[l] is not None
            )
            return D

        grad_E_active = jax.grad(discrepancy_fn)(E_active)

        delta_E = [None] * L
        for idx, l in enumerate(active_indices):
            delta_E[l] = grad_E_active[idx]

        return delta_E

    # ------------------------------------------------------------------
    # Applying updates
    # ------------------------------------------------------------------

    def apply_hebbian_updates(self, bundle, delta_W, delta_E, lr_W, lr_E):
        """SGD step for W and E."""
        layers      = list(bundle["model"])
        E_matrices  = list(bundle["E"])

        for l in range(len(layers)):
            new_w = layers[l].weight - lr_W * delta_W[l]
            layers[l] = eqx.tree_at(lambda lay: lay.weight, layers[l], new_w)

        for l in range(len(E_matrices)):
            if delta_E[l] is not None and E_matrices[l] is not None:
                E_matrices[l] = E_matrices[l] - lr_E * delta_E[l]

        return {**bundle, "model": layers, "E": E_matrices}

    def apply_optax_updates(self, bundle, delta_W, delta_E,
                             w_optim, w_opt_state,
                             e_optim, e_opt_state):
        """Adam step for W and E (treats Hebbian deltas as pseudo-gradients)."""
        layers      = list(bundle["model"])
        E_matrices  = list(bundle["E"])

        for l in range(len(layers)):
            updates, w_opt_state[l] = w_optim.update(
                delta_W[l], w_opt_state[l], layers[l].weight
            )
            new_w = layers[l].weight + updates   # optax negates internally
            layers[l] = eqx.tree_at(lambda lay: lay.weight, layers[l], new_w)

        for l in range(len(E_matrices)):
            if delta_E[l] is not None and E_matrices[l] is not None:
                updates, e_opt_state[l] = e_optim.update(
                    delta_E[l], e_opt_state[l], E_matrices[l]
                )
                E_matrices[l] = E_matrices[l] + updates

        return {**bundle, "model": layers, "E": E_matrices}, w_opt_state, e_opt_state

    def init_w_optim_states(self, bundle, w_optim):
        """Return per-layer Adam states for W matrices."""
        return [w_optim.init(layer.weight) for layer in bundle["model"]]

    # ------------------------------------------------------------------
    # PCVariant protocol stubs (needed by run_training.py infrastructure)
    # ------------------------------------------------------------------

    def init_activities(self, bundle, x_batch):
        z, _ = self.forward_pass(bundle, x_batch)
        return z, None, bundle

    def get_params_for_jpc(self, bundle):
        return (bundle["model"], None)

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, bundle):
        return (eqx.filter(bundle["model"], eqx.is_array), None)

    def post_learning_step(self, bundle, result, batch_stats):
        return {"model": result["model"]}

    def compute_condition_number(self, bundle, x, y):
        return float('nan'), jnp.array([])

    # ------------------------------------------------------------------
    # Evaluation & weight access
    # ------------------------------------------------------------------

    def evaluate(self, bundle, test_loader):
        total_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch   = img_batch.numpy()
            label_batch = label_batch.numpy()
            z, _ = self.forward_pass(bundle, img_batch)
            preds = z[-1]
            acc = float(jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100)
            total_acc += acc
        return total_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        return [layer.weight for layer in bundle["model"]]

    def get_E_arrays(self, bundle):
        return [e for e in bundle["E"] if e is not None]
