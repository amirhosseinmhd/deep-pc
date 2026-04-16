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


def _source_dim(l, n_conv, layer_shapes, flat_dims):
    """Effective scalar dimension of e[l] when used as E-matrix source.

    For conv layers we spatially-average before multiplying by E, so the
    effective dim is the channel count; for FC layers it is the flat dim.
    """
    if l < n_conv:
        return int(layer_shapes[l][0])   # C_l
    return int(flat_dims[l])             # D_l


def _get_e_source(e_l, l, n_conv):
    """Return the error signal from layer l suitable for E-matrix operations.

    Conv layers: spatially average to [B, C_l].
    FC / output layers: return as-is [B, D_l].
    """
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
                                  (default [32,64,64,128,128,256,256])
            cnn_fc_width       : hidden FC layer width (default 512)
            n_fc_hidden        : number of hidden FC layers (default 1)
            kernel_size        : conv kernel size, same for all layers (default 3)
            input_shape        : (C, H, W) of one input sample (default (3,32,32))
            output_dim         : number of classes (default 10)
            forward_skip_every : int, default 0 (disabled; conv dims rarely match)
            error_skip_every   : int, default 2

        Striding rule: stride=2 at odd-indexed conv layers (1, 3, 5, …);
                       stride=1 at even-indexed conv layers (0, 2, 4, …).
        """
        cnn_channels       = kwargs.get("cnn_channels",       [32, 64, 64, 128, 128, 256, 256])
        cnn_fc_width       = kwargs.get("cnn_fc_width",       512)
        n_fc_hidden        = kwargs.get("n_fc_hidden",        1)
        kernel_size        = kwargs.get("kernel_size",        3)
        input_shape        = kwargs.get("input_shape",        (3, 32, 32))
        output_dim         = kwargs.get("output_dim",         _DEFAULT_OUTPUT_DIM)
        forward_skip_every = kwargs.get("forward_skip_every", 0)
        error_skip_every   = kwargs.get("error_skip_every",   2)

        C_in, H_in, W_in = input_shape
        act_fn_callable   = jpc.get_act_fn(act_fn)
        pad               = (kernel_size - 1) // 2

        layers       = []
        layer_shapes = []
        cur_C, cur_H, cur_W = C_in, H_in, W_in

        # --- conv layers ---
        # Striding rule: stride=2 at odd indices (1, 3, 5, …), stride=1 at even.
        # This gives VGG-style progressive downsampling and keeps spatial resolution
        # valid for networks with many conv layers.
        for i, out_C in enumerate(cnn_channels):
            stride = 2 if (i % 2 == 1) else 1
            key, sub = jr.split(key)
            layers.append(nn.Conv2d(
                cur_C, out_C, kernel_size,
                stride=stride, padding=pad,
                use_bias=False, key=sub,
            ))
            cur_H = (cur_H + 2 * pad - kernel_size) // stride + 1
            cur_W = (cur_W + 2 * pad - kernel_size) // stride + 1
            layer_shapes.append((out_C, cur_H, cur_W))
            cur_C = out_C

        n_conv   = len(cnn_channels)
        flat_dim = int(cur_C * cur_H * cur_W)

        # --- hidden FC layers (n_fc_hidden of them) ---
        fc_in = flat_dim
        for _ in range(n_fc_hidden):
            key, sub = jr.split(key)
            layers.append(nn.Linear(fc_in, cnn_fc_width, use_bias=False, key=sub))
            layer_shapes.append((cnn_fc_width,))
            fc_in = cnn_fc_width

        # --- output FC layer ---
        key, sub = jr.split(key)
        layers.append(nn.Linear(fc_in, output_dim, use_bias=False, key=sub))
        layer_shapes.append((output_dim,))

        total_depth = len(layers)   # n_conv + n_fc_hidden + 1
        flat_dims   = [int(np.prod(s)) for s in layer_shapes]

        # --- E matrices ---
        # E[l] maps an error source signal to a displacement for layer l.
        #   Conv layer l  → shape (C_l,       C_source or D_source)
        #   FC   layer l  → shape (D_l,       D_source)
        # Indexed l = 1 … total_depth-2  (no E for input-proxy or output layer).
        m = error_skip_every
        E = [None] * total_depth
        for l in range(1, total_depth - 1):
            tgt_dim = _source_dim(l, n_conv, layer_shapes, flat_dims)
            if m > 0 and l % m == 0:
                # skip connection from output layer
                src_dim = output_dim
            else:
                # adjacent connection from layer l+1
                src_dim = _source_dim(l + 1, n_conv, layer_shapes, flat_dims)
            fan_in = src_dim
            key, sub = jr.split(key)
            E[l] = jr.normal(sub, (tgt_dim, fan_in)) / jnp.sqrt(float(fan_in))

        return {
            "model":              layers,
            "E":                  E,
            "act_fn":             act_fn_callable,
            "depth":              total_depth,
            "n_conv":             n_conv,
            "n_fc_hidden":        n_fc_hidden,
            "layer_shapes":       layer_shapes,
            "flat_dims":          flat_dims,
            "input_shape":        input_shape,
            "forward_skip_every": forward_skip_every,
            "error_skip_every":   error_skip_every,
        }

    # ------------------------------------------------------------------
    # Forward pass  (RUNMODEL)
    # ------------------------------------------------------------------

    def forward_pass(self, bundle, x_batch):
        """Single forward pass.

        x_batch: [B, C*H*W] (flat CIFAR-10 as delivered by the dataloader)
        Returns z (post-act), h (pre-act) — each a list of L tensors.
        """
        layers       = bundle["model"]
        act_fn       = bundle["act_fn"]
        n_conv       = bundle["n_conv"]
        n            = bundle["forward_skip_every"]
        L            = bundle["depth"]
        input_shape  = bundle["input_shape"]

        B = x_batch.shape[0]
        # reshape flat input to spatial
        z_prev = x_batch.reshape(B, *input_shape)

        z = [None] * L
        h = [None] * L

        for l in range(L):
            # Transition from spatial to flat at the first FC layer
            if l == n_conv:
                z_prev = z_prev.reshape(B, -1)

            h[l] = vmap(layers[l])(z_prev)

            # Forward skip connection (dimension-guarded)
            if n > 0 and l >= n and l % n == 0:
                src = z[l - n]
                if src is not None and src.shape == h[l].shape:
                    h[l] = h[l] + src

            # Post-activation (no activation on output layer)
            if l < L - 1:
                z[l] = vmap(act_fn)(h[l])
            else:
                z[l] = h[l]

            z_prev = z[l]

        return z, h

    # ------------------------------------------------------------------
    # Target & error computation  (CALCERRRUNITS)
    # ------------------------------------------------------------------

    def compute_targets_and_errors(self, bundle, z, h, y_batch, beta):
        """Backward sweep: compute per-layer targets and errors."""
        E_matrices   = bundle["E"]
        m            = bundle["error_skip_every"]
        act_fn       = bundle["act_fn"]
        L            = bundle["depth"]
        n_conv       = bundle["n_conv"]
        layer_shapes = bundle["layer_shapes"]
        B            = z[0].shape[0]

        e = [None] * L
        d = [None] * L

        # Output error (always flat [B, D_out])
        e[L - 1] = z[L - 1] - y_batch

        for l in range(L - 2, -1, -1):
            if l == 0:
                d[l] = jnp.zeros_like(z[l])
            elif E_matrices[l] is not None:
                if m > 0 and l % m == 0:
                    # Skip from output layer
                    e_src = e[L - 1]                           # [B, D_out]
                else:
                    # Adjacent: source is layer l+1
                    e_src = _get_e_source(e[l + 1], l + 1, n_conv)  # [B, C or D]

                d_vec = e_src @ E_matrices[l].T                # [B, tgt_dim]

                if l < n_conv:
                    # Broadcast channel-wise correction to full spatial shape
                    C_l, H_l, W_l = layer_shapes[l]
                    d[l] = d_vec.reshape(B, C_l, 1, 1) * jnp.ones((B, C_l, H_l, W_l))
                else:
                    d[l] = d_vec                               # [B, D_l]
            else:
                d[l] = jnp.zeros_like(z[l])

            # Target and error
            if l < L - 1:
                y_l = vmap(act_fn)(h[l] - beta * d[l])
            else:
                y_l = h[l] - beta * d[l]

            e[l] = z[l] - y_l

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

        # --- E updates (Hebbian Eq. 6): ΔE = -γ * d_avg ⊗ e_src_avg ---
        for l in range(2, L - 1):
            if E_matrices[l] is None:
                continue

            # Spatially-averaged displacement at layer l
            if l < n_conv:
                d_avg = d[l].mean(axis=(2, 3))    # [B, C_l]
            else:
                d_avg = d[l]                       # [B, D_l]

            # Error source
            if m > 0 and l % m == 0:
                e_src = e[L - 1]                   # [B, D_out]
            else:
                e_src = _get_e_source(e[l + 1], l + 1, n_conv)

            delta_E[l] = -gamma * jnp.einsum('bi,bj->ij', d_avg, e_src) / B

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
        m            = bundle["error_skip_every"]
        act_fn       = bundle["act_fn"]
        L            = bundle["depth"]
        n_conv       = bundle["n_conv"]
        layer_shapes = bundle["layer_shapes"]
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
            e[L - 1] = z[L - 1] - y_batch

            for l in range(L - 2, -1, -1):
                if l == 0:
                    d_l = jnp.zeros_like(z[l])
                elif E_full[l] is not None:
                    if m > 0 and l % m == 0:
                        e_src = e[L - 1]
                    else:
                        e_src = _get_e_source(e[l + 1], l + 1, n_conv)

                    d_vec = e_src @ E_full[l].T

                    if l < n_conv:
                        C_l, H_l, W_l = layer_shapes[l]
                        d_l = d_vec.reshape(B, C_l, 1, 1) * jnp.ones((B, C_l, H_l, W_l))
                    else:
                        d_l = d_vec
                else:
                    d_l = jnp.zeros_like(z[l])

                if l < L - 1:
                    y_l = vmap(act_fn)(h[l] - beta * d_l)
                else:
                    y_l = h[l] - beta * d_l

                e[l] = z[l] - y_l

            # Sum squared errors over all layers
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
