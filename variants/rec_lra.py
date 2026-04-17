"""rec-LRA variant — Recursive Local Representation Alignment.

Implements the Hebbian learning algorithm from:
  "Backpropagation-Free Deep Learning with Recursive Local Representation
   Alignment" (Ororbia et al., AAAI 2023)

Key differences from standard PC variants:
  - No iterative inference loop; single forward pass + backward target sweep
  - Hebbian (outer-product) weight updates instead of gradient-based
  - Error skip connections (E matrices) transmit error signals across layers
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial
import jpc
import equinox as eqx
import equinox.nn as nn

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list
from variants.rec_lra_common import alpha_mix_error


class RecLRAVariant:
    """PC variant using rec-LRA: Hebbian updates + error skip connections."""

    @property
    def name(self):
        return "rec-LRA"

    @property
    def has_batch_stats(self):
        return False

    def create_model(self, key, depth, width, act_fn, **kwargs):
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        forward_skip_every = kwargs.get("forward_skip_every", 2)
        error_skip_every = kwargs.get("error_skip_every", 2)
        # Paper p.15: convex L1+L2 error neurons. α=0.19 for skip-error
        # endpoints, α=0.24 for adjacent endpoints. The output layer is
        # treated as a skip endpoint.
        alpha_e_skip = kwargs.get("alpha_e_skip", 0.19)
        alpha_e_adj = kwargs.get("alpha_e_adj", 0.24)

        key, subkey = jr.split(key)
        model = jpc.make_mlp(
            key=subkey, input_dim=input_dim, width=width, depth=depth,
            output_dim=output_dim, act_fn=act_fn, use_bias=False,
            param_type="sp",
        )

        act_fn_callable = jpc.get_act_fn(act_fn)

        # Build layer dimensions list: [input_dim, width, ..., width, output_dim]
        dims = []
        for i in range(depth):
            _out = output_dim if (i + 1) == depth else width
            dims.append(_out)

        # Paper p.16: forward and error weights initialised from a unit
        # Gaussian (Xavier and orthogonal "yielded unsatisfactory
        # performance"). Replace existing forward weights to match.
        model = self._reinit_unit_gaussian(model, key)
        key, _ = jr.split(key)

        # Create E matrices for error transmission.
        # E[l] computes d[l] in the backward sweep (l from L-2 down to 1).
        # E[0] = None (no error to input proxy layer).
        E = [None] * depth
        for l in range(1, depth - 1):
            key, subkey = jr.split(key)
            if error_skip_every > 0 and l % error_skip_every == 0:
                fan_in = dims[-1]
                E[l] = jr.normal(subkey, (dims[l], fan_in))
            else:
                fan_in = dims[l + 1]
                E[l] = jr.normal(subkey, (dims[l], fan_in))

        return {
            "model": model,
            "E": E,
            "forward_skip_every": forward_skip_every,
            "error_skip_every": error_skip_every,
            "act_fn": act_fn_callable,
            "depth": depth,
            "alpha_e_skip": alpha_e_skip,
            "alpha_e_adj": alpha_e_adj,
        }

    @staticmethod
    def _reinit_unit_gaussian(model, key):
        """Re-initialise every hidden Linear in the MLP Sequential to N(0, 1).

        The output read-out is left at the equinox default (1/√fan_in) so
        MSE on one-hot targets stays well-scaled.
        """
        new_layers = list(model)
        L = len(new_layers)
        for l in range(L - 1):  # skip output
            seq = new_layers[l]
            linear = seq.layers[1]
            key, sub = jr.split(key)
            new_w = jr.normal(sub, linear.weight.shape)
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_w)
            new_layers[l] = eqx.tree_at(
                lambda s: s.layers[1], seq, new_linear
            )
        return new_layers

    def forward_pass(self, bundle, x_batch):
        """Run the forward pass (RUNMODEL from Algorithm 1).

        Returns:
            z: list of post-activation values, length L
            h: list of pre-activation values, length L
        """
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        n = bundle["forward_skip_every"]
        L = bundle["depth"]

        z = [None] * L
        h = [None] * L

        z_prev = x_batch
        for l in range(L):
            # model[l] is Sequential([Lambda(phi_l), Linear])
            # It computes W_l * phi_l(z_{l-1}) = h_l
            h[l] = vmap(model[l])(z_prev)

            # Forward skip connection every n layers
            if n > 0 and l >= n and l % n == 0:
                src = z[l - n]
                # Only skip if dimensions match (hidden-to-hidden)
                if src.shape[-1] == h[l].shape[-1]:
                    h[l] = h[l] + src

            # Post-activation (no activation on output layer)
            if l < L - 1:
                z[l] = vmap(act_fn)(h[l])
            else:
                z[l] = h[l]

            z_prev = z[l]

        return z, h

    def compute_targets_and_errors(self, bundle, z, h, y_batch, beta):
        """Compute targets and error neurons (CALCERRRUNITS from Algorithm 1).

        Returns:
            e: list of error vectors, length L
            d: list of displacement vectors, length L
        """
        E_matrices = bundle["E"]
        m = bundle["error_skip_every"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]
        alpha_skip = bundle.get("alpha_e_skip", 0.19)
        alpha_adj = bundle.get("alpha_e_adj", 0.24)

        e = [None] * L
        d = [None] * L

        # Output error: treat as a skip endpoint per the paper
        e[L - 1] = alpha_mix_error(z[L - 1], y_batch, alpha_skip)

        for l in range(L - 2, -1, -1):
            is_skip = (m > 0 and l % m == 0)
            if l == 0:
                d[l] = jnp.zeros_like(z[l])
            elif E_matrices[l] is not None:
                if is_skip:
                    d[l] = e[L - 1] @ E_matrices[l].T
                else:
                    d[l] = e[l + 1] @ E_matrices[l].T
            else:
                d[l] = jnp.zeros_like(z[l])

            if l < L - 1:
                y_l = vmap(act_fn)(h[l] - beta * d[l])
            else:
                y_l = h[l] - beta * d[l]

            alpha_l = alpha_skip if is_skip else alpha_adj
            e[l] = alpha_mix_error(z[l], y_l, alpha_l)

        return e, d

    def compute_hebbian_updates(self, bundle, z, e, d, x_batch):
        """Compute Hebbian weight updates (COMPUTEUPDATES from Algorithm 1).

        W update (Eq. 5): ΔW_l = e_l · phi(z_{l-1})^T
        E update (Eq. 6): ΔE_{j→i} = -γ · (d_i · e_j^T)
          where E_{j→i} at index l computes d[l] from e[source].

        Returns:
            delta_W: list of weight update matrices, length L
            delta_E: list of E matrix updates, length L (None where no E exists)
        """
        E_matrices = bundle["E"]
        m = bundle["error_skip_every"]
        gamma = bundle.get("gamma_E", 0.01)
        L = bundle["depth"]
        batch_size = x_batch.shape[0]

        delta_W = []
        delta_E = [None] * L

        for l in range(L):
            # --- W update ---
            # model[l] = Sequential([Lambda(act_fn), Linear])
            # It computes W_l * phi_l(z_{l-1}), so the effective input to
            # the Linear is phi(z_{l-1}) (Identity for l=0).
            z_prev = x_batch if l == 0 else z[l - 1]
            if l == 0:
                a_prev = z_prev  # Identity activation for first layer
            else:
                a_prev = vmap(bundle["act_fn"])(z_prev)

            # Hebbian: ΔW_l = e_l · a_prev^T, averaged over batch
            dW = jnp.einsum('bi,bj->ij', e[l], a_prev) / batch_size
            delta_W.append(dW)

        # --- E updates (Eq. 6) ---
        # ΔE_{j→i} = -γ · d_i · e_j^T
        # E[l] computes d[l] from e_source; update uses d[l] and e_source
        for l in range(2, L - 1):
            if E_matrices[l] is None or d[l] is None:
                continue
            if m > 0 and l % m == 0:
                # Error skip: E_{L->l}, source = e[L-1]
                dE = -gamma * jnp.einsum(
                    'bi,bj->ij', d[l], e[L - 1]
                ) / batch_size
            else:
                # Adjacent: E_{(l+1)->l}, source = e[l+1]
                dE = -gamma * jnp.einsum(
                    'bi,bj->ij', d[l], e[l + 1]
                ) / batch_size
            delta_E[l] = dE

        return delta_W, delta_E

    def compute_grad_E_updates(self, bundle, z, h, y_batch, beta):
        """Compute gradient-based E updates (Variant 2 / rLRA-dx).

        Computes dD/dE where D = sum_l ||e_l||^2 using JAX autodiff.
        Unlike the Hebbian rule (Eq. 6), this includes activation function
        derivatives, making it mathematically precise but biologically
        implausible. See paper appendix: "rLRA, dx" variant.

        W updates remain Hebbian — only E updates change.
        """
        E_matrices = bundle["E"]
        m = bundle["error_skip_every"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]

        # Collect non-None E matrices and their indices
        active_indices = [l for l in range(L) if E_matrices[l] is not None]
        E_active = [E_matrices[l] for l in active_indices]

        if not E_active:
            return [None] * L

        def discrepancy_fn(E_active_list):
            # Reconstruct full E list from active subset
            E_full = [None] * L
            for idx, l in enumerate(active_indices):
                E_full[l] = E_active_list[idx]

            # Recompute backward sweep (mirrors compute_targets_and_errors)
            e = [None] * L
            e[L - 1] = z[L - 1] - y_batch

            for l in range(L - 2, -1, -1):
                if l == 0:
                    d_l = jnp.zeros_like(z[l])
                elif E_full[l] is not None:
                    if m > 0 and l % m == 0:
                        d_l = e[L - 1] @ E_full[l].T
                    else:
                        d_l = e[l + 1] @ E_full[l].T
                else:
                    d_l = jnp.zeros_like(z[l])

                if l < L - 1:
                    y_l = vmap(act_fn)(h[l] - beta * d_l)
                else:
                    y_l = h[l] - beta * d_l

                e[l] = z[l] - y_l

            # Total discrepancy D = sum_l ||e_l||^2 averaged over batch
            D = sum(
                jnp.mean(jnp.sum(e[l] ** 2, axis=1))
                for l in range(L) if e[l] is not None
            )
            return D

        grad_E_active = jax.grad(discrepancy_fn)(E_active)

        delta_E = [None] * L
        for idx, l in enumerate(active_indices):
            delta_E[l] = grad_E_active[idx]

        return delta_E

    def apply_hebbian_updates(self, bundle, delta_W, delta_E, lr_W, lr_E):
        """Apply Hebbian updates to W and E matrices.

        For Variant 2 (future), this method would be overridden to use
        gradient-based E updates instead.
        """
        model = list(bundle["model"])
        E_matrices = list(bundle["E"])

        # Update W weights inside each Sequential layer
        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]  # Sequential([Lambda, Linear])
            new_weight = linear.weight - lr_W * delta_W[l]
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(
                lambda seq: seq.layers[1], layer, new_linear
            )

        # Update E matrices
        for l in range(len(E_matrices)):
            if delta_E[l] is not None and E_matrices[l] is not None:
                E_matrices[l] = E_matrices[l] - lr_E * delta_E[l]

        return {**bundle, "model": model, "E": E_matrices}

    def apply_optax_updates(self, bundle, delta_W, delta_E, w_optim,
                            w_opt_state, e_optim, e_opt_state):
        """Apply updates using optax optimizers (treats deltas as gradients).

        Returns updated bundle, w_opt_state, e_opt_state.
        """
        model = list(bundle["model"])
        E_matrices = list(bundle["E"])

        # Update W
        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]
            updates, w_opt_state[l] = w_optim.update(
                delta_W[l], w_opt_state[l], linear.weight
            )
            new_weight = linear.weight + updates  # optax negates internally
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(
                lambda seq: seq.layers[1], layer, new_linear
            )

        # Update E
        for l in range(len(E_matrices)):
            if delta_E[l] is not None and E_matrices[l] is not None:
                updates, e_opt_state[l] = e_optim.update(
                    delta_E[l], e_opt_state[l], E_matrices[l]
                )
                E_matrices[l] = E_matrices[l] + updates

        return {**bundle, "model": model, "E": E_matrices}, w_opt_state, e_opt_state

    # --- PCVariant protocol methods ---

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
        return {
            "model": result["model"],
            "skip_model": result.get("skip_model"),
        }

    def evaluate(self, bundle, test_loader):
        avg_test_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            z, _ = self.forward_pass(bundle, img_batch)
            preds = z[-1]
            acc = float(jnp.mean(
                jnp.argmax(preds, axis=1) == jnp.argmax(label_batch, axis=1)
            ) * 100)
            avg_test_acc += acc
        return avg_test_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        return get_weight_list(bundle["model"])

    def init_w_optim_states(self, bundle, w_optim):
        """Return per-layer Adam states for W matrices (MLP Sequential structure)."""
        states = []
        for layer in bundle["model"]:
            linear = layer.layers[1]   # Sequential([Lambda(act_fn), Linear])
            states.append(w_optim.init(linear.weight))
        return states

    def get_E_arrays(self, bundle):
        """Return list of non-None E matrices for tracking."""
        return [e for e in bundle["E"] if e is not None]

    def compute_condition_number(self, bundle, x, y):
        # Not applicable for rec-LRA (no PC energy-based inference)
        return float('nan'), jnp.array([])
