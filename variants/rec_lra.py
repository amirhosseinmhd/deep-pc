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
import jpc
import equinox as eqx
import equinox.nn as nn

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list


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

        # Create E matrices for error transmission.
        # E[l] computes d[l] in the backward sweep (l from L-2 down to 1).
        # E[0] = None (no error to input proxy layer).
        # E matrices exist for l=1..L-2 so all hidden layers receive error.
        # Hebbian E UPDATES are only for l > 1 (paper guard), but E[1] still
        # exists as a connection so layer 1 gets a nonzero error signal.
        E = [None] * depth
        for l in range(1, depth - 1):  # l from 1 to L-2
            key, subkey = jr.split(key)
            if error_skip_every > 0 and l % error_skip_every == 0:
                # Error skip from output: shape (dims[l], dims[L-1])
                fan_in = dims[-1]
                E[l] = jr.normal(subkey, (dims[l], fan_in)) / jnp.sqrt(fan_in)
            else:
                # Adjacent error: shape (dims[l], dims[l+1])
                fan_in = dims[l + 1]
                E[l] = jr.normal(subkey, (dims[l], fan_in)) / jnp.sqrt(fan_in)

        return {
            "model": model,
            "E": E,
            "forward_skip_every": forward_skip_every,
            "error_skip_every": error_skip_every,
            "act_fn": act_fn_callable,
            "depth": depth,
        }

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

        e = [None] * L
        d = [None] * L

        # Output error
        e[L - 1] = z[L - 1] - y_batch

        # Backward sweep: compute targets and errors for hidden layers
        for l in range(L - 2, -1, -1):
            if l == 0:
                # No error connection to first layer
                d[l] = jnp.zeros_like(z[l])
            elif E_matrices[l] is not None:
                if m > 0 and l % m == 0:
                    # Error skip from output: d_l = e_L @ E_{L->l}^T
                    d[l] = e[L - 1] @ E_matrices[l].T
                else:
                    # Error from adjacent next layer: d_l = e_{l+1} @ E_{l+1->l}^T
                    d[l] = e[l + 1] @ E_matrices[l].T
            else:
                d[l] = jnp.zeros_like(z[l])

            # Target: y_l = phi(h_l - beta * d_l)
            if l < L - 1:
                y_l = vmap(act_fn)(h[l] - beta * d[l])
            else:
                y_l = h[l] - beta * d[l]

            # Error: e_l = z_l - y_l
            e[l] = z[l] - y_l

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

    def get_E_arrays(self, bundle):
        """Return list of non-None E matrices for tracking."""
        return [e for e in bundle["E"] if e is not None]

    def compute_condition_number(self, bundle, x, y):
        # Not applicable for rec-LRA (no PC energy-based inference)
        return float('nan'), jnp.array([])
