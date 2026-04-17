"""res-error-net variant — deep iterative PC with residual error highways.

Extends standard predictive coding with learnable V_{L→i} matrices that carry
the output error e^L directly back to hidden layer i, analogous to ResNet
skip connections but on the backward/error pathway. Addresses the vanishing
error signal problem in deep PC networks.

Augmented free energy (from friend's derivation):
    F = Σ_ℓ (1/2)‖e^ℓ‖²  +  Σ_ℓ α (e^ℓ)ᵀ V_{L→ℓ} e^L

Inference dynamics (Euler on dF/dz):
    ż^i = -∂F/∂z^i   for i ∈ {0, …, L-2};  z^{L-1} hard-clamped to y

V update rules (config flag):
    "energy": ΔV_{L→i} = α e^i (e^L)ᵀ        (derived from ∂F/∂V)
    "state" : ΔV_{L→i} = -α z^i (e^L)ᵀ       (original Hebbian sketch, stored
                                              negated so optax subtraction
                                              yields +α z^i (e^L)ᵀ step)
W updates: standard PC gradient on F_pc (highway term ignored per friend's
note — "W update rule is unchanged").
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import equinox as eqx
import jpc

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list


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
        v_init_scale = kwargs.get("v_init_scale", 0.01)
        # "unit_gaussian" mirrors rec-LRA paper (p.16); "jpc_default" keeps
        # the 1/√fan_in scale which is stable without ZCA preprocessing.
        init_scheme = kwargs.get("res_init_scheme", "jpc_default")

        key, subkey = jr.split(key)
        model = jpc.make_mlp(
            key=subkey, input_dim=input_dim, width=width, depth=depth,
            output_dim=output_dim, act_fn=act_fn, use_bias=False,
            param_type="sp",
        )
        act_fn_callable = jpc.get_act_fn(act_fn)

        if init_scheme == "unit_gaussian":
            model = self._reinit_unit_gaussian(model, key)
            key, _ = jr.split(key)

        dims = [output_dim if (i + 1) == depth else width for i in range(depth)]
        output_idx = depth - 1

        # S = {output_idx - k, output_idx - 2k, …} ∩ [1, output_idx - 1]
        S = []
        for step in range(1, depth):
            i = output_idx - highway_every_k * step
            if 1 <= i <= output_idx - 1:
                S.append(i)
            else:
                break
        S.sort()

        V_list = []
        for i in S:
            key, sub = jr.split(key)
            V_list.append(v_init_scale * jr.normal(sub, (dims[i], dims[output_idx])))

        return {
            "model": model,
            "V_list": V_list,
            "highway_indices": S,
            "highway_every_k": highway_every_k,
            "v_init_scale": v_init_scale,
            "init_scheme": init_scheme,
            "act_fn": act_fn_callable,
            "act_fn_name": act_fn,
            "depth": depth,
            "dims": dims,
        }

    @staticmethod
    def _reinit_unit_gaussian(model, key):
        new_layers = list(model)
        for l in range(len(new_layers) - 1):
            seq = new_layers[l]
            linear = seq.layers[1]
            key, sub = jr.split(key)
            new_w = jr.normal(sub, linear.weight.shape)
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_w)
            new_layers[l] = eqx.tree_at(lambda s: s.layers[1], seq, new_linear)
        return new_layers

    # --- Forward / predictions / errors ---

    def forward_pass(self, bundle, x_batch):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]

        z = [None] * L
        h = [None] * L
        z_prev = x_batch
        for l in range(L):
            h[l] = vmap(model[l])(z_prev)
            z[l] = vmap(act_fn)(h[l]) if l < L - 1 else h[l]
            z_prev = z[l]
        return z, h

    @staticmethod
    def _predictions_and_errors(model, z_list, x_batch, act_fn, L):
        mu = [None] * L
        h = [None] * L
        for l in range(L):
            prev = x_batch if l == 0 else z_list[l - 1]
            h[l] = vmap(model[l])(prev)
            mu[l] = vmap(act_fn)(h[l]) if l < L - 1 else h[l]
        e = [z_list[l] - mu[l] for l in range(L)]
        return e, mu, h

    def compute_errors(self, bundle, z, x_batch, y_batch):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]
        z_list = list(z)
        z_list[L - 1] = y_batch
        e, _, _ = self._predictions_and_errors(model, z_list, x_batch, act_fn, L)
        return e

    # --- Energy ---

    @staticmethod
    def _F_pc(e_list):
        return sum(0.5 * jnp.mean(jnp.sum(el ** 2, axis=1)) for el in e_list)

    @staticmethod
    def _F_highway(e_list, V_list, S, alpha, L):
        if not S:
            return jnp.array(0.0)
        e_L = e_list[L - 1]
        total = jnp.array(0.0)
        for idx, i in enumerate(S):
            Vi = V_list[idx]
            e_i = e_list[i]
            total = total + alpha * jnp.mean(jnp.sum(e_i * (e_L @ Vi.T), axis=1))
        return total

    def free_energy_z(self, z_free, bundle, x_batch, y_batch, alpha):
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]
        S = bundle["highway_indices"]
        V_list = bundle["V_list"]

        z_list = list(z_free) + [y_batch]
        e, _, _ = self._predictions_and_errors(model, z_list, x_batch, act_fn, L)
        return self._F_pc(e) + self._F_highway(e, V_list, S, alpha, L)

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
        if z_init is None:
            z_init, _ = self.forward_pass(bundle, x_batch)
        z = list(z_init)
        z[L - 1] = y_batch

        energies = [] if record_energy else None
        for _ in range(T):
            z = self.inference_step(bundle, z, x_batch, y_batch, alpha, dt)
            if record_energy:
                z_free = [z[l] for l in range(L - 1)]
                energies.append(float(
                    self.free_energy_z(z_free, bundle, x_batch, y_batch, alpha)
                ))
        return z, energies

    # --- Parameter updates ---

    def compute_W_updates(self, bundle, z, x_batch, y_batch):
        """∂F_pc/∂W^l via autodiff. Treat as gradient: W ← W - lr·ΔW."""
        model = bundle["model"]
        act_fn = bundle["act_fn"]
        L = bundle["depth"]

        z_list = list(z)
        z_list[L - 1] = y_batch

        def pc_energy(model_):
            e, _, _ = self._predictions_and_errors(model_, z_list, x_batch, act_fn, L)
            return self._F_pc(e)

        grads_model = eqx.filter_grad(pc_energy)(model)

        delta_W = []
        for l in range(L):
            seq_grad = grads_model[l]
            linear_grad = seq_grad.layers[1]
            delta_W.append(linear_grad.weight)
        return delta_W

    def compute_V_updates(self, bundle, z, x_batch, y_batch, alpha, rule="energy"):
        """ΔV_{L→i}. Sign convention: treat as 'gradient', i.e. V ← V - lr·ΔV.

        energy: ΔV = +α e^i (e^L)ᵀ   (derived from ∂F_hw/∂V)
        state : ΔV = -α z^i (e^L)ᵀ   (so subtracting yields Hebbian growth)
        """
        S = bundle["highway_indices"]
        L = bundle["depth"]
        batch_size = x_batch.shape[0]

        e = self.compute_errors(bundle, z, x_batch, y_batch)
        e_L = e[L - 1]

        delta_V = []
        for i in S:
            if rule == "energy":
                dV = alpha * jnp.einsum("bi,bj->ij", e[i], e_L) / batch_size
            elif rule == "state":
                dV = -alpha * jnp.einsum("bi,bj->ij", z[i], e_L) / batch_size
            else:
                raise ValueError(f"Unknown v_update_rule: {rule!r}")
            delta_V.append(dV)
        return delta_V

    # --- Apply updates ---

    def apply_optax_updates(self, bundle, delta_W, delta_V,
                            w_optim, w_opt_state, v_optim, v_opt_state):
        model = list(bundle["model"])
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]

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

        return (
            {**bundle, "model": model, "V_list": V_list},
            w_opt_state, v_opt_state,
        )

    def apply_sgd_updates(self, bundle, delta_W, delta_V, lr_W, lr_V):
        model = list(bundle["model"])
        V_list = list(bundle["V_list"])
        S = bundle["highway_indices"]

        for l in range(len(model)):
            layer = model[l]
            linear = layer.layers[1]
            new_weight = linear.weight - lr_W * delta_W[l]
            new_linear = eqx.tree_at(lambda lin: lin.weight, linear, new_weight)
            model[l] = eqx.tree_at(lambda seq: seq.layers[1], layer, new_linear)

        for idx in range(len(S)):
            V_list[idx] = V_list[idx] - lr_V * delta_V[idx]

        return {**bundle, "model": model, "V_list": V_list}

    # --- Optimizer state init ---

    def init_w_optim_states(self, bundle, w_optim):
        states = []
        for layer in bundle["model"]:
            linear = layer.layers[1]
            states.append(w_optim.init(linear.weight))
        return states

    def init_v_optim_states(self, bundle, v_optim):
        return [v_optim.init(V) for V in bundle["V_list"]]

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
