"""MetricsCollector: accumulates all per-iteration metrics during training."""

import numpy as np
import jax.numpy as jnp


class MetricsCollector:
    """Accumulates all per-iteration metrics during a single training run."""

    def __init__(
        self,
        track_weight_updates=True,
        track_activity_norms=True,
        track_grad_norms=True,
        track_layer_energy=True,
    ):
        self.train_losses = []
        self.test_accs = []
        self.test_iters = []

        self.track_weight_updates = track_weight_updates
        self.weight_update_norms = [] if track_weight_updates else None

        self.track_activity_norms = track_activity_norms
        self.activity_norms_init = [] if track_activity_norms else None
        self.activity_norms_post = [] if track_activity_norms else None

        self.track_grad_norms = track_grad_norms
        self.grad_norms = [] if track_grad_norms else None

        self.track_layer_energy = track_layer_energy
        self.energy_per_layer = [] if track_layer_energy else None

    def record_train_loss(self, loss):
        self.train_losses.append(loss)

    def record_test(self, iter_num, accuracy):
        self.test_iters.append(iter_num)
        self.test_accs.append(accuracy)

    def record_activity_norms_pre(self, activities):
        norms = [
            float(jnp.linalg.norm(a, axis=1, ord=2).mean())
            for a in activities
        ]
        self.activity_norms_init.append(norms)

    def record_activity_norms_post(self, activities):
        norms = [
            float(jnp.linalg.norm(a, axis=1, ord=2).mean())
            for a in activities
        ]
        self.activity_norms_post.append(norms)

    def record_weight_update_norms(self, old_weights, new_weights):
        norms = [
            float(jnp.linalg.norm(jnp.ravel(w_new - w_old)))
            for w_old, w_new in zip(old_weights, new_weights)
        ]
        self.weight_update_norms.append(norms)

    def record_grad_norms(self, grads_model):
        """Extract per-layer gradient Frobenius norms.

        grads_model: the model-part of the grads (first element of params tuple).
        """
        from jax.tree_util import tree_leaves
        norms = []
        # grads_model is a list of layer grads (same structure as model layers)
        if isinstance(grads_model, (list, tuple)):
            for layer_grad in grads_model:
                layer_params = tree_leaves(layer_grad)
                weight_params = [
                    p for p in layer_params
                    if isinstance(p, jnp.ndarray) and p.ndim >= 2
                ]
                if weight_params:
                    layer_norm = sum(
                        float(jnp.linalg.norm(jnp.ravel(w)))
                        for w in weight_params
                    )
                    norms.append(layer_norm)
                else:
                    norms.append(0.0)
        self.grad_norms.append(norms)

    def record_layer_energy(self, layer_energies):
        """Record per-layer energy.

        jpc returns energies as [output, hidden_1, ..., hidden_{L-2}, input].
        We reorder to [input, hidden_1, ..., hidden_{L-2}, output].
        """
        energies = [float(e) for e in layer_energies]
        # Reorder: [output, hidden_1, ..., input] -> [input, hidden_1, ..., output]
        reordered = [energies[-1]] + energies[1:-1] + [energies[0]]
        self.energy_per_layer.append(reordered)

    def finalize(self):
        """Return all collected metrics as a dict of numpy arrays."""
        out = {
            "train_losses": np.array(self.train_losses),
            "test_accs": np.array(self.test_accs),
            "test_iters": np.array(self.test_iters),
        }
        if self.track_weight_updates and self.weight_update_norms:
            out["weight_update_norms"] = np.array(self.weight_update_norms)
        if self.track_activity_norms:
            if self.activity_norms_init:
                out["activity_norms_init"] = np.array(self.activity_norms_init)
            if self.activity_norms_post:
                out["activity_norms_post"] = np.array(self.activity_norms_post)
        if self.track_grad_norms and self.grad_norms:
            out["grad_norms"] = np.array(self.grad_norms)
        if self.track_layer_energy and self.energy_per_layer:
            out["energy_per_layer"] = np.array(self.energy_per_layer)
        return out
