"""Abstract base protocol for PC network variants.

Each variant encapsulates:
  - Model creation
  - Activity initialization (forward pass)
  - How to pass params to jpc functions
  - Post-learning step (EMA, model reconstruction)
  - Evaluation
  - Weight extraction for tracking
  - Condition number computation
"""

from typing import Protocol, Any, Optional, Tuple, List
import jax.numpy as jnp


class PCVariant(Protocol):
    """Protocol that every architecture variant must implement."""

    @property
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    def has_batch_stats(self) -> bool:
        """Whether this variant needs batch-stats freezing."""
        ...

    def create_model(self, key, depth, width, act_fn, **kwargs) -> Any:
        """Construct the model."""
        ...

    def init_activities(self, model, x_batch):
        """Initialize activities via forward pass.

        Returns:
            activities: list of activity arrays
            batch_stats: per-layer batch statistics (None if not applicable)
            effective_model: model to use for inference (frozen for BF)
        """
        ...

    def get_params_for_jpc(self, model) -> tuple:
        """Return (model_or_layers, skip_model_or_None) for jpc calls."""
        ...

    def get_param_type(self) -> str:
        """Return the param_type string for jpc calls."""
        ...

    def get_optimizer_target(self, model) -> tuple:
        """Return the pytree to init the param optimizer on."""
        ...

    def post_learning_step(self, model, result, batch_stats) -> Any:
        """Reconstruct model after jpc.update_pc_params + any post-update work."""
        ...

    def evaluate(self, model, test_loader) -> float:
        """Evaluate test accuracy."""
        ...

    def get_weight_arrays(self, model) -> list:
        """Extract flat list of weight matrices for tracking."""
        ...

    def compute_condition_number(self, model, x, y):
        """Compute kappa(H_z). Returns (cond, eigenvalues)."""
        ...
