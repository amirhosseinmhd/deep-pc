"""Vanilla MLP variant — no skip connections."""

import jax
import jax.numpy as jnp
import jpc
import equinox as eqx

from config import INPUT_DIM as _DEFAULT_INPUT_DIM, OUTPUT_DIM as _DEFAULT_OUTPUT_DIM
from common.utils import get_weight_list
from common.hessian import unwrap_hessian_pytree


class BaselineVariant:
    """Vanilla PC network using jpc.make_mlp, no skip connections."""

    @property
    def name(self):
        return "Vanilla MLP"

    @property
    def has_batch_stats(self):
        return False

    def create_model(self, key, depth, width, act_fn, **kwargs):
        input_dim = kwargs.get("input_dim", _DEFAULT_INPUT_DIM)
        output_dim = kwargs.get("output_dim", _DEFAULT_OUTPUT_DIM)
        model = jpc.make_mlp(
            key=key, input_dim=input_dim, width=width, depth=depth,
            output_dim=output_dim, act_fn=act_fn, use_bias=False,
            param_type="sp",
        )
        return {"model": model, "skip_model": None}

    def init_activities(self, bundle, x_batch):
        activities = jpc.init_activities_with_ffwd(
            model=bundle["model"], input=x_batch,
            skip_model=bundle["skip_model"], param_type="sp",
        )
        return activities, None, bundle

    def get_params_for_jpc(self, bundle):
        return (bundle["model"], bundle["skip_model"])

    def get_param_type(self):
        return "sp"

    def get_optimizer_target(self, bundle):
        return (eqx.filter(bundle["model"], eqx.is_array), bundle["skip_model"])

    def post_learning_step(self, bundle, result, batch_stats):
        return {
            "model": result["model"],
            "skip_model": result["skip_model"],
        }

    def evaluate(self, bundle, test_loader):
        avg_test_acc = 0.0
        for _, (img_batch, label_batch) in enumerate(test_loader):
            img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
            _, test_acc = jpc.test_discriminative_pc(
                model=bundle["model"], output=label_batch, input=img_batch,
                skip_model=bundle["skip_model"], param_type="sp",
            )
            avg_test_acc += float(test_acc)
        return avg_test_acc / len(test_loader)

    def get_weight_arrays(self, bundle):
        return get_weight_list(bundle["model"])

    def compute_condition_number(self, bundle, x, y):
        activities = jpc.init_activities_with_ffwd(
            model=bundle["model"], input=x,
            skip_model=bundle["skip_model"], param_type="sp",
        )
        hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
            (bundle["model"], bundle["skip_model"]), activities, y,
            x=x, param_type="sp",
        )
        H = unwrap_hessian_pytree(hessian_pytree, activities)
        eigenvals = jnp.linalg.eigvalsh(H)
        lam_max = jnp.abs(eigenvals[-1])
        lam_min = jnp.abs(eigenvals[0])
        cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
        return cond, eigenvals
