#!/usr/bin/env python
"""Pinpoint what causes the ~100x exponential decay per layer.

For PC energy: E_l = 0.5 * ||z_l - f_l(z_{l-1})||^2
Gradient w.r.t z_{l-1}: dE_l/dz_{l-1} = -(z_l - f_l(z_{l-1})) * J_l
where J_l = df_l/dz_{l-1}

At ffwd init, the error (z_l - f_l(z_{l-1})) = 0, so dE_l/dz_{l-1} = 0.
But after the output error perturbs z_{L-1}, a chain reaction starts.
The question: what determines the ~100x decay per hop?

The update to z_{l-1} from E_l is:
  delta_z_{l-1} = -lr * J_l^T * error_l

So the decay factor per layer is: lr * ||J_l^T|| * (error amplification)

We measure:
1. Weight spectral norms (||W_l||)
2. ReLU sparsity (fraction of dead neurons)
3. Jacobian norms (||J_l||) - the actual df_l/dz_{l-1}
4. Combined: lr * ||J_l|| to see effective signal transfer
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jacrev
import jpc
import optax

from common.data import set_seed, get_mnist_loaders
from variants import get_variant

DEPTH = 10
WIDTH = 128
ACT_FN = "relu"
SEED = 42
ACTIVITY_LR = 1e-2
BATCH_SIZE = 128


def diagnose():
    set_seed(SEED)
    key = jr.PRNGKey(SEED)
    variant = get_variant("resnet")
    model = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn=ACT_FN)

    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    activities, _, effective_model = variant.init_activities(model, img_batch)
    params = variant.get_params_for_jpc(effective_model)
    model_layers, skip_model = params

    # =========================================================
    # 1. Weight spectral norms
    # =========================================================
    print("=" * 60)
    print("1. WEIGHT SPECTRAL NORMS (||W_l||_2)")
    print("=" * 60)
    for i, layer in enumerate(model_layers):
        # layer is Sequential([Lambda(act_fn), Linear])
        linear = layer.layers[1]
        W = linear.weight
        sv = jnp.linalg.svdvals(W)
        print(f"  Layer {i}: shape={W.shape}, "
              f"spectral_norm={float(sv[0]):.4f}, "
              f"frobenius={float(jnp.linalg.norm(W)):.4f}, "
              f"min_sv={float(sv[-1]):.6f}")

    # =========================================================
    # 2. ReLU sparsity per layer (fraction of zeros after relu)
    # =========================================================
    print(f"\n{'='*60}")
    print("2. RELU SPARSITY (fraction of dead units in activities)")
    print("=" * 60)
    for i, a in enumerate(activities):
        frac_zero = float(jnp.mean(a == 0.0))
        frac_near_zero = float(jnp.mean(jnp.abs(a) < 1e-6))
        print(f"  Activity[{i}]: shape={a.shape}, "
              f"exact_zero={frac_zero:.3f}, "
              f"near_zero(<1e-6)={frac_near_zero:.3f}, "
              f"mean={float(jnp.mean(a)):.4e}")

    # =========================================================
    # 3. Per-layer Jacobian norm: ||df_l/dz_{l-1}||
    # f_l(z) = W_l * relu(z) + z  (for resnet with skip)
    # J_l = W_l * diag(relu'(z)) + I
    # =========================================================
    print(f"\n{'='*60}")
    print("3. JACOBIAN NORMS ||df_l/dz_{l-1}|| (single sample)")
    print("=" * 60)
    # Use first sample for Jacobian computation
    x_single = img_batch[0]

    # Layer 0: f_0(x) = W_0 * x (no activation, no skip for first layer)
    def f0(inp):
        return model_layers[0](inp)
    J0 = jacrev(f0)(x_single)
    sv0 = jnp.linalg.svdvals(J0)
    print(f"  Layer 0 (input->hidden): Jacobian spectral norm = {float(sv0[0]):.6f}, "
          f"shape={J0.shape}")

    # Hidden layers: f_l(z) = model_l(z) + skip_l(z)
    for l in range(1, len(model_layers) - 1):
        z_prev = activities[l-1][0]  # single sample

        def fl(z, _l=l):
            out = model_layers[_l](z)
            if skip_model is not None and skip_model[_l] is not None:
                out = out + skip_model[_l](z)
            return out

        Jl = jacrev(fl)(z_prev)
        svl = jnp.linalg.svdvals(Jl)
        print(f"  Layer {l}: Jacobian spectral norm = {float(svl[0]):.6f}, "
              f"min_sv = {float(svl[-1]):.6f}, "
              f"cond = {float(svl[0]/jnp.maximum(svl[-1], 1e-30)):.2f}")

    # Output layer
    z_last = activities[-2][0]
    def fL(z):
        return model_layers[-1](z)
    JL = jacrev(fL)(z_last)
    svL = jnp.linalg.svdvals(JL)
    print(f"  Layer {len(model_layers)-1} (output): Jacobian spectral norm = {float(svL[0]):.6f}")

    # =========================================================
    # 4. Decompose: for resnet, J_l = W_l * diag(relu'(z)) + I
    # The relu mask kills some directions, W scales others.
    # What's ||W_l * diag(mask)||?
    # =========================================================
    print(f"\n{'='*60}")
    print("4. DECOMPOSITION: W_l * relu_mask vs skip (Identity)")
    print("   J_l = W_l * diag(relu'(z)) + I  for resnet hidden layers")
    print("=" * 60)
    for l in range(1, len(model_layers) - 1):
        z_prev = activities[l-1][0]
        linear = model_layers[l].layers[1]
        W = linear.weight

        # ReLU mask: which units in z_prev are > 0
        relu_mask = (z_prev > 0).astype(jnp.float32)
        n_active = int(jnp.sum(relu_mask))
        n_total = len(relu_mask)

        # W * diag(mask) — the "main path" Jacobian contribution
        W_masked = W * relu_mask[None, :]  # broadcast mask over rows
        sv_masked = jnp.linalg.svdvals(W_masked)

        # J_l = W_masked + I (for resnet)
        J_full = W_masked + jnp.eye(W.shape[0], W.shape[1])
        sv_full = jnp.linalg.svdvals(J_full)

        print(f"  Layer {l}: active_neurons={n_active}/{n_total} ({n_active/n_total:.1%}), "
              f"||W*mask||={float(sv_masked[0]):.4f}, "
              f"||W*mask+I||={float(sv_full[0]):.4f}, "
              f"min_sv(W*mask+I)={float(sv_full[-1]):.6f}")

    # =========================================================
    # 5. Effective signal transfer: lr * ||J_l^T|| * ||error_l||
    # After 1 inference step, error at layer L-1 (last hidden) is:
    #   error_{L-1} = z_{L-1} - f_{L-1}(z_{L-2}) (still ~0 since only z_{L-2}
    #   updated slightly from output error)
    # The update chain is:
    #   delta_z_{L-2} = -lr * (dE_{L-1}/dz_{L-2} + dE_{L-2}/dz_{L-2})
    # Since E_{L-2} = 0 initially:
    #   delta_z_{L-2} = -lr * J_{L-1}^T * error_{L-1}
    # But error_{L-1} = 0! The only nonzero gradient comes from E_output on z_{L-2}.
    # Let's trace this step by step.
    # =========================================================
    print(f"\n{'='*60}")
    print("5. STEP-BY-STEP GRADIENT CHAIN (1 inference step)")
    print("   How much gradient does each activity get from the output error?")
    print("=" * 60)

    # dE_total/dz_l at ffwd init
    def energy_fn(acts):
        return jpc.pc_energy_fn(
            params=params, activities=acts,
            y=label_batch, x=img_batch,
            param_type="sp",
        )

    grads = jax.grad(energy_fn)(activities)
    for i, g in enumerate(grads):
        gnorm = float(jnp.mean(jnp.linalg.norm(g, axis=1)))
        gmax = float(jnp.max(jnp.abs(g)))
        print(f"  dE/dz[{i}]: mean_L2={gnorm:.6e}, max_abs={gmax:.6e}")

    # =========================================================
    # 6. Try without ReLU (identity activation) to isolate ReLU effect
    # =========================================================
    print(f"\n{'='*60}")
    print("6. COMPARISON: same depth with 'tanh' activation")
    print("=" * 60)
    model_tanh = variant.create_model(key, depth=DEPTH, width=WIDTH, act_fn="tanh")
    activities_tanh, _, eff_tanh = variant.init_activities(model_tanh, img_batch)
    params_tanh = variant.get_params_for_jpc(eff_tanh)

    def energy_fn_tanh(acts):
        return jpc.pc_energy_fn(
            params=params_tanh, activities=acts,
            y=label_batch, x=img_batch,
            param_type="sp",
        )
    grads_tanh = jax.grad(energy_fn_tanh)(activities_tanh)
    for i, g in enumerate(grads_tanh):
        gnorm = float(jnp.mean(jnp.linalg.norm(g, axis=1)))
        print(f"  dE/dz[{i}] (tanh): mean_L2={gnorm:.6e}")

    # Also check tanh activity sparsity
    print("\n  Tanh activity stats:")
    for i, a in enumerate(activities_tanh):
        print(f"    Activity[{i}]: mean={float(jnp.mean(a)):.4e}, "
              f"std={float(jnp.std(a)):.4e}, "
              f"near_zero(<1e-6)={float(jnp.mean(jnp.abs(a) < 1e-6)):.3f}")

    # =========================================================
    # 7. Check at smaller depth (5) for comparison
    # =========================================================
    print(f"\n{'='*60}")
    print("7. COMPARISON: depth=5 (where things work)")
    print("=" * 60)
    model_d5 = variant.create_model(key, depth=5, width=WIDTH, act_fn=ACT_FN)
    activities_d5, _, eff_d5 = variant.init_activities(model_d5, img_batch)
    params_d5 = variant.get_params_for_jpc(eff_d5)

    def energy_fn_d5(acts):
        return jpc.pc_energy_fn(
            params=params_d5, activities=acts,
            y=label_batch, x=img_batch,
            param_type="sp",
        )
    grads_d5 = jax.grad(energy_fn_d5)(activities_d5)
    for i, g in enumerate(grads_d5):
        gnorm = float(jnp.mean(jnp.linalg.norm(g, axis=1)))
        print(f"  dE/dz[{i}] (depth=5): mean_L2={gnorm:.6e}")


if __name__ == "__main__":
    diagnose()
