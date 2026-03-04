"""Diagnose why deep (depth=20+) DyT networks may struggle.

Traces per-layer activation statistics through a single forward pass
for different depths, printing magnitudes, saturation, and gradient norms.
No training — just init-time diagnostics.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from common_dyt import (
    FCResNetDyT, InputLayer, HiddenLayerDyT, OutputLayer,
    INPUT_DIM, OUTPUT_DIM, WIDTH, SEED,
    set_seed, get_mnist_loaders,
)

set_seed(SEED)
train_loader, _ = get_mnist_loaders(64)
x_batch, y_batch = next(iter(train_loader))
x_batch, y_batch = x_batch.numpy(), y_batch.numpy()


def trace_forward(model, x_batch, label=""):
    """Run forward pass and print per-layer activation diagnostics."""
    h = x_batch
    print(f"\n{'='*75}")
    print(f" {label}  (depth={len(model)})")
    print(f"{'='*75}")
    print(f"{'Layer':>6} {'Type':>12} {'Mean |h|':>10} {'Std(h)':>10} "
          f"{'Max |h|':>10} {'% |tanh·α·h|>0.95':>20} {'α val':>8}")
    print("-" * 82)

    for i, layer in enumerate(model.layers):
        h = vmap(layer)(h)

        mean_abs = float(jnp.mean(jnp.abs(h)))
        std_h = float(jnp.std(h))
        max_abs = float(jnp.max(jnp.abs(h)))

        # Check tanh saturation for layers with DyT
        sat_pct = ""
        alpha_val = ""
        if isinstance(layer, (HiddenLayerDyT, OutputLayer)):
            alpha = float(layer.dyt.alpha[0])
            alpha_val = f"{alpha:.4f}"
            # Compute what fraction of pre-tanh values are in saturation zone
            # We need to reconstruct pre-tanh input; approximate from output
            # Actually let's compute it directly
            # For HiddenLayerDyT: DyT input is linear(act(x_prev)) + x_prev
            # The tanh argument is alpha * (that input)
            # After DyT: output = gamma * tanh(alpha * z) + beta
            # If gamma≈1, beta≈0, then |output| close to 1 means saturated
            # More precisely, check |h - beta| / gamma ≈ |tanh(alpha*z)|
            gamma = layer.dyt.gamma
            beta = layer.dyt.beta
            # h = gamma * tanh(alpha * z) + beta
            # tanh_val = (h - beta) / gamma
            # Only compute saturation if shapes match (skip OutputLayer which changes dim)
            if h.shape[-1] == gamma.shape[0]:
                tanh_val = (h - beta[None, :]) / jnp.clip(gamma[None, :], 1e-8, None)
                frac_saturated = float(jnp.mean(jnp.abs(tanh_val) > 0.95) * 100)
            else:
                frac_saturated = -1.0  # shape mismatch (output layer)
            sat_pct = f"{frac_saturated:.1f}%"

        ltype = type(layer).__name__[:12]
        print(f"{i:>6} {ltype:>12} {mean_abs:>10.4f} {std_h:>10.4f} "
              f"{max_abs:>10.4f} {sat_pct:>20} {alpha_val:>8}")

    return h


def trace_gradients(model, x_batch, y_batch, label=""):
    """Check gradient magnitudes flowing back through the network."""
    from jax import grad

    def loss_fn(layers_list, x, y):
        def single_forward(layers, xi):
            h = xi
            for layer in layers:
                h = layer(h)
            return h
        preds = vmap(single_forward, in_axes=(None, 0))(layers_list, x)
        return jnp.mean((preds - y) ** 2)

    grads = grad(loss_fn)(model.layers, x_batch, y_batch)

    print(f"\n  Gradient norms per layer ({label}):")
    print(f"  {'Layer':>6} {'Type':>12} {'Linear W grad':>15} {'DyT α grad':>12} "
          f"{'DyT γ grad':>12} {'DyT β grad':>12}")
    print("  " + "-" * 72)

    for i, (layer, g) in enumerate(zip(model.layers, grads)):
        ltype = type(layer).__name__[:12]
        w_grad = ""
        a_grad = ""
        g_grad = ""
        b_grad = ""

        if isinstance(layer, InputLayer):
            w_grad = f"{float(jnp.linalg.norm(g.linear.weight)):.6f}"
        elif isinstance(layer, (HiddenLayerDyT, OutputLayer)):
            w_grad = f"{float(jnp.linalg.norm(g.linear.weight)):.6f}"
            a_grad = f"{float(jnp.linalg.norm(g.dyt.alpha)):.6f}"
            g_grad = f"{float(jnp.linalg.norm(g.dyt.gamma)):.6f}"
            b_grad = f"{float(jnp.linalg.norm(g.dyt.beta)):.6f}"

        print(f"  {i:>6} {ltype:>12} {w_grad:>15} {a_grad:>12} "
              f"{g_grad:>12} {b_grad:>12}")


# ============================================================================
# Run diagnostics for each depth
# ============================================================================
print("\n" + "#"*75)
print("# PART 1: Forward-pass activation statistics at init")
print("#"*75)

for depth in [5, 10, 20, 40]:
    key = jr.PRNGKey(SEED)
    model = FCResNetDyT(
        key=key, in_dim=INPUT_DIM, width=WIDTH, depth=depth,
        out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5
    )
    trace_forward(model, x_batch, label=f"relu depth={depth}")

print("\n\n" + "#"*75)
print("# PART 2: Gradient norms at init")
print("#"*75)

for depth in [5, 10, 20, 40]:
    key = jr.PRNGKey(SEED)
    model = FCResNetDyT(
        key=key, in_dim=INPUT_DIM, width=WIDTH, depth=depth,
        out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5
    )
    trace_gradients(model, x_batch, y_batch, label=f"relu depth={depth}")


# ============================================================================
# PART 3: Compare different init_alpha values at depth=20
# ============================================================================
print("\n\n" + "#"*75)
print("# PART 3: Effect of init_alpha on depth=20 relu")
print("#"*75)

for alpha in [0.1, 0.25, 0.5, 1.0, 2.0]:
    key = jr.PRNGKey(SEED)
    model = FCResNetDyT(
        key=key, in_dim=INPUT_DIM, width=WIDTH, depth=20,
        out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=alpha
    )
    trace_forward(model, x_batch, label=f"relu d=20, α₀={alpha}")


# ============================================================================
# PART 4: Compare DyT at every layer vs every-other-layer at depth=20
# ============================================================================
print("\n\n" + "#"*75)
print("# PART 4: DyT at every layer vs every-other-layer (depth=20)")
print("#"*75)

# Every layer (default)
key = jr.PRNGKey(SEED)
model_all = FCResNetDyT(
    key=key, in_dim=INPUT_DIM, width=WIDTH, depth=20,
    out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5
)
trace_forward(model_all, x_batch, label="DyT at ALL layers")

# Every other layer
key = jr.PRNGKey(SEED)
model_half = FCResNetDyT(
    key=key, in_dim=INPUT_DIM, width=WIDTH, depth=20,
    out_dim=OUTPUT_DIM, act_fn="relu", init_alpha=0.5,
    dyt_enabled_layers=[i for i in range(20) if i % 2 == 1],
)
trace_forward(model_half, x_batch, label="DyT at ODD layers only")
