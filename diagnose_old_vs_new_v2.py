#!/usr/bin/env python
"""
OLD dyt_v2 vs NEW dyt_v2: same math, different behavior.

OLD: layer(x) = DyT(W·relu(x)) + x,  skip_model = None
     init: manual forward pass
     jpc sees: params = (layers, None)

NEW: layer(x) = DyT(W·relu(x)),       skip_model = [Identity, ...]
     init: jpc.init_activities_with_ffwd with skip_model
     jpc sees: params = (layers, skip_model)

Both compute z_l = DyT(W·relu(z_{l-1})) + z_{l-1}. Mathematically identical.
We show exactly where and why they diverge.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import jpc
import equinox as eqx
import equinox.nn as nn
import optax

from typing import List, Callable
from common.data import set_seed, get_mnist_loaders
from config import INPUT_DIM, OUTPUT_DIM
from variants.dyt import DyTLayer, InputLayer

DEPTH = 50
WIDTH = 128
SEED = 42
BATCH_SIZE = 128


# ============================================================================
# Reconstruct the OLD hidden layer (skip INSIDE)
# ============================================================================
class OldHiddenLayer(eqx.Module):
    """OLD v2: act_fn -> DyT(linear(x)) + x (skip inside)."""
    act_fn: Callable = eqx.field(static=True)
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.dyt(self.linear(h)) + x   # <-- skip INSIDE
        return h


# Reconstruct the OLD output layer (had DyT)
class OldOutputLayer(eqx.Module):
    act_fn: Callable = eqx.field(static=True)
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.dyt(h)
        return self.linear(h)


# ============================================================================
# NEW hidden layer (no skip — jpc handles it)
# ============================================================================
class NewHiddenLayer(eqx.Module):
    """NEW v2: act_fn -> DyT(linear(x)) (no skip)."""
    act_fn: Callable = eqx.field(static=True)
    dyt: DyTLayer
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        h = self.dyt(self.linear(h))        # <-- no skip
        return h


class NewOutputLayer(eqx.Module):
    act_fn: Callable = eqx.field(static=True)
    linear: nn.Linear

    def __call__(self, x):
        h = self.act_fn(x)
        return self.linear(h)


def build_models(key):
    """Build OLD and NEW models with identical weights."""
    act = jpc.get_act_fn("relu")
    keys = jr.split(key, DEPTH)

    # Shared input layer
    input_layer = InputLayer(
        linear=nn.Linear(INPUT_DIM, WIDTH, use_bias=False, key=keys[0])
    )

    # Build OLD layers (skip inside)
    old_layers = [input_layer]
    for i in range(1, DEPTH - 1):
        old_layers.append(OldHiddenLayer(
            act_fn=act,
            dyt=DyTLayer(WIDTH, init_alpha=0.5),
            linear=nn.Linear(WIDTH, WIDTH, use_bias=False, key=keys[i]),
        ))
    old_layers.append(OldOutputLayer(
        act_fn=act,
        dyt=DyTLayer(WIDTH, init_alpha=0.5),
        linear=nn.Linear(WIDTH, OUTPUT_DIM, use_bias=False, key=keys[-1]),
    ))

    # Build NEW layers (no skip) with SAME weights
    new_layers = [input_layer]
    for i in range(1, DEPTH - 1):
        new_layers.append(NewHiddenLayer(
            act_fn=act,
            dyt=DyTLayer(WIDTH, init_alpha=0.5),
            linear=nn.Linear(WIDTH, WIDTH, use_bias=False, key=keys[i]),
        ))
    new_layers.append(NewOutputLayer(
        act_fn=act,
        linear=nn.Linear(WIDTH, OUTPUT_DIM, use_bias=False, key=keys[-1]),
    ))

    skip_model = jpc.make_skip_model(DEPTH)
    return old_layers, new_layers, skip_model


def run():
    set_seed(SEED)
    train_loader, _ = get_mnist_loaders(BATCH_SIZE)
    img_batch, label_batch = next(iter(train_loader))
    img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

    key = jr.PRNGKey(SEED)
    old_layers, new_layers, skip_model = build_models(key)

    # Verify weights are identical
    for i in range(DEPTH):
        if hasattr(old_layers[i], 'linear') and hasattr(new_layers[i], 'linear'):
            diff = float(jnp.max(jnp.abs(
                old_layers[i].linear.weight - new_layers[i].linear.weight
            )))
            assert diff == 0.0, f"Layer {i} weights differ!"
    print("Weights verified: identical between OLD and NEW.\n")

    # ==================================================================
    # PART 1: INIT — How activities are computed
    # ==================================================================
    print("=" * 70)
    print("PART 1: INITIALIZATION")
    print("Both compute z_l = DyT(W·relu(z_{l-1})) + z_{l-1}")
    print("=" * 70)

    # OLD init: manual forward pass (old code from git)
    old_activities = []
    h = img_batch
    for layer in old_layers:
        h = vmap(layer)(h)
        old_activities.append(h)

    # NEW init: jpc's init with skip_model
    new_activities = jpc.init_activities_with_ffwd(
        model=new_layers, input=img_batch,
        skip_model=skip_model, param_type="sp",
    )

    print("\n  Activity comparison (max |OLD - NEW| per layer):")
    for i in [0, 1, 2, 47, 48, 49]:
        diff = float(jnp.max(jnp.abs(old_activities[i] - new_activities[i])))
        print(f"    Layer {i}: {diff:.6e}")

    print("\n  These should be VERY close (same math, just float32 ordering diffs)")

    # ==================================================================
    # PART 2: What jpc sees — THE KEY DIFFERENCE
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PART 2: HOW jpc COMPUTES PREDICTION ERRORS")
    print("This is WHERE the two versions diverge.")
    print("=" * 70)

    old_params = (old_layers, None)         # no skip_model
    new_params = (new_layers, skip_model)   # with skip_model

    # Show what jpc does at ONE hidden layer (e.g., layer 25)
    L = 25
    print(f"\n  === At hidden layer {L} ===")
    print(f"\n  OLD version (skip inside layer, no skip_model):")
    print(f"    Init computed:  z_{L} = vmap(old_layer_{L})(z_{L-1})")
    print(f"                         = DyT(W·relu(z_{L-1})) + z_{L-1}")
    print(f"    Energy computes: err = z_{L} - vmap(old_layer_{L})(z_{L-1})")
    print(f"                        = z_{L} - [DyT(W·relu(z_{L-1})) + z_{L-1}]")
    print(f"    Same function, same input → same bits → err = 0 EXACTLY")

    # Actually compute it
    recomputed_old = vmap(old_layers[L])(old_activities[L - 1])
    err_old = old_activities[L] - recomputed_old
    print(f"\n    Actual: max|err| = {float(jnp.max(jnp.abs(err_old))):.6e}")

    print(f"\n  NEW version (no skip in layer, skip_model = Identity):")
    print(f"    Init computed (jpc._init.py lines 72-77):")
    print(f"      A = 1.0 * vmap(new_layer_{L})(z_{L-1})     = DyT(W·relu(z_{L-1}))")
    print(f"      B = vmap(skip[{L}])(z_{L-1})                = z_{L-1}")
    print(f"      z_{L} = A + B                               ← float32 addition, ROUNDS")
    print(f"    Energy computes (jpc._energies.py lines 106-108):")
    print(f"      err  = z_{L} - 1.0 * vmap(new_layer_{L})(z_{L-1})   = (A⊕B) - A")
    print(f"      err -= vmap(skip[{L}])(z_{L-1})                      = (A⊕B) - A - B")
    print(f"    Three operations instead of one → float32 rounding doesn't cancel")

    # Actually compute it step by step
    A = vmap(new_layers[L])(new_activities[L - 1])
    B = vmap(skip_model[L])(new_activities[L - 1])
    z_stored = new_activities[L]  # = A ⊕ B (from init)

    step1 = z_stored - A          # (A ⊕ B) ⊖ A
    err_new = step1 - B           # (A ⊕ B) ⊖ A ⊖ B

    print(f"\n    Actual step by step:")
    print(f"      A (model output):  mean|A| = {float(jnp.mean(jnp.abs(A))):.6e}")
    print(f"      B (skip output):   mean|B| = {float(jnp.mean(jnp.abs(B))):.6e}")
    print(f"      z_stored = A ⊕ B")
    print(f"      step1 = z_stored - A:  max|step1 - B| = {float(jnp.max(jnp.abs(step1 - B))):.6e}")
    print(f"      err = step1 - B:       max|err| = {float(jnp.max(jnp.abs(err_new))):.6e}")
    print(f"      err should be 0 mathematically, but is {float(jnp.max(jnp.abs(err_new))):.6e}")

    # Use jpc's actual energy function to verify
    old_energies = jpc.pc_energy_fn(
        params=old_params, activities=old_activities,
        y=label_batch, x=img_batch, param_type="sp",
        record_layers=True,
    )
    new_energies = jpc.pc_energy_fn(
        params=new_params, activities=new_activities,
        y=label_batch, x=img_batch, param_type="sp",
        record_layers=True,
    )

    print(f"\n  === Per-layer energy from jpc (all layers) ===")
    print(f"  {'Layer':<10} {'OLD energy':<20} {'NEW energy':<20}")
    print(f"  {'-'*50}")
    # energies[0] = output, energies[1..N-2] = hidden, energies[-1] = input
    for i in range(len(old_energies)):
        e_old = float(old_energies[i])
        e_new = float(new_energies[i])
        if i == 0:
            label = "output"
        elif i == len(old_energies) - 1:
            label = "input"
        else:
            label = f"hidden {i}"
        # Only print selected
        if i <= 5 or i >= len(old_energies) - 3 or i == L:
            print(f"  {label:<10} {e_old:<20.6e} {e_new:<20.6e}")
        elif i == 6:
            print(f"  {'...':<10}")

    old_zeros = sum(1 for i in range(1, len(old_energies)-1) if float(old_energies[i]) == 0.0)
    new_zeros = sum(1 for i in range(1, len(new_energies)-1) if float(new_energies[i]) == 0.0)
    print(f"\n  Hidden layers with EXACTLY 0.0 energy:")
    print(f"    OLD: {old_zeros} / {len(old_energies) - 2}")
    print(f"    NEW: {new_zeros} / {len(new_energies) - 2}")

    # ==================================================================
    # PART 3: Activity gradients at step 0
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PART 3: ACTIVITY GRADIENTS AT STEP 0")
    print("=" * 70)

    def energy_old(acts):
        return jpc.pc_energy_fn(params=old_params, activities=acts,
                                 y=label_batch, x=img_batch, param_type="sp")
    def energy_new(acts):
        return jpc.pc_energy_fn(params=new_params, activities=acts,
                                 y=label_batch, x=img_batch, param_type="sp")

    grads_old = jax.grad(energy_old)(old_activities)
    grads_new = jax.grad(energy_new)(new_activities)

    print(f"\n  {'Layer':<8} {'OLD mean|grad|':<22} {'NEW mean|grad|':<22}")
    print(f"  {'-'*52}")
    for i in [0, 1, 2, 10, 25, 40, 45, 47, 48, 49]:
        if i < len(grads_old):
            g_old = float(jnp.mean(jnp.abs(grads_old[i])))
            g_new = float(jnp.mean(jnp.abs(grads_new[i])))
            print(f"  {i:<8} {g_old:<22.6e} {g_new:<22.6e}")

    # ==================================================================
    # PART 4: Run inference and compare
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PART 4: INFERENCE (500 steps)")
    print("=" * 70)

    N_STEPS = 500
    ACTIVITY_LR = 0.5

    results = {}
    for name, params, activities in [
        ("OLD (skip inside)", old_params, [jnp.array(a) for a in old_activities]),
        ("NEW (skip outside)", new_params, [jnp.array(a) for a in new_activities]),
    ]:
        optim = optax.sgd(ACTIVITY_LR)
        opt_state = optim.init(activities)

        first_nonzero = [None] * len(activities)

        for t in range(N_STEPS):
            old_acts = [jnp.array(a) for a in activities]
            result = jpc.update_pc_activities(
                params=params, activities=activities,
                optim=optim, opt_state=opt_state,
                output=label_batch, input=img_batch,
                param_type="sp",
            )
            activities = result["activities"]
            opt_state = result["opt_state"]

            for i in range(len(activities)):
                delta = float(jnp.max(jnp.abs(activities[i] - old_acts[i])))
                if first_nonzero[i] is None and delta > 1e-20:
                    first_nonzero[i] = t

        # Weight gradients after inference
        grads = jpc.compute_pc_param_grads(
            params=params, activities=activities,
            y=label_batch, x=img_batch, param_type="sp",
        )
        model_grads = grads[0]
        grad_norms = []
        for i in range(len(model_grads)):
            g = model_grads[i]
            gnorm = 0.0
            if hasattr(g, 'linear') and hasattr(g.linear, 'weight') and g.linear.weight is not None:
                gnorm = float(jnp.linalg.norm(g.linear.weight))
            grad_norms.append(gnorm)

        results[name] = (first_nonzero, grad_norms)

    fn_old, gn_old = results["OLD (skip inside)"]
    fn_new, gn_new = results["NEW (skip outside)"]

    print(f"\n  Wavefront: first inference step with any activity change")
    print(f"  {'Layer':<8} {'OLD first step':<16} {'NEW first step':<16}")
    print(f"  {'-'*40}")
    for i in [0, 1, 10, 20, 30, 40, 44, 47, 48, 49]:
        if i < len(fn_old):
            s_old = str(fn_old[i]) if fn_old[i] is not None else "NEVER"
            s_new = str(fn_new[i]) if fn_new[i] is not None else "NEVER"
            print(f"  {i:<8} {s_old:<16} {s_new:<16}")

    print(f"\n  Weight gradient norms AFTER {N_STEPS} inference steps:")
    print(f"  {'Layer':<8} {'OLD ||dE/dW||':<22} {'NEW ||dE/dW||':<22}")
    print(f"  {'-'*52}")
    for i in [0, 1, 10, 20, 30, 40, 44, 47, 48, 49]:
        if i < len(gn_old):
            print(f"  {i:<8} {gn_old[i]:<22.6e} {gn_new[i]:<22.6e}")

    # ==================================================================
    # PART 5: What Adam does with the first training step
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("PART 5: FIRST ADAM WEIGHT UPDATE")
    print("=" * 70)

    lr = 1e-3
    eps = 1e-8
    for name, gnorms in [("OLD", gn_old), ("NEW", gn_new)]:
        deep_g = gnorms[10] if len(gnorms) > 10 else 0
        near_g = gnorms[48] if len(gnorms) > 48 else 0
        deep_update = lr * deep_g / (deep_g + eps) if deep_g > 0 else 0
        near_update = lr * near_g / (near_g + eps) if near_g > 0 else 0
        print(f"\n  {name}:")
        print(f"    Deep layer 10: ||dE/dW|| = {deep_g:.2e} → "
              f"Adam update ≈ {deep_update:.2e}")
        print(f"    Near layer 48: ||dE/dW|| = {near_g:.2e} → "
              f"Adam update ≈ {near_update:.2e}")


if __name__ == "__main__":
    run()
