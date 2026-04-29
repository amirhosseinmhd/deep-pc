# res-error-net: Deep Iterative PC with Residual Error Highways

A deep predictive coding (PC) network augmented with learnable **residual error highways** `V_{L→i}` that carry the output error `e^L` directly back to hidden layer `i`, analogous to how ResNet's forward skip connections carry activations forward. Designed to address the vanishing error-signal problem in deep PC.

## Motivation

Deep PC networks suffer a well-known **vanishing-gradient / vanishing-signal** problem: because learning is driven by *local* prediction errors propagated only between adjacent layers, layers far from the supervisory signal see progressively weaker updates, and their `z`-state dynamics stall. ResNet solves the analogous problem on the **forward** pathway with skip connections. This variant does the same on the **backward / error** pathway: each hidden layer `i ∈ S` gets a direct learnable projection of the output error `e^L` into its inference dynamics.

### Relation to rec-LRA

rec-LRA uses `E` matrices to route error signals in a **single-pass backward target sweep** (no iterative inference; W updates are Hebbian outer products).

res-error-net does something structurally different:

- It is a **true iterative PC network**: `z` states evolve under an ODE on the free energy `F`, standard PC W gradient descent at the fixed point, no Hebbian approximation on `W`.
- The `V` matrices enter the **inference dynamics** (how `z` updates), not the target computation.
- The `V` update rule drops out of an **augmented free energy** (derivation below) rather than by Hebbian analogy.
- It can also optionally use **forward residual skips** in the MLP backbone, so you can combine a ResNet-like forward path with the output-error highways.

## Augmented Free Energy

Standard PC free energy plus a bilinear coupling between each layer's local error and the output error:

```
F({z^ℓ}, Θ, {V_{L→ℓ}}) = Σ_ℓ (1/2) mean ‖e^ℓ‖²                         (standard PC)
                         + α · Σ_{ℓ ∈ S} mean( z^ℓ · ( sg(e^L) @ V_{L→ℓ}ᵀ ) )   (error highway coupling)
```

The highway factor is the **state** `z^ℓ` (not the local error `e^ℓ`), and `e^L`
is wrapped in `stop_gradient` so the highway term has no influence on `z^{L-1}`
during inference (the last free layer aligns to the clamped target via `F_pc`
only) and contributes 0 to `∂F/∂W`.

where `e^ℓ = z^ℓ − μ^ℓ(z^{ℓ-1})`, `μ^ℓ` is the forward prediction from below, and `S ⊂ {1, …, L-1}` is the set of highway-endpoint layer indices. The output is hard-clamped: `z^L = y` during training, so `e^L = y − μ^L(z^{L-1})`.

If `res_forward_skip_every = n > 0`, the hidden-layer prediction also gets a residual term every `n` layers:

```
μ^ℓ = f(W_ℓ z^{ℓ-1}) + z^{ℓ-n}
```

when `ℓ ≥ n` and the source and target widths match. This skip is applied to the predicted hidden state itself (post-activation), not to the highway energy term.

### Inference dynamics

Gradient descent on `F` with respect to free `z^i` (for `i < L-1`, since `z^L` is clamped). With `e^L` wrapped in `stop_gradient`, the highway term contributes only through its `z^i` factor, giving:

```
ż^i = -e^i + J_iᵀ e^{i+1} - α · (V_{L→i} sg(e^L)   if i ∈ S else 0)
```

where `J_i = ∂μ^{i+1}/∂z^i`. In practice we use `jax.grad` on the full augmented `F`; the `stop_gradient` blocks any flow through `e^L` so `z^{L-1}` is shaped only by `F_pc`.

### Learning rules

- **W update** (standard PC; `F_hw` contributes 0):
  ```
  ΔW^ℓ = ∂F_pc/∂W^ℓ      → applied as W^ℓ ← W^ℓ - η_W · ΔW^ℓ
  ```
  Under the new energy `z` is constant w.r.t. model params and `stop_gradient` blocks flow through `e^L`, so `∂F_hw/∂W = 0` and the W update reduces to ∂F_pc/∂W. At the inference fixed point this recovers the usual PC rule `ΔW^ℓ ∝ e^ℓ · φ(z^{ℓ-1})ᵀ`. Computed via `jax.grad` on the augmented closure (which simplifies to F_pc for W).
- **V update** (two options, selected by `--res-v-update-rule`):
  - `energy` (default): dropped from `∂F_hw/∂V`:
    ```
    ΔV_{L→i} = +α · z^i (e^L)ᵀ        (batch-averaged)
    ```
    Jointly grounded with `W` in the same energy.
  - `state` (Hebbian anti-gradient, kept for ablation):
    ```
    ΔV_{L→i} = -α · z^i (e^L)ᵀ        (stored negated so optax subtraction yields +α·z^i·(e^L)ᵀ growth)
    ```
    Same direction as `energy` but opposite sign convention.

## Visual Intuition

```
     Forward direction -->
  x --[W0]--> z0 --[W1]--> z1 --[W2]--> z2 --[W3]--> z3 --[W4]--> z4 --[W5]--> z5 = y (clamped)
                              ^                          ^
                              |                          |
                          V_{L→1}                    V_{L→3}
                              |                          |
                              +------- α·e^L <-----------+
                                       (highways, every_k=2)

  Each V_{L→i} carries the output error e^L directly to layer i's ż^i dynamics,
  bypassing all intermediate error passes.
```

With `highway_every_k=2` and `L=6`, the highway endpoint set becomes `S = {1, 3}` (i.e. `L−2·k` for `k=1, 2, …` clipped to `[1, L−2]`).

## Algorithm (per batch)

```python
# Step 1. Feed-forward initialise z
z_init, _ = forward_pass(model, x)

# Step 2. Hard-clamp output
z[L-1] = y

# Step 3. Run T Euler inference steps
for t = 1 .. T:
    for i = 0 .. L-2:   # free layers only
        grad_i = ∂F_aug/∂z^i   (via jax.grad)
        z^i ← z^i − dt · grad_i

# Step 4. Compute errors at the fixed point
e^ℓ = z^ℓ − μ^ℓ(z^{ℓ-1})       for ℓ = 0 .. L-1   (with z^{L-1} = y)

# Step 5. Compute ΔW from F_pc only (jax.grad)
ΔW^ℓ = ∂F_pc/∂W^ℓ

# Step 6. Compute ΔV per configured rule
ΔV_{L→i} = +α · z^i (e^L)ᵀ / B         (energy)
       or  −α · z^i (e^L)ᵀ / B         (state)

# Step 7. Re-project both to Gaussian ball of radius c (paper trick from rec-LRA)
# Step 8. Apply via AdamW (default) or SGD
```

## Parameters

### Architecture

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `depth` | `--depths` | — | Number of layers `L` (counting input-side Linear as layer 0 and output as layer L-1). |
| `width` | (in config) | 128 | Hidden layer width (uniform). |
| `act_fn` | `--act-fns` | relu | Activation: `tanh`, `relu`, `selu`, etc. |
| `res_forward_skip_every` | `--res-forward-skip-every` | 0 | Forward residual skip interval for the MLP backbone. `0` disables skips; `n>0` adds `z^{ℓ-n}` into the prediction of layer `ℓ` when widths match. |
| `res_highway_every_k` | `--res-highway-every-k` | 2 | Stride of the V highways. `S = {L−1−k, L−1−2k, …} ∩ [1, L−2]`. Larger k = sparser highways. |
| `res_v_init_scale` | `--res-v-init-scale` | 0.01 | Init scale for `V_{L→i}` (N(0, σ²)). Small so the highway starts near off. |
| `res_init_scheme` | `--res-init-scheme` | `jpc_default` | `jpc_default` = 1/√fan_in (stable without ZCA); `unit_gaussian` = rec-LRA paper style (needs ZCA / small dt to avoid activation blow-up). |

### Inference

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `res_inference_T` | `--res-inference-T` | 20 | Euler steps per batch. More steps = closer to the inference fixed point. |
| `res_inference_dt` | `--res-inference-dt` | 0.1 | Euler step size. If activations are large (unit_gaussian), reduce to 0.01 or below. |
| `res_output_clamp` | (config only) | `hard` | `hard`: `z^L = y`. `soft` reserved for future work. |

### Learning

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `param_lr` | `--param-lr` | 1e-4 | Learning rate for W updates. |
| `res_v_lr` | `--res-v-lr` | 1e-4 | Learning rate for V highways. |
| `res_alpha` | `--res-alpha` | 0.1 | Global coupling strength `α`. `α=0` disables the highways (reduces to plain iterative PC). |
| `res_v_update_rule` | `--res-v-update-rule` | energy | `energy` (from `∂F_aug/∂V`) or `state` (original Hebbian sketch). |
| `res_optim` | `--res-optim` | adamw | `sgd`, `adam`, or `adamw` for both W and V. |
| `res_loss` | `--res-loss` | mse | Loss for reporting: `mse` or `ce`. |
| `reproject_c` | `--reproject-c` | 1.0 | Gaussian-ball radius for update re-projection (applied to both ΔW and ΔV). 0 disables. |
| `input_noise_sigma` | `--input-noise-sigma` | 0.1 | σ of additive Gaussian input noise during training. |
| `weight_decay` | `--weight-decay` | 1e-4 | AdamW weight decay (applied to both W and V). |

### How the knobs interact

- **`α`** controls how strongly `e^L` drives inference at highway endpoints. Too small → highway does nothing and you recover plain iterative PC. Too large → `z^i` is yanked toward an output-conditioned fixed point and local prediction errors can't settle.
- **`res_inference_dt` × `res_inference_T`** sets the effective "integration time" of the ODE. If dt is too large the Euler scheme diverges; if T is too small z doesn't converge. Total integration `dt·T` ≈ 1–5 is usually enough.
- **`res_v_lr`** should be comparable to or smaller than `param_lr` at first — V matrices live in a different regime from W and can destabilise training if they grow too fast. The reproject-to-ball trick helps bound per-step growth.
- **`res_forward_skip_every`** changes the forward prediction family itself. Smaller values create denser residual shortcuts; `0` keeps the original plain-MLP backbone.
- **`res_highway_every_k`**: lower `k` = denser highways (more V matrices), more parameters, stronger credit-assignment signal; higher `k` = sparser, cleaner to analyse.

## Quick Start

```bash
conda activate jax_env

# Basic run: depth 6 MLP, MNIST, 1000 iterations, default (α=0.1 energy rule)
python run_training.py --variant res_error_net --depths 6 --n-iters 1000 \
    --act-fns relu --no-wandb

# CIFAR-10 (flat MLP on 3072-dim flattened pixels). input_dim is derived
# from --dataset; --use-zca enables GCN+ZCA preprocessing.
python run_training.py --variant res_error_net --dataset CIFAR10 \
    --depths 6 --n-iters 2000 --act-fns relu --use-zca --no-wandb

# Deeper network — the regime where the highway should matter most
python run_training.py --variant res_error_net --depths 12 --n-iters 3000 \
    --act-fns relu --res-alpha 0.1 --res-highway-every-k 2 --no-wandb

# Add forward residual skips every 2 layers as well
python run_training.py --variant res_error_net --depths 12 --n-iters 3000 \
    --act-fns relu --res-forward-skip-every 2 \
    --res-alpha 0.1 --res-highway-every-k 2 --no-wandb

# α = 0 (disables highway — reduces to plain iterative PC, useful as a baseline)
python run_training.py --variant res_error_net --depths 12 --n-iters 3000 \
    --act-fns relu --res-alpha 0.0 --no-wandb

# Compare energy vs. state V-update rule
python run_training.py --variant res_error_net --depths 8 --n-iters 2000 \
    --act-fns relu --res-v-update-rule energy --no-wandb
python run_training.py --variant res_error_net --depths 8 --n-iters 2000 \
    --act-fns relu --res-v-update-rule state --no-wandb

# Rec-LRA-style paper init (needs ZCA to avoid blow-up)
python run_training.py --variant res_error_net --depths 8 --n-iters 2000 \
    --act-fns relu --res-init-scheme unit_gaussian --use-zca \
    --res-inference-dt 0.01 --no-wandb

# More inference steps per batch
python run_training.py --variant res_error_net --depths 6 --n-iters 1000 \
    --act-fns relu --res-inference-T 50 --res-inference-dt 0.05 --no-wandb
```

## Experiment Ideas

### 1. Vanishing-error diagnostic — the primary claim of this variant

Does the highway keep deep-layer error signals from collapsing? Compare `α=0` (plain PC) vs `α>0` at depth 12 and track `‖e^ℓ‖` per layer:

```bash
for alpha in 0.0 0.01 0.1 0.3 1.0; do
    python run_training.py --variant res_error_net --depths 12 \
        --n-iters 3000 --act-fns relu --res-alpha $alpha --no-wandb
done
```

Expectation: without highway, `‖e^ℓ‖` decays with depth (layers far from output barely update). With a good `α`, `‖e^ℓ‖` stays order-of-magnitude constant across depth. Look at `energy_per_layer` in saved metrics and `v_norm/layer_i` in W&B.

### 2. V update rule: energy vs state

the `energy` rule (`e^i (e^L)ᵀ`) is principled and the `state` rule (`z^i (e^L)ᵀ`) keeps drifting at the fixed point. Empirically test it:

```bash
for rule in energy state; do
    python run_training.py --variant res_error_net --depths 8 \
        --n-iters 3000 --act-fns relu --res-v-update-rule $rule --no-wandb
done
```

Expectation: `energy` should plateau in `‖V‖` once local errors vanish; `state` should keep growing. Whether `state` actually hurts performance is the interesting question.

### 3. Highway stride

Does denser-is-better?

```bash
for k in 2 3 4; do
    python run_training.py --variant res_error_net --depths 12 \
        --n-iters 3000 --act-fns relu --res-highway-every-k $k --no-wandb
done
```

### 4. Inference-step scaling

How many Euler steps are needed before `z` converges enough that gradient descent on `W` is meaningful?

```bash
for T in 5 10 20 50 100; do
    python run_training.py --variant res_error_net --depths 8 \
        --n-iters 2000 --act-fns relu --res-inference-T $T --no-wandb
done
```

### 5. Depth scaling

Compare plain PC (α=0) vs highway (α=0.1) across depths — the expected win grows with depth.

```bash
for depth in 4 6 8 12 16; do
    python run_training.py --variant res_error_net --depths $depth \
        --n-iters 3000 --act-fns relu --res-alpha 0.0 --no-wandb
    python run_training.py --variant res_error_net --depths $depth \
        --n-iters 3000 --act-fns relu --res-alpha 0.1 --no-wandb
done
```

### 6. Stability check — does ‖V‖ stay bounded?

Track `V_arrays` norm over a long run with and without re-projection:

```bash
python run_training.py --variant res_error_net --depths 8 --n-iters 5000 \
    --act-fns relu --reproject-c 1.0 --no-wandb

python run_training.py --variant res_error_net --depths 8 --n-iters 5000 \
    --act-fns relu --reproject-c 0.0 --no-wandb   # disabled
```

If `V` explodes without re-projection, tighten it (e.g., `--reproject-c 0.5`). The `diagnose_cnn_stability.py` pattern can be adapted.

## File Map

```
variants/res_error_net.py            # ResErrorNetVariant — all algorithm logic
  ├── create_model()                 #   Builds MLP stack + V_{L→i} matrices; init scheme flag
  ├── forward_pass()                 #   Feed-forward init of z states
  ├── _predictions_and_errors()      #   μ^ℓ = f_ℓ(z^{ℓ-1});  e^ℓ = z^ℓ − μ^ℓ
  ├── free_energy_z()                #   Augmented F(z_free, x, y, α) for inference grad
  ├── inference_step()               #   One Euler step on z^0 … z^{L-2}
  ├── run_inference()                #   T-step inference loop (optionally records F(t))
  ├── compute_W_updates()            #   ∂F_pc/∂W^ℓ via eqx.filter_grad
  ├── compute_V_updates()            #   α·e^i·(e^L)ᵀ  or  −α·z^i·(e^L)ᵀ
  ├── apply_optax_updates()          #   AdamW / Adam step on both W and V
  ├── apply_sgd_updates()            #   Plain SGD alternative
  └── evaluate()                     #   Test accuracy via feed-forward (no inference, no clamp)

training/res_error_net_trainer.py    # Per-batch training loop
  └── train_res_error_net()          #   init → clamp → inference → ΔW, ΔV → reproject → apply

config.py                            # ExperimentConfig res_* fields
run_training.py                      # CLI dispatch (--variant res_error_net + --res-* flags)
docs/res_error_net.md                # This file
```

## Model Bundle Structure

```python
{
    "model":       List[Sequential],   # W weights (jpc.make_mlp output, optionally re-init'd)
                                       #   Each layer: Sequential([Lambda(act_fn), Linear])
    "V_list":      List[Array],        # One V_{L→i} per i ∈ highway_indices, shape (dims[i], dims[L-1])
    "highway_indices": List[int],      # S = {L-1 − k, L-1 − 2k, …} ∩ [1, L-2], sorted ascending
    "highway_every_k": int,
    "v_init_scale":    float,
    "init_scheme":     str,            # "jpc_default" or "unit_gaussian"
    "act_fn":          callable,
    "act_fn_name":     str,
    "depth":           int,
    "dims":            List[int],      # [width, width, …, width, output_dim]
}
```

## Notes, Caveats, and TODOs

- **Output clamp**: `hard` is the only supported mode right now. Soft-clamp (z^L evolves under `-e^L`) is more faithful to the continuous derivation but risks positive-feedback loops through `V e^L`. The config field is reserved but not wired.
- **`α` is a global scalar**: per-layer `α_i` or learnable `α_i` (trained via the energy gradient) would be principled extensions. Not implemented — sweep `α` globally first.
- **Multi-source highways**: currently only `V_{L→i}`. Adding `V_{j→i}` for intermediate sources `j < L` is an obvious generalisation (O(L²) matrices); deferred.
- **CNN variant**: the MLP variant already runs on CIFAR-10 as a flat MLP over flattened pixels (input_dim=3072), which is enough for ablation studies of the highway mechanism itself. A convolutional `cnn_res_error_net.py` (analogous to `cnn_rec_lra.py`) is still TODO if spatial structure matters for the experiment. The ResNet-18 backbone variant (`res_error_net_resnet18`) is the convolutional counterpart already in tree.
- **Inference loop is a Python `for`, not `jax.lax.scan`**: fine for correctness, slower than it could be. If running becomes a bottleneck, wrap `inference_step` in `scan` with `(z, e)` as carry.
- **Sign convention sanity check**: with `e^ℓ = z^ℓ − μ^ℓ`, standard PC `ΔW^ℓ ∝ e^ℓ · φ(z^{ℓ-1})ᵀ` points downhill. `compute_W_updates` uses `jax.grad` on `F_pc`, yielding `+∂F/∂W`; applied as `W ← W − lr · ΔW` via optax. A flipped sign would silently make learning go uphill — verify by confirming train loss decreases in the first few iterations.
- **`α = 0` regression**: running with `--res-alpha 0.0` removes the highway contribution entirely and must match a plain iterative PC baseline. Verified in smoke tests; keep as a sanity check when changing inference code.

## Related

- [rec_lra.md](rec_lra.md) — the single-pass target-generation variant this was structurally inspired by (though mathematically quite different).
- `variants/rec_lra_common.py` — shared `reproject_to_ball` / `add_input_noise` helpers reused here.
- `training/rec_lra_trainer.py` — the trainer this one mirrors in structure (but with an inner T-step inference loop and W autodiff instead of Hebbian).
