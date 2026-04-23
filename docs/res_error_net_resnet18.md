# `res_error_net_resnet18`

Detailed README for the ResNet-18-based residual-error predictive-coding variant in this repository.

## Overview

`res_error_net_resnet18` is the convolutional, CIFAR-oriented version of the repo's residual-error network idea.

It keeps a genuine ResNet-18-style forward backbone:

- 1 stem convolution
- 4 residual stages
- 2 basic blocks per stage
- 8 total residual blocks
- global average pooling
- linear classifier

On top of that forward backbone, it adds a set of learnable **error highways** `V_{L→i}`. Each highway maps the output-layer error directly into an earlier activity tensor, so the model does not rely only on layer-by-layer error transport during predictive-coding inference.

In this implementation, highways can target:

- the stem output
- every residual block output

That means up to **9 highway target activities** in total:

- stem output
- block 1 output
- block 2 output
- block 3 output
- block 4 output
- block 5 output
- block 6 output
- block 7 output
- block 8 output

The classifier output is not itself a highway target; it is the source of the supervisory error signal.

## Why This Variant Exists

Deep predictive-coding models can struggle because the output supervision must influence early layers through iterative dynamics. As the network gets deeper, that signal can weaken or become expensive to recover with many inference steps.

This variant tries to help with that by learning direct projections from the output error to intermediate ResNet activities. Conceptually:

- ResNet skip connections help the **forward** information path
- residual-error highways help the **backward / inference** credit path

The goal is not to replace predictive coding with backprop, but to make iterative inference in a deep CNN backbone more usable.

## Core Idea

The implementation augments the usual predictive-coding free energy with a highway term:

```text
F = F_pc + F_highway
```

where:

```text
F_pc = Σ_ℓ (1/2) mean ||e^ℓ||²
```

and the highway contribution is:

```text
F_highway = α Σ_{i∈S} mean( flatten(z^i) · ( stopgrad(e^L) @ V_i^T ) )
```

Interpretation:

- `z^i` is the activity at target layer `i`
- `e^L` is the classifier/output prediction error
- `V_i` is the learnable highway matrix for target layer `i`
- `α` controls highway strength
- `S` is the set of highway target indices

Two details matter a lot in this repo's implementation:

1. The highway term uses the **state** `z^i`, not the local error `e^i`.
2. `e^L` is wrapped in `stop_gradient`, so the highway term does not directly reshape the penultimate state through gradients flowing backward from the output error term.

That design keeps the last-layer alignment governed by standard predictive-coding error while still letting earlier states feel the output-level signal.

## Architecture

### Forward backbone

The forward model is built in `variants/res_error_net_resnet18.py` and consists of:

1. `Stem`
   - `3x3` convolution
   - optional DyT normalization
   - activation
2. `BasicBlock` repeated 8 times
   - conv -> optional DyT -> activation
   - conv -> optional DyT
   - residual addition
   - activation
3. `Head`
   - global average pooling
   - linear classifier

Default channel layout:

```text
[64, 64, 128, 256, 512]
```

This means:

- stem output channels: `64`
- stage 1 blocks: `64`
- stage 2 blocks: `128`
- stage 3 blocks: `256`
- stage 4 blocks: `512`

Default blocks per stage:

```text
2
```

So the default model is a standard ResNet-18-style layout.

### Highway matrices

For each target activity `z^i`, the code creates a full-rank matrix:

```text
V_i ∈ R^(D_i × output_dim)
```

where:

- `D_i` is the flattened size of that activity tensor
- `output_dim` is the number of classes, usually `10`

Example:

- a stem activity of shape `(64, 32, 32)` becomes `D_i = 64 * 32 * 32`
- the corresponding `V_i` maps from class-space error to that flattened feature space

This is expressive, but also expensive in memory compared with narrow feedback projections.

## Normalization Choice: DyT Instead of BatchNorm

This variant intentionally does **not** use BatchNorm.

Instead it supports:

- `dyt`
- `none`

DyT here is:

```text
DyT(x) = γ ⊙ tanh(αx) + β
```

applied per channel.

The reason is practical: predictive-coding inference performs multiple iterative state updates per batch. BatchNorm's running statistics are a poor fit for that loop, because the states seen during iterative inference do not behave like ordinary single-pass feedforward activations. DyT is stateless and per-example, so it stays consistent across inference steps.

## Training / Inference Procedure

This variant is trained by `training/res_error_net_trainer.py`.

Per batch, the procedure is:

1. Feedforward initialize all layer states.
2. Clamp the output state to the target labels.
3. Run `T` Euler inference steps on the free states.
4. Compute converged prediction errors.
5. Compute `ΔW` for model parameters.
6. Compute `ΔV` for the highway matrices.
7. Optionally re-project updates to a Gaussian ball.
8. Apply updates with Adam, AdamW, or SGD.

## Inference Dynamics

Inference updates the hidden states by gradient descent on the augmented free energy:

```text
z <- z - dt * ∂F/∂z
```

The implementation does this with autodiff over the free states and runs the loop with a JIT-compiled `jax.lax.scan` when energy logging is disabled.

Important consequences:

- larger `T` gives states more time to settle
- larger `dt` makes inference more aggressive, but can destabilize it
- larger `α` increases the effect of output-error highways on hidden states

The ResNet-18 variant is much more expensive per inference step than the MLP version, which is why it has its own `res_resnet_inference_T` setting.

## Parameter Updates

### Model-parameter updates (`W`)

`compute_W_updates` differentiates the model energy with respect to the learnable model parameters:

- convolution weights
- linear weights
- DyT parameters (`alpha`, `gamma`, `beta`) when DyT is enabled

There is also a fused path, `compute_updates_fused`, which computes:

- errors
- `ΔW`
- `ΔV`
- per-layer energies

in one JITted pass to avoid redundant forward/backward work.

### Highway updates (`V`)

The code currently supports two rules:

#### `energy`

```text
ΔV_i = α * z^i * (e^L)^T / B
```

where `z^i` is flattened and batch-averaged.

This is the default in the codebase.

#### `state`

```text
ΔV_i = -α * z^i * (e^L)^T / B
```

This is kept as an ablation / alternative rule.

### Update re-projection

Both `ΔW` and `ΔV` can be reprojected to a Gaussian ball using the same helper used by related variants in this repo. This is a stabilization measure and is controlled by `--reproject-c`.

## Repo Defaults

The current code defaults come from `config.py` and the variant implementation.

### Variant-specific defaults

| Setting | Current default |
|---|---|
| `variant` | `res_error_net_resnet18` |
| `dataset` | usually used with `CIFAR10` |
| `res_alpha` | `1.0` |
| `res_inference_dt` | `0.1` |
| `res_v_lr` | `1e-4` |
| `res_v_update_rule` | `energy` |
| `res_v_reg` | `0.0` |
| `res_optim` | `adamw` |
| `res_loss` | `mse` |
| `reproject_c` | `1.0` |
| `input_noise_sigma` | `0.1` |
| `weight_decay` | `1e-4` |

### ResNet-18-specific defaults

| Setting | Current default |
|---|---|
| `res_resnet_channels` | `[64, 64, 128, 256, 512]` |
| `res_resnet_blocks_per_stage` | `2` |
| `res_resnet_normalization` | `dyt` |
| `res_resnet_dyt_init_alpha` | `0.5` |
| `res_resnet_highway_include_stem` | `True` |
| `res_resnet_inference_T` | `30` |
| `res_v_init_scale` used by this variant | `0.01` in `create_model(...)` unless overridden |

## Important Note About a CLI Help Mismatch

`run_training.py` currently advertises:

- `--res-resnet-inference-T` default: `15`

But `config.py` currently sets:

- `res_resnet_inference_T = 30`

So if you rely on the code rather than the help text, the effective repo default is **30** unless explicitly overridden on the command line.

## Command-Line Flags

The main entry point is `run_training.py`.

### Common training flags

```bash
--variant
--dataset
--depths
--act-fns
--n-iters
--seed
--batch-size
--param-lr
--test-every
--no-wandb
```

### Residual-error flags

```bash
--res-alpha
--res-inference-T
--res-inference-dt
--res-v-lr
--res-v-update-rule
--res-v-init-scale
--res-optim
--res-loss
--res-v-reg
--reproject-c
--input-noise-sigma
--weight-decay
```

### ResNet-18-specific flags

```bash
--res-resnet-channels
--res-resnet-blocks-per-stage
--res-resnet-normalization
--res-resnet-dyt-init-alpha
--res-resnet-no-stem-highway
--res-resnet-inference-T
```

## How To Run It

### Minimal example

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --n-iters 100 \
  --batch-size 32 \
  --test-every 5 \
  --no-wandb
```

Why `--depths 1`?

For this variant, the ResNet-18 layout is fixed inside the variant itself. The external `depth` loop still exists because the training script expects it, but the variant's `create_model` ignores `depth` and `width`.

### Baseline-style CIFAR-10 run

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --seed 42 \
  --n-iters 200 \
  --batch-size 32 \
  --test-every 5 \
  --param-lr 1e-4 \
  --res-v-lr 1e-4 \
  --res-alpha 1.0 \
  --res-inference-dt 0.1 \
  --res-resnet-inference-T 30 \
  --res-v-update-rule energy \
  --res-optim adamw \
  --res-loss mse \
  --res-resnet-normalization dyt \
  --res-resnet-dyt-init-alpha 0.5 \
  --no-wandb
```

### Ablations worth running

#### Remove stem highway

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --res-resnet-no-stem-highway \
  --no-wandb
```

#### Disable DyT

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --res-resnet-normalization none \
  --no-wandb
```

#### Compare highway learning rules

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --res-v-update-rule energy \
  --no-wandb

python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --res-v-update-rule state \
  --no-wandb
```

#### Change inference budget

```bash
python run_training.py \
  --variant res_error_net_resnet18 \
  --dataset CIFAR10 \
  --depths 1 \
  --act-fns relu \
  --res-resnet-inference-T 10 \
  --res-inference-dt 0.1 \
  --no-wandb
```

## Outputs

Runs are saved under:

```text
results/res_error_net_resnet18/
```

For a given activation/depth combination, the training script writes:

- raw metric arrays as `.npy`
- plots such as performance and train-loss curves

Typical log lines look like:

```text
Iter 20, loss=..., train acc=..., test acc=...
```

The training script evaluates with a plain feedforward pass at test time, not with clamped iterative inference.

## Evaluation Behavior

`evaluate(...)` in the variant:

- runs a normal forward pass
- uses the final classifier logits
- computes top-1 accuracy

So the reported test accuracy is the standard feedforward accuracy of the learned model.

## Things To Watch Out For

### 1. This variant is expensive

A ResNet-18 backbone inside iterative predictive-coding inference is much heavier than the MLP variants. Increasing `T` can quickly multiply runtime.

### 2. Highway matrices can be large

Because each `V_i` is full-rank in flattened feature space, memory usage can rise noticeably, especially for early high-resolution layers.

### 3. `depth` does not mean network depth here

The script still loops over `--depths`, but the variant internally uses a fixed ResNet-18 topology. In practice, use `--depths 1` for this variant.

### 4. CLI help is not perfectly synced with config defaults

The `res_resnet_inference_T` mismatch mentioned above is the clearest example. When in doubt, trust the actual config and variant code.

### 5. The implementation comments and formulas are not perfectly aligned everywhere

The top-of-file docstring and some trainer comments describe the highway and `V` update rule in slightly different terms from the executable code. The most reliable source is the actual implementation:

- highway energy uses `z^i`
- `compute_V_updates(..., rule="energy")` also uses `z^i`

If you plan to write a paper or report from this variant, it is worth re-deriving the exact update equations from the final code rather than relying only on comments.

## Recommended First Experiments

If you are trying to understand whether the variant is working, these are good first checks:

1. Compare `energy` vs `state` V updates.
2. Compare `dyt` vs `none`.
3. Compare stem-highway enabled vs disabled.
4. Sweep `res_alpha` over a small range such as `0.05, 0.1, 0.3, 1.0`.
5. Sweep `res_resnet_inference_T` over `5, 10, 20, 30`.
6. Watch for divergence or degradation in the logs.

## Implementation Map

Main files for this variant:

- `variants/res_error_net_resnet18.py`
  ResNet-18 backbone, highway definitions, inference dynamics, update rules, evaluation.
- `training/res_error_net_trainer.py`
  Training loop for residual-error variants, including this one.
- `run_training.py`
  CLI entry point, config overrides, dispatch to the trainer.
- `config.py`
  Global defaults, including the ResNet-18-specific settings.
- `program_res_error_net.md`
  Search notes and experiment strategy for this variant family.

## Summary

`res_error_net_resnet18` is a deep predictive-coding CNN that combines:

- a ResNet-18-style forward architecture
- iterative predictive-coding inference
- learnable output-error highways into early and intermediate feature maps
- DyT normalization to keep the inference loop stateless

It is one of the more ambitious variants in this repository: more expressive than the MLP residual-error model, but also heavier and more sensitive to inference/training hyperparameters. If you treat it as a CIFAR-10 ResNet-18 backbone with predictive-coding state inference plus direct error-routing highways, you will have the right mental model.
