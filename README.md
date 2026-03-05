# Predictive Coding Experiments

Experiments comparing different normalization and skip-connection strategies in Predictive Coding Networks (PCNs) on MNIST.

## What are we testing?

We train PC networks with 6 architectural variants and measure how they affect training dynamics:

| Variant | Description |
|---------|-------------|
| **baseline** | Vanilla MLP, no skip connections |
| **resnet** | MLP with residual (skip) connections |
| **bf** | ResNet + BatchNorm (BN wraps linear+skip) |
| **bf_v2** | ResNet + BatchNorm (BN wraps linear only, skip added after) |
| **dyt** | ResNet + Dynamic Tanh replacing BN (DyT wraps linear+skip) |
| **dyt_v2** | ResNet + Dynamic Tanh (DyT wraps linear only, skip added after) |

## What do we measure?

A single training run collects all of these:

- **Performance** — train loss and test accuracy over time
- **Weight update norms** — how much each layer's weights change per step
- **Activity norms** — magnitude of hidden layer activations before and after PC inference
- **Condition number** — κ(H_z) of the activity Hessian (run separately, expensive)

## Quick start

```bash
conda activate jax_env

# Train one variant
python run_training.py --variant resnet --depths 10 20

# Train multiple variants
python run_training.py --variant resnet bf dyt --depths 10 20 40

# Shorter test run
python run_training.py --variant resnet --depths 5 --n-iters 200

# Condition number experiment
python run_condition.py --variant resnet bf dyt

# Cross-variant comparison plots (after training)
python run_comparison.py
```

Results and plots are saved to `results/<variant>/`.

## Entry points

There are three scripts, each with a different purpose:

### `run_training.py` — Train and collect metrics

This is the main script. It creates the model, runs the unified training loop, saves all metrics as `.npy` files, and generates plots. Without this you'd need to write a script every time.

```bash
python run_training.py --variant resnet bf --depths 10 20 40
python run_training.py --variant dyt --depths 5 --n-iters 500 --param-lr 0.005
```

### `run_condition.py` — Condition number (separate, expensive)

Condition number requires computing the full Hessian and its eigenvalues. This uses a smaller width (16 instead of 128) and 64-bit precision, so it's run separately from training — not something you want inside every training loop.

```bash
python run_condition.py --variant resnet bf dyt
```

### `run_comparison.py` — Cross-variant comparison plots

A post-hoc tool. After you've trained multiple variants, this loads their saved `.npy` results and puts them on the same plots for direct comparison. It doesn't train anything.

```bash
# First train variants
python run_training.py --variant resnet --depths 10 20
python run_training.py --variant bf --depths 10 20

# Then compare
python run_comparison.py --variant resnet bf --depths 10 20
```

## Project structure

```
config.py           — hyperparameters (depths, learning rates, etc.)
common/             — shared code (data loading, metrics, hessian utils)
variants/           — one file per architectural variant
training/           — unified training loop and condition number computation
plotting/           — all plotting functions
run_training.py     — main entry point for training
run_condition.py    — condition number experiments
run_comparison.py   — cross-variant comparison plots
```

The old per-experiment folders are in `obsolete/` for reference.

## How PC training works

Each training iteration has three steps:

1. **Forward pass** — initialize layer activities via a feedforward pass
2. **Inference** — update activities (not weights) for `depth` steps to minimize prediction error across layers
3. **Learning** — update weights via Adam to reduce the remaining prediction error

BF variants add a batch-stats freezing step before inference. DyT variants use a learnable `γ·tanh(α·x)+β` instead of BatchNorm — no batch statistics needed.
