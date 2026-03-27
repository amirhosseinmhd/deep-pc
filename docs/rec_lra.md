# rec-LRA: Recursive Local Representation Alignment

Implementation of the algorithm from **"Backpropagation-Free Deep Learning with Recursive Local Representation Alignment"** (Ororbia & Mali, AAAI 2023).

## What is rec-LRA?

rec-LRA is a biologically plausible alternative to backpropagation. Instead of propagating gradients backward through the entire network (like backprop), it:

1. Runs a **single forward pass** through the network
2. Generates **local targets** for each layer using error skip connections
3. Updates weights with **Hebbian learning rules** (local outer products, no chain rule)

The key innovation is **error skip connections** — learnable matrices `E` that transmit error signals directly from the output layer to distant hidden layers, bypassing intermediate layers. This solves the credit assignment problem without backprop.

## How It Differs from Standard PC Variants

| | Standard PC (baseline, resnet, etc.) | rec-LRA |
|---|---|---|
| **Inference** | Iterative energy minimization (`jpc.update_pc_activities` in a loop) | Single forward pass |
| **Learning** | Gradient of PC energy (`jpc.update_pc_params`) | Hebbian outer products |
| **Error flow** | Through prediction errors between adjacent layers | Via learnable error skip connections `E` |
| **Training loop** | `training/trainer.py` | `training/rec_lra_trainer.py` (custom) |

## Architecture

An L-layer MLP with two types of skip connections controlled by parameters `n` and `m`:

```
Forward skip connections (every n layers):
    h_l = W_l * phi(z_{l-1}) + z_{l-n}     when l mod n == 0

Error skip connections (every m layers):
    d_l = E_{L->l} * e_L                   when l mod m == 0   (skip from output)
    d_l = E_{(l+1)->l} * e_{l+1}           otherwise           (adjacent)
```

### Visual intuition

```
                  Forward direction -->
  x --> [W1] --> [W2] --> [W3] --> [W4] --> [W5] --> z_L
         z1       z2       z3       z4      output
                   ^                 ^
                   |   <-- Error direction
                   |                 |
              E_{L->2}          E_{5->4}
                   |                 |
                   +--- e_L ---------+     (error skip: layers where l%m==0)
                                           (adjacent:   all other layers)
```

## Algorithm (3 steps per iteration)

### Step 1: Forward Pass (RUNMODEL)

Standard forward propagation with optional residual skip connections every `n` layers:

```python
z_0 = x  (input)
for l = 1 to L:
    h_l = W_l * phi(z_{l-1})           # pre-activation
    if l % n == 0 and l >= n:
        h_l = h_l + z_{l-n}            # forward skip
    z_l = phi(h_l)                      # post-activation (no phi on output)
```

### Step 2: Target Generation (CALCERRRUNITS)

Backward sweep that computes a **target** for each layer. The target is what the layer's activation *should have been*. The error skip connections `E` transmit correction signals:

```python
e_L = z_L - y                          # output error

for l = (L-1) down to 1:
    if l % m == 0:
        d_l = E_{L->l} * e_L           # error skip: correction from output
    else:
        d_l = E_{(l+1)->l} * e_{l+1}   # adjacent: correction from next layer

    y_l = phi(h_l - beta * d_l)         # target = what z_l should have been
    e_l = z_l - y_l                     # mismatch between actual and target
```

`beta` controls how strongly the error signal nudges the target. Larger beta = more aggressive correction.

### Step 3: Hebbian Weight Updates (COMPUTEUPDATES)

All updates are local outer products — no chain rule, no backprop:

```python
# Forward weight update (Eq. 5 in paper):
delta_W_l = e_l * phi(z_{l-1})^T       # Hebbian: error x pre-synaptic activity

# Error synapse update (Eq. 6 in paper):
delta_E_{j->i} = -gamma * d_i * e_j^T  # Hebbian: displacement x source error
```

The W update has the form of a classic Hebbian rule: the weight change is proportional to the outer product of the post-synaptic error and the pre-synaptic activation.

## Parameters

### Architecture

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `depth` | `--depths` | — | Number of layers (L). Set per experiment. |
| `width` | (in config) | 128 | Hidden layer width (uniform across all hidden layers) |
| `act_fn` | `--act-fns` | relu | Activation function: `tanh`, `relu`, `selu`, etc. |
| `forward_skip_every` | `--forward-skip-every` | 2 | Forward residual skip interval (n). Set to 0 to disable. |
| `error_skip_every` | `--error-skip-every` | 2 | Error skip connection interval (m). Set to 0 for all-adjacent. |

### Training

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `param_lr` | `--param-lr` | 0.001 | Learning rate for W weight updates |
| `e_lr` | `--e-lr` | 0.01 | Learning rate for E matrix updates |
| `beta` | `--beta` | 1.0 | Target nudging strength. Controls how much the error signal perturbs the target. |
| `gamma_E` | `--gamma-e` | 0.1 | Scaling factor in the E update rule (Eq. 6). Controls E learning speed relative to displacement magnitude. |
| `rec_lra_optim` | `--rec-lra-optim` | sgd | Optimizer: `sgd` (vanilla, paper-faithful) or `adam` (wraps Hebbian deltas as pseudo-gradients) |
| `rec_lra_loss` | `--rec-lra-loss` | mse | Loss for reporting: `mse` or `ce` (cross-entropy). Note: the error signal `e_L = z_L - y` is always the raw difference regardless of this setting. |
| `rec_lra_e_update` | `--rec-lra-e-update` | hebbian | E update rule: `hebbian` (Eq. 6, biologically plausible) or `grad` (rLRA-dx, true gradient via autodiff) |

### How the parameters interact

- **`beta`** and **`gamma_E`** work together: `beta` controls how much `d_l` perturbs targets, and `gamma_E` controls how fast `E` learns from `d`. If `beta` is too small, errors don't propagate and lower layers won't learn. If too large, targets overshoot and training becomes unstable.
- **`param_lr`** and **`e_lr`** are independent learning rates. Generally `e_lr >= param_lr` works well because E matrices need to learn quickly to provide useful error signals.
- **`forward_skip_every=0`** disables forward skips entirely (plain MLP). **`error_skip_every=0`** makes all error connections adjacent (no long-range error skips).

## Quick Start

```bash
conda activate jax_env

# Basic run (5-layer MLP, MNIST, 1000 iterations)
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --no-wandb

# Deeper network with custom skip intervals
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --act-fns tanh --param-lr 0.01 --forward-skip-every 3 --error-skip-every 5 --no-wandb

# No forward skips, only error skips (isolate the E contribution)
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --forward-skip-every 0 --error-skip-every 2 --no-wandb

# No error skips, only adjacent error connections
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --error-skip-every 0 --no-wandb

# Using Adam optimizer instead of SGD for Hebbian updates
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --rec-lra-optim adam --no-wandb

# Cross-entropy loss reporting
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --rec-lra-loss ce --no-wandb

# Compare rec-LRA against standard PC baselines
python run_training.py --variant rec_lra baseline resnet --depths 5 10 \
    --act-fns tanh --n-iters 1000 --param-lr 0.01 --no-wandb
```

## Experiment Ideas

### 1. Effect of error skip interval (m)

How does the error skip interval affect learning in deep networks?

```bash
# m=0 (all adjacent), m=2 (every other), m=5 (sparse), m=depth (only output-to-all)
for m in 0 2 5; do
    python run_training.py --variant rec_lra --depths 20 --n-iters 2000 \
        --act-fns tanh --param-lr 0.01 --error-skip-every $m --no-wandb
done
```

Hypothesis: larger m means more layers rely on adjacent error propagation, which may weaken the signal at lower layers in deep networks. Smaller m (more error skips) should help deep networks learn faster.

### 2. Depth scaling

Does rec-LRA maintain learning ability as depth increases?

```bash
python run_training.py --variant rec_lra --depths 5 10 20 40 \
    --act-fns tanh --param-lr 0.01 --n-iters 3000 --no-wandb
```

### 3. Forward skip + error skip interaction

Are forward skips and error skips complementary or redundant?

```bash
# Neither skip
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --forward-skip-every 0 --error-skip-every 0 --param-lr 0.01 --no-wandb

# Only forward skips
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --forward-skip-every 2 --error-skip-every 0 --param-lr 0.01 --no-wandb

# Only error skips
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --forward-skip-every 0 --error-skip-every 2 --param-lr 0.01 --no-wandb

# Both
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --forward-skip-every 2 --error-skip-every 2 --param-lr 0.01 --no-wandb
```

### 4. Beta sensitivity

How sensitive is learning to the target nudging strength?

```bash
for beta in 0.1 0.5 1.0 2.0 5.0; do
    python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
        --act-fns tanh --param-lr 0.01 --beta $beta --no-wandb
done
```

### 5. SGD vs Adam for Hebbian updates

The paper uses plain SGD, but Adam might help by adapting per-parameter learning rates.

```bash
python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --act-fns tanh --rec-lra-optim sgd --param-lr 0.01 --no-wandb

python run_training.py --variant rec_lra --depths 10 --n-iters 2000 \
    --act-fns tanh --rec-lra-optim adam --param-lr 0.001 --no-wandb
```

### 6. Activation function comparison

The paper shows rec-LRA works with non-differentiable activations (like `sign`). Test different activations:

```bash
python run_training.py --variant rec_lra --depths 5 --n-iters 2000 \
    --act-fns tanh relu --param-lr 0.01 --no-wandb
```

## File Map

```
variants/rec_lra.py          # RecLRAVariant class — all algorithm logic
  ├── create_model()         #   Creates W (via jpc.make_mlp) and E matrices
  ├── forward_pass()         #   RUNMODEL: forward prop with skip connections
  ├── compute_targets_and_errors()  #   CALCERRRUNITS: backward target sweep
  ├── compute_hebbian_updates()     #   COMPUTEUPDATES: Hebbian deltas for W and E
  ├── apply_hebbian_updates()       #   SGD update (hook point for Variant 2)
  ├── apply_optax_updates()         #   Adam update (wraps deltas as pseudo-grads)
  └── evaluate()             #   Test accuracy via forward pass

training/rec_lra_trainer.py  # Custom training loop (no jpc inference/learning)
  └── train_rec_lra()        #   Main loop: forward -> targets -> Hebbian -> apply

config.py                    # Hyperparameters (beta, gamma_E, e_lr, skip intervals, etc.)
run_training.py              # CLI entry point (--variant rec_lra + rec-LRA flags)
```

## Model Bundle Structure

The model is a Python dict (not a single Equinox module):

```python
{
    "model": List[Sequential],    # W weights — jpc.make_mlp layers
                                  #   Each layer: Sequential([Lambda(act_fn), Linear])
    "E": List[Optional[Array]],   # Error skip matrices, length L
                                  #   E[0] = None (no error to input layer)
                                  #   E[l] for l=1..L-2: learnable error connections
    "forward_skip_every": int,    # n
    "error_skip_every": int,      # m
    "act_fn": callable,           # Activation function
    "depth": int,                 # L
    "gamma_E": float,             # Set by trainer at runtime
}
```

## Variant 2: Gradient-based E updates (rLRA-dx)

The default (Variant 1) updates E with the Hebbian rule from Eq. 6: `ΔE = -γ · d · e^T`. Variant 2 ("rLRA-dx", from the paper's appendix) computes the true gradient `∂D/∂E` where `D = Σ||e_l||²`, using JAX autodiff. This includes activation function derivatives that the Hebbian rule drops, making it mathematically precise but biologically implausible.

W updates remain Hebbian in both variants — only E updates differ.

### Usage

```bash
# Variant 1: Hebbian E updates (default)
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --no-wandb

# Variant 2: Gradient-based E updates (rLRA-dx)
python run_training.py --variant rec_lra --depths 5 --n-iters 1000 \
    --act-fns tanh --param-lr 0.01 --rec-lra-e-update grad --no-wandb
```

Note: `gamma_E` is not used in Variant 2 since the gradient is already correctly scaled. The E learning rate (`--e-lr`) still controls the step size.

## Reference

- Paper: [Backpropagation-Free Deep Learning with Recursive Local Representation Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/26093) (AAAI 2023)
- Algorithm 1 in the paper is the primary reference for this implementation
- Equation 5: W update rule (Hebbian)
- Equation 6: E update rule (Hebbian)
