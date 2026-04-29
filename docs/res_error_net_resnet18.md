# `res_error_net_resnet18`

Code-first README for the ResNet-18 residual-error predictive-coding variant in this repo.

## What This Model Is

`res_error_net_resnet18` is the convolutional version of the repo's residual-error network idea. It keeps a real ResNet-style forward backbone, but replaces ordinary one-shot backprop-style credit assignment with iterative predictive-coding inference over hidden activities.

Its distinguishing addition is a set of learnable output-error highways `V_{L→i}` that inject the classifier error directly into earlier activities during inference.

The mental model is:

- forward path: standard ResNet-18-style feature extraction
- inference path: predictive-coding state updates
- extra credit path: direct output-error shortcuts into earlier feature maps

## Why This Variant Exists

Deep predictive-coding models have a practical problem: the supervisory signal at the output has to influence early layers through iterative dynamics. As depth grows, this can become slow, weak, or unstable.

This variant tries to make deep predictive coding more usable by pairing:

- ResNet residual connections for the forward computation
- residual-error highways for the backward/inference signal

So the goal is not "a ResNet trained normally", and not "predictive coding with only local adjacent-layer transport", but a hybrid where deep convolutional features still receive a strong global teaching signal during inference.

## Forward Architecture

The implementation lives in `variants/res_error_net_resnet18.py`.

The forward backbone is:

1. `Stem`
   - `3x3` conv
   - optional DyT normalization
   - activation
2. `BasicBlock` repeated across 4 stages
   - conv -> optional DyT -> activation
   - conv -> optional DyT
   - residual add
   - activation
3. `Head`
   - global average pooling over spatial dimensions
   - bias-free linear classifier

All conv and linear layers are created with `use_bias=False`.

### Default layout

The default config is:

```text
res_resnet_channels = [64, 64, 128, 256, 512]
res_resnet_blocks_per_stage = 2
input_shape = (3, 32, 32)
output_dim = 10
```

That gives:

| Activity | Module | Shape by default | Highway target? |
|---|---|---:|---|
| `z^0` | stem output | `(64, 32, 32)` | yes, if `highway_include_stem=True` |
| `z^1` | block 1 output | `(64, 32, 32)` | yes |
| `z^2` | block 2 output | `(64, 32, 32)` | yes |
| `z^3` | block 3 output | `(128, 16, 16)` | yes |
| `z^4` | block 4 output | `(128, 16, 16)` | yes |
| `z^5` | block 5 output | `(256, 8, 8)` | yes |
| `z^6` | block 6 output | `(256, 8, 8)` | yes |
| `z^7` | block 7 output | `(512, 4, 4)` | yes |
| `z^8` | block 8 output | `(512, 4, 4)` | yes |
| `z^9` | classifier logits | `(10,)` | no |

So the default model has 10 predictive-coding activities total:

- 1 stem activity
- 8 block activities
- 1 output activity

`run_training.py` treats this as a fixed-depth architecture. The user-facing `--depths` loop is effectively ignored for this variant, and the effective depth is computed from:

```text
1 stem + 4 stages * blocks_per_stage + 1 head
```

With defaults, that is:

```text
1 + 4*2 + 1 = 10
```

### ResNet details that matter

- The first block of stages 2, 3, and 4 uses stride 2.
- When stride or channel count changes, the skip path uses a `1x1` projection conv.
- Otherwise the skip path is identity.
- The head pools by simple spatial mean, not max pool.
- The head returns raw logits.

## What Counts As The State

For this variant, each module returns the post-module activity directly:

- stem returns post-conv, post-norm, post-activation features
- each basic block returns post-residual, post-activation features
- the head returns logits

During a plain feedforward pass, the code stores these as `z[0], ..., z[L-1]`.

Those `z` tensors are also the variables that predictive-coding inference updates, except the last one, which is clamped to the target during training-time inference.

## Highway Construction

The core extra parameters are the highway matrices:

```text
V_i ∈ R^(D_i × output_dim)
```

where:

- `D_i` is the flattened size of activity `z^i`
- `output_dim` is the class dimension

Each highway takes output error in class space and projects it into the flattened coordinates of an earlier activity.

With default CIFAR-10 shapes and `highway_include_stem=True`, the total flattened target size is:

```text
3*(64*32*32) + 2*(128*16*16) + 2*(256*8*8) + 2*(512*4*4) = 311,296
```

So the default highway bank has:

```text
311,296 * 10 = 3,112,960
```

learnable `V` parameters.

That is one of the main design tradeoffs in this model:

- advantage: full-rank, very expressive direct error routing
- cost: nontrivial memory footprint, especially in early high-resolution layers

## Free Energy And Loss

The code optimizes an augmented predictive-coding free energy:

```text
F(z; W, V) = F_pc(z; W) + F_highway(z; V)
```

with:

```text
F_pc = Σ_l 1/2 * mean(||e^l||_2^2)
```

and:

```text
F_highway = α Σ_{i∈S} mean( flatten(z^i) · ( stopgrad(e^L) @ V_i^T ) )
```

Here:

- `e^l = z^l - μ^l`
- `μ^l` is the model prediction for layer `l`
- `S` is the set of highway target indices
- `α` is the highway strength
- `e^L` is the output-layer prediction error after clamping the output activity to the target

In the code, layers are indexed `0 ... L-1`, so the mathematical `e^L` here corresponds to `e[L-1]`.

### Predictions and errors

Given hidden states `z` and input `x`, the model computes:

```text
μ^0 = stem(x)
μ^l = block_l(z^{l-1})   for hidden blocks
μ^{L-1} = head(z^{L-2})
e^l = z^l - μ^l
```

In `compute_errors(...)`, the code explicitly replaces the final state with the target before computing errors:

```text
z^{L-1} := y
```

So the output error is always measured against a hard-clamped target during inference/training updates.

### Important design detail: the highway term uses `z^i`, not `e^i`

This is easy to miss if you only skim comments elsewhere in the repo.

In the executable code:

- the highway energy depends on `z^i`
- the default `V` update rule also depends on `z^i`

It does not use the local hidden-layer error `e^i`.

### Important design detail: `e^L` is wrapped in `stop_gradient`

The highway term uses:

```text
stop_gradient(e^L)
```

This has two important consequences:

1. The highway term does not differentiate through the output-error computation itself.
2. During weight updates, `F_highway` contributes zero gradient to the forward model parameters `W`, because `z` is treated as fixed and the only remaining dependence on `W` through `e^L` is stopped.

So:

- `F_highway` changes hidden-state inference
- `F_highway` updates `V`
- `F_highway` does not directly update forward weights `W`

That is a very deliberate separation of roles.

## Inference Dynamics

Training-time inference solves for hidden activities while keeping the output clamped:

```text
z_free <- argmin_z F(z; W, V)
```

The actual implementation uses iterative gradient-based updates on the free states.

### Euler inference

The default update is:

```text
z^i <- z^i - dt * ∂F/∂z^i
```

for all free layers `i = 0, ..., L-2`.

The output layer is always:

```text
z^{L-1} = y
```

### Adam-on-z inference

The code also supports:

```text
inference_method = "adam"
```

In that mode, hidden states are still optimized against the same free energy, but each coordinate gets Adam-style first/second-moment adaptation. In this mode `dt` acts like the inference learning rate.

### Practical meaning of the main inference knobs

- `res_resnet_alpha`: how strongly the highways influence hidden-state inference
- `res_resnet_inference_dt`: step size for Euler, or learning rate for Adam-on-z
- `res_resnet_inference_T`: number of inference steps per batch
- `res_resnet_inference_method`: `"euler"` or `"adam"`

Because each step runs a full ResNet-18-like computation inside predictive-coding inference, this variant is far more expensive per step than the MLP version. That is why it has its own lower default `T`.

## Parameter Updates

After inference settles the hidden states, the trainer computes two kinds of updates.

### Forward-parameter updates `ΔW`

`compute_W_updates(...)` differentiates the augmented energy with respect to the forward model leaves:

- conv weights
- head linear weights
- DyT parameters when enabled

Even though the code calls the augmented energy, the `stop_gradient(e^L)` design means the highway term contributes zero direct gradient to `W`. In effect, `ΔW` is the predictive-coding weight gradient evaluated at the inferred states.

### Highway updates `ΔV`

For each highway target `i`, the code forms a batch average outer product between flattened state `z^i` and output error `e^L`.

Default rule:

```text
rule = "energy"
ΔV_i = α * (1/B) Σ_b z^i_b (e^L_b)^T
```

Alternative rule:

```text
rule = "state"
ΔV_i = -α * (1/B) Σ_b z^i_b (e^L_b)^T
```

If `v_reg > 0`, the code adds:

```text
v_reg * V_i
```

to each update. This acts like an L2 penalty and helps keep `V` bounded.

### Fused update path

The variant also implements `compute_updates_fused(...)`, which returns:

- `e`
- `ΔW`
- `ΔV`
- per-layer energies

from one JITted pass.

This is an implementation optimization, not a different algorithm.

## Training Loop

The training loop is in `training/res_error_net_trainer.py`.

Per batch it does:

1. load a minibatch
2. optionally add Gaussian input noise
3. run a feedforward pass to initialize `z`
4. clamp the output activity to the target
5. run `T` inference steps on hidden states
6. compute `ΔW` and `ΔV`
7. optionally reproject each update leaf into a Gaussian ball
8. apply global-norm clipping to `ΔW`
9. apply optimizer updates
10. log feedforward loss and accuracy after the parameter update

### Stabilization choices in the trainer

- `reproject_c`: optional per-leaf reprojection for both `ΔW` and `ΔV`
- `global_clip_norm`: global norm clipping on `ΔW`
- `weight_decay`: used by AdamW when selected
- `input_noise_sigma`: optional Gaussian noise on inputs

The code comments explicitly note that for res-error-net experiments, global clipping is usually preferable to per-leaf reprojection because it preserves relative gradient magnitudes better.

## Reported Loss Versus Training Objective

This is another subtle but important point.

The actual training objective is the predictive-coding free energy above, minimized indirectly by:

- inferring hidden states
- then updating `W` and `V`

The `loss_type` setting (`"mse"` or `"ce"`) is only used for reporting:

- train loss is measured on a fresh feedforward pass after the update
- test accuracy is measured on a plain feedforward pass
- inference diagnostics compute loss/accuracy on `μ^{L-1} = head(z^{L-2})`, not on the clamped output state

So `res_resnet_loss` changes logging/evaluation, not the core predictive-coding training rule.

## Why DyT Is Used Instead Of BatchNorm

This variant supports:

- `normalization="dyt"`
- `normalization="none"`

DyT is:

```text
DyT(x) = gamma * tanh(alpha * x) + beta
```

with:

- scalar `alpha`
- per-channel `gamma`
- per-channel `beta`

The reason BatchNorm is avoided is architectural, not cosmetic:

- predictive-coding inference updates states repeatedly inside a single batch
- those evolving states do not match the assumptions behind running batch statistics
- a stateless per-example normalization is much easier to keep consistent across inference steps

So DyT is part of the model's training dynamics story, not just a normalization swap.

## Main Design Choices And Their Consequences

### 1. Real ResNet forward path

Using actual residual blocks gives a strong convolutional backbone instead of a toy CNN or MLP. The variant is trying to test residual-error predictive coding in a serious deep vision model.

### 2. Full-rank output-error highways

Each targeted activity gets a dense projection from output error into its flattened coordinates. This maximizes flexibility, but it is expensive.

### 3. Hard output clamp

The output activity is replaced by the target during inference. This makes the supervisory signal explicit and keeps the inference problem close to standard predictive-coding formulations.

### 4. Highway energy acts on states, not local errors

This means highways nudge the hidden activities themselves rather than transporting a local error object.

### 5. `stop_gradient` on the output error

This prevents the highway term from becoming a direct auxiliary loss on forward weights. The highways modify inference and learn their own projections, but do not directly rewrite the `W` update rule.

### 6. DyT instead of BatchNorm

The normalization choice is tailored to iterative inference, not standard feedforward training.

### 7. Separate inference optimizer

Allowing Euler or Adam-on-z makes inference itself a tunable optimization process. This is useful because stability of hidden-state settling is a major bottleneck in deep predictive-coding models.

## Default Config Surface

Important defaults from `config.py` for this variant:

| Setting | Default |
|---|---|
| `dataset` | `CIFAR10` is the intended use case |
| `res_resnet_alpha` | `1.0` |
| `res_resnet_channels` | `[64, 64, 128, 256, 512]` |
| `res_resnet_blocks_per_stage` | `2` |
| `res_resnet_normalization` | `dyt` |
| `res_resnet_dyt_init_alpha` | `0.5` |
| `res_resnet_highway_include_stem` | `True` |
| `res_resnet_inference_dt` | `0.1` |
| `res_resnet_inference_T` | `30` |
| `res_resnet_inference_method` | `"euler"` |
| `res_resnet_v_lr` | `1e-4` |
| `res_resnet_v_update_rule` | `"energy"` |
| `res_resnet_v_init_scale` | `0.01` |
| `res_resnet_optim` | `"adamw"` |
| `res_resnet_loss` | `"mse"` |
| `res_resnet_v_reg` | `0.1` |
| `reproject_c` | `1.0` |
| `global_clip_norm` | `10.0` |
| `input_noise_sigma` | `0.1` |
| `weight_decay` | `1e-4` |

Useful CLI flags in `run_training.py`:

```bash
--variant res_error_net_resnet18
--dataset CIFAR10
--res-resnet-alpha
--res-resnet-channels
--res-resnet-blocks-per-stage
--res-resnet-normalization
--res-resnet-dyt-init-alpha
--res-resnet-no-stem-highway
--res-resnet-inference-dt
--res-resnet-inference-T
--res-resnet-inference-method
--res-resnet-v-lr
--res-resnet-v-update-rule
--res-resnet-v-init-scale
--res-resnet-optim
--res-resnet-loss
--res-resnet-v-reg
--reproject-c
--global-clip-norm
--input-noise-sigma
--weight-decay
```

## Evaluation And Diagnostics

`evaluate(...)` uses a plain feedforward pass and reports top-1 accuracy from the logits.

So the headline test metric in training logs is:

- feedforward accuracy of the learned model
- not clamped-inference accuracy

The variant also exposes `diagnose_inference(...)`, which tracks:

- free energy across inference steps
- loss across inference steps
- accuracy across inference steps

Those diagnostics compute prediction quality from:

```text
μ^{L-1} = head(z^{L-2})
```

because the actual output activity is clamped and would otherwise give a trivial perfect score.

## Practical Caveats

### Runtime

This variant is expensive. Every training batch includes a forward initialization plus many predictive-coding inference steps through a ResNet-like backbone.

### Memory

The `V` matrices are largest exactly where spatial resolution is highest, so early highways are the main memory cost.

### Sensitivity

The model is sensitive to:

- inference step count
- inference step size
- highway strength
- choice of inference optimizer
- stabilization settings such as clipping and `V` regularization

### Documentation subtlety

Some comments elsewhere in the repo use older wording for the `V` update rule. The implementation in `variants/res_error_net_resnet18.py` is the authoritative behavior:

- highway energy uses `z^i`
- default `ΔV` uses `z^i` and `e^L`
- `stop_gradient(e^L)` is part of the design

## Minimal Mental Model

If you want the shortest accurate summary, it is this:

`res_error_net_resnet18` is a CIFAR-scale ResNet-18-style predictive-coding model where hidden convolutional states are iteratively inferred under an energy objective, while a bank of learnable dense highways projects output error directly into the stem and residual-block activities to improve long-range credit assignment.

## Implementation Map

- `variants/res_error_net_resnet18.py`
  Forward architecture, free energy, inference, `V` updates, evaluation, diagnostics.
- `training/res_error_net_trainer.py`
  Training loop, stabilization logic, optimizer application, logging.
- `run_training.py`
  CLI wiring and variant-specific config plumbing.
- `config.py`
  Defaults for channels, inference budget, optimizer, regularization, and normalization.
