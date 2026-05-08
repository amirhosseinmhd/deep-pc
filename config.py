"""Central configuration for all μPC experiments.

This is the single source of truth for all hyperparameters and constants.

Usage:
    from config import SEED, DEPTHS, ACTIVITY_LR, ...
    from config import ExperimentConfig
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

from common.data import get_input_dim

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# Dataset & model architecture
# ---------------------------------------------------------------------------
DATASET = "MNIST"
INPUT_DIM = get_input_dim(DATASET)  # 784 for MNIST, 3072 for CIFAR10
OUTPUT_DIM = 10
WIDTH = 128

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
ACTIVITY_LR = 5e-1
PARAM_LR = 1e-4
BATCH_SIZE = 128
TEST_EVERY = 1
N_TRAIN_ITERS = 1000

# ---------------------------------------------------------------------------
# Condition number experiment
# ---------------------------------------------------------------------------
COND_WIDTH = 16

# ---------------------------------------------------------------------------
# Variant name constants
# ---------------------------------------------------------------------------
VARIANT_BASELINE = "baseline"
VARIANT_RESNET = "resnet"
VARIANT_BF = "bf"
VARIANT_BF_V2 = "bf_v2"
VARIANT_DYT = "dyt"
VARIANT_DYT_V2 = "dyt_v2"
VARIANT_DYT_V3 = "dyt_v3"
VARIANT_MUPC = "mupc"
VARIANT_REC_LRA = "rec_lra"
VARIANT_CNN_REC_LRA = "cnn_rec_lra"
VARIANT_RES_ERROR_NET = "res_error_net"
VARIANT_RES_ERROR_NET_RESNET18 = "res_error_net_resnet18"
VARIANT_RES_ERROR_NET_SIMPLE_CNN = "res_error_net_simple_cnn"

ALL_VARIANTS = [
    VARIANT_BASELINE, VARIANT_RESNET, VARIANT_BF,
    VARIANT_BF_V2, VARIANT_DYT, VARIANT_DYT_V2,
    VARIANT_DYT_V3, VARIANT_MUPC, VARIANT_REC_LRA,
    VARIANT_CNN_REC_LRA, VARIANT_RES_ERROR_NET,
    VARIANT_RES_ERROR_NET_RESNET18,
    VARIANT_RES_ERROR_NET_SIMPLE_CNN,
]


# ---------------------------------------------------------------------------
# Experiment configuration dataclass.  
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Full configuration for a single experiment run."""

    # Variant
    variant: str = VARIANT_RESNET

    # Dataset
    dataset: str = DATASET

    # Architecture
    input_dim: int = INPUT_DIM
    output_dim: int = OUTPUT_DIM
    width: int = WIDTH
    depths: List[int] = field(default_factory=lambda: [4])
    act_fns: List[str] = field(default_factory=lambda: ["relu"])

    # Training
    seed: int = SEED
    activity_lr: float = ACTIVITY_LR
    param_lr: float = PARAM_LR
    batch_size: int = BATCH_SIZE
    n_train_iters: int = N_TRAIN_ITERS
    test_every: int = TEST_EVERY
    inference_multiplier: float = 4.0
    activity_init: str = "ffwd"  # "ffwd" or "zeros"
    param_optim_type: str = "adam"  # "adam" or "sgd"

    # Metrics
    track_weight_updates: bool = True
    track_activity_norms: bool = True
    track_grad_norms: bool = True
    track_layer_energy: bool = True

    # DyT-specific
    init_alpha: float = 0.5
    activity_noise: float = 0.0

    # rec-LRA specific (defaults from Ororbia & Mali 2023, p.15)
    forward_skip_every: int = 2
    error_skip_every: int = 2

    beta: float = 0.1205
    gamma_E: float = 0.1524
    e_lr: float = 1e-4
    rec_lra_optim: str = "adamw"
    rec_lra_loss: str = "mse"
    rec_lra_e_update: str = "grad"   # "hebbian" (Eq.6) or "grad" (rLRA-dx)
    # α=1.0 = pure L2/MSE error neurons. Paper uses 0.19/0.24 with
    # softmax+CE; with raw MSE the sign(z-y) term dominates and stalls
    # training. Pass --alpha-e-skip 0.19 --alpha-e-adj 0.24 to opt back in.
    alpha_e_skip: float = 1.0
    alpha_e_adj: float = 1.0
    reproject_c: float = 0        # Per-leaf Gaussian-ball update radius (0 disables)
    global_clip_norm: float = 0    # Global-norm clipping threshold for delta_W (res-error-net only; 0 disables)
    input_noise_sigma: float = 0.1
    weight_decay: float = 1e-4
    use_layer_norm: bool = True
    # GCN+ZCA hurts in our raw-MSE setup (29.5% → 19.4% at 400 iters).
    # Re-enable with --use-zca for paper-fidelity comparisons.
    use_zca: bool = False

    # CNN-rec-LRA specific
    # Default: 7 conv + 1 FC hidden + output = 9 total layers
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 128, 128, 256, 256])
    cnn_fc_width: int = 512
    n_fc_hidden: int = 1
    kernel_size: int = 3

    # res-error-net specific
    res_highway_every_k: int = 3
    # Optional forward residual skip interval for the MLP res-error-net.
    # 0 disables skips; n>0 adds z^{l-n} into the prediction of layer l when
    # dimensions match.
    res_forward_skip_every: int = 0
    res_alpha: float = 1
    res_inference_T: int = 100
    res_inference_dt: float = 0.005
    # "euler" → plain gradient flow ż = -∂F/∂z (sensitive to dt).
    # "adam"  → per-coordinate adaptive Adam-on-z; dt is treated as a
    # learning rate, much more robust to its value.
    res_inference_method: str = "adam"
    res_v_lr: float = 1e-5
    res_v_update_rule: str = "state"   # "energy" or "state"
    res_v_init_scale: float = 0.01
    res_output_clamp: str = "hard"       # soft reserved for future
    res_optim: str = "adamw"
    res_loss: str = "mse"
    res_init_scheme: str = "jpc_default"  # "jpc_default" or "unit_gaussian"
    # "sp" (standard) or "mupc" (μPC). When "mupc", per-layer scalings
    # (1/√D, 1/√N, …, 1/N) are applied in the forward pass and weights are
    # re-initialised as N(0, 1); this overrides res_init_scheme.
    res_param_type: str = "sp"
    # L2 penalty ρ on V_{L→i}. Adds (ρ/2)·Σ‖V‖² to F_aug so ΔV gains a +ρ·V
    # term, keeping V bounded and F bounded below in V.
    res_v_reg: float = 1

    # Ablation knobs
    res_alpha_schedule: str = "fixed"      # "fixed" | "linear" | "cosine"
    res_alpha_min: float = 0.0             # endpoint of decay schedules
    res_v_frozen: bool = False             # DFA-style: V never updated
    res_highway_s_mode: str = "stride"     # "stride" | "dense" | "sparse" | "random"

    # DyT (Dynamic Tanh) normalization for the MLP res-error-net.
    # "off" → no DyT (default; bit-exact with prior runs).
    # "pre" → DyT applied to a layer's input (before the linear).
    # "post" → DyT applied to the layer output (after the activation).
    res_dyt_norm: str = "off"
    res_dyt_init_alpha: float = 0.5
    # Which layers get a DyT module. "hidden" = layers 1..L-2; "all_internal"
    # = layers 0..L-2. The output layer L-1 is always excluded (z[L-1] is
    # hard-clamped to y, so DyT'ing the prediction would distort supervision).
    res_dyt_layers: str = "hidden"

    # res-error-net-resnet18 specific (CIFAR-10 ResNet-18 backbone)
    # Strength of the highway term in the augmented free energy. Larger values
    # make the output-error shortcuts influence hidden-state inference more.
    res_resnet_alpha: float = 1
    # `res_resnet_channels` = [stem_C, stage1_C, stage2_C, stage3_C, stage4_C];
    # blocks_per_stage=2 → 8 basic blocks (ResNet-18 layout).
    # Example: [64, 64, 128, 256, 512] = 64-channel stem, then 4 stages.
    res_resnet_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 256, 512])
    # Number of residual BasicBlocks in each stage. With 4 stages and value 2,
    # this gives the usual CIFAR-style ResNet-18 backbone.
    res_resnet_blocks_per_stage: int = 2
    # "dyt" → DyT(x)=γ·tanh(α·x)+β per channel (stateless, per-example);
    # "none" → identity. No BatchNorm: running stats go stale during T-step
    # iterative inference.
    res_resnet_normalization: str = "dyt"
    # Initial scalar α inside each DyT layer. Higher values make DyT saturate
    # faster; lower values keep it closer to linear near the origin.
    res_resnet_dyt_init_alpha: float = 0.5
    # Whether z_s (stem output) gets a V_{L→s} highway in addition to the 8
    # block-output highways.
    res_resnet_highway_include_stem: bool = True
    # Euler step size for predictive-coding inference on hidden states z.
    # Larger values move states faster each step but can destabilize dynamics.
    res_resnet_inference_dt: float = 0.1
    # CNN-specific T override. Each inference step on a ResNet-18 is ~100×
    # more expensive than on the MLP variant, so the MLP default (50) is
    # wasteful here. This is the number of inference steps per training batch.
    res_resnet_inference_T: int = 30
    # "euler" → plain gradient flow ż = -∂F/∂z (sensitive to dt).
    # "adam"  → per-coordinate adaptive Adam-on-z; dt is treated as a learning
    # rate, much more robust to its value. Same hand-rolled Adam as the MLP
    # variant, applied per-coordinate to (B, C, H, W) z tensors.
    res_resnet_inference_method: str = "euler"
    # Learning rate for the highway matrices V_{L→i}.
    res_resnet_v_lr: float = 1e-4
    # How V is updated:
    # "energy" = gradient-style update from the highway energy term,
    # "state"  = alternative Hebbian/state-based rule.
    res_resnet_v_update_rule: str = "energy"   # "energy" or "state"
    # Standard deviation multiplier for initializing the highway matrices V.
    # Smaller values weaken the shortcut signal at initialization.
    res_resnet_v_init_scale: float = 0.01
    # Optimizer used for both model weights and V matrices in this variant.
    res_resnet_optim: str = "adamw"
    # Loss used only for reporting/evaluation during training:
    # "mse" compares logits to one-hot targets, "ce" uses cross-entropy.
    res_resnet_loss: str = "mse"
    # ResNet-specific L2 penalty on V_{L→i}. Kept separate from `res_v_reg`
    # so the CNN variant can be stabilized without affecting the MLP one.
    # Larger values shrink V more aggressively and keep highway norms bounded.
    res_resnet_v_reg: float = 0.1

    # res-error-net-simple-cnn specific (plain conv-stack backbone). Uses the
    # same Stem/Head modules and DyT normalization as the ResNet-18 sibling,
    # but the forward path is a chain of single-conv blocks (no in-block
    # residual). Designed to clear MNIST 95%+ at 3 conv blocks and scale to
    # deeper CIFAR-10 configurations by extending `res_simple_cnn_channels`.
    # Trainer-side knobs (alpha, dt, v_lr, optim, loss, v_reg, v_init_scale)
    # are shared with the MLP variant via the existing `res_*` fields — the
    # entries below cover only architecture, not training.
    # Channel list — first entry is the stem, the rest are conv blocks.
    res_simple_cnn_channels: List[int] = field(
        default_factory=lambda: [16, 32, 64]
    )
    # Per-block stride. None ⇒ stem at stride 1, every subsequent block at
    # stride 2 (canonical halving layout). When provided, must match the
    # length of `res_simple_cnn_channels`.
    res_simple_cnn_strides: Optional[List[int]] = None
    res_simple_cnn_kernel_size: int = 3
    # Auto-derived from dataset when None: MNIST/FashionMNIST → (1,28,28),
    # CIFAR10 → (3,32,32). Override only for non-standard inputs.
    res_simple_cnn_input_shape: Optional[List[int]] = None
    res_simple_cnn_normalization: str = "dyt"          # "dyt" or "none"
    res_simple_cnn_dyt_init_alpha: float = 0.5
    # "post" (conv → act → DyT) tracks the user-validated MLP recipe
    # (--res-dyt-norm post). "pre" puts DyT before the activation.
    res_simple_cnn_dyt_position: str = "post"
    # "flatten" (default): full Linear(C·H·W → output_dim). Required for the
    # backward pull from e^L to z^{L-2} to be strong enough that T-step
    # inference can settle hidden states. "gap" uses GAP+Linear (fewer
    # params, but pull is divided by H·W).
    res_simple_cnn_head_type: str = "flatten"
    res_simple_cnn_highway_include_stem: bool = True
    # T-step inference loop length. CNN steps are ~10× costlier than the MLP
    # equivalent; 20 is a reasonable starting point for MNIST. Override of
    # the shared `res_inference_T` for CNN cost reasons.
    res_simple_cnn_inference_T: int = 20
    # Adam-on-z (matches the MLP variant's default and the user-validated MNIST
    # recipe). With Euler and the shared `res_inference_dt`=0.005 the per-step
    # movement is too small for the conv state to settle in T=20 steps.
    res_simple_cnn_inference_method: str = "adam"      # "euler" or "adam"
    # 0.01 matches the MLP variant's default and the v_nonlearnable report's
    # recommendation. With this scale the CNN reaches ~93.5% on MNIST in
    # 3000 iters under both `--res-v-frozen` and learnable V (within seed
    # jitter, mirroring the MLP report's headline). Larger init (e.g. 0.1)
    # gives faster early learning but plateaus around 89% — same pathology
    # as the report's group 4 result.
    res_simple_cnn_v_init_scale: float = 0.01

    # Condition number experiment
    cond_width: int = COND_WIDTH

    # W&B
    use_wandb: bool = True
    wandb_project: str = "pcn-experiments"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Output
    results_dir: Optional[str] = None

    def __post_init__(self):
        # Derive input_dim from dataset
        self.input_dim = get_input_dim(self.dataset)
        if self.results_dir is None:
            self.results_dir = os.path.join(RESULTS_DIR, self.variant)

    @classmethod
    def from_variant(cls, variant, **overrides):
        """Create config with variant-specific defaults.

        All variants use the same learning rates.
        """
        return cls(variant=variant, **overrides)
