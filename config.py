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

ALL_VARIANTS = [
    VARIANT_BASELINE, VARIANT_RESNET, VARIANT_BF,
    VARIANT_BF_V2, VARIANT_DYT, VARIANT_DYT_V2,
    VARIANT_DYT_V3, VARIANT_MUPC, VARIANT_REC_LRA,
    VARIANT_CNN_REC_LRA, VARIANT_RES_ERROR_NET,
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
    reproject_c: float = 1.0          # Gaussian-ball update radius
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
    res_alpha: float = 1
    res_inference_T: int = 50
    res_inference_dt: float = 0.1
    res_v_lr: float = 1e-4
    res_v_update_rule: str = "energy"   # "energy" or "state"
    res_v_init_scale: float = 1
    res_output_clamp: str = "hard"       # soft reserved for future
    res_optim: str = "adamw"
    res_loss: str = "mse"
    res_init_scheme: str = "jpc_default"  # "jpc_default" or "unit_gaussian"
    # L2 penalty ρ on V_{L→i}. Adds (ρ/2)·Σ‖V‖² to F_aug so ΔV gains a +ρ·V
    # term, keeping V bounded and F bounded below in V.
    res_v_reg: float = 0.0

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
