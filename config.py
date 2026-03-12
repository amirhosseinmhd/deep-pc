"""Central configuration for all μPC experiments.

This is the single source of truth for all hyperparameters and constants.

Usage:
    from config import SEED, DEPTHS, ACTIVITY_LR, ...
    from config import ExperimentConfig
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
INPUT_DIM = 784
OUTPUT_DIM = 10
WIDTH = 128

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
ACTIVITY_LR = 5e-1
PARAM_LR = 1e-3
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

ALL_VARIANTS = [
    VARIANT_BASELINE, VARIANT_RESNET, VARIANT_BF,
    VARIANT_BF_V2, VARIANT_DYT, VARIANT_DYT_V2,
    VARIANT_DYT_V3, VARIANT_MUPC,
]


# ---------------------------------------------------------------------------
# Experiment configuration dataclass.  
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Full configuration for a single experiment run."""

    # Variant
    variant: str = VARIANT_RESNET

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
        if self.results_dir is None:
            self.results_dir = os.path.join(RESULTS_DIR, self.variant)

    @classmethod
    def from_variant(cls, variant, **overrides):
        """Create config with variant-specific defaults.

        All variants use the same learning rates.
        """
        return cls(variant=variant, **overrides)
