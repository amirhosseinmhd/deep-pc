"""Central configuration for all μPC experiments.

This is the single source of truth for all hyperparameters and constants.
Both exper/common.py and experiment scripts import from here.

Usage:
    from config import SEED, DEPTHS, ACTIVITY_LR, ...
"""
import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "exper", "results")

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
ACTIVITY_LR = 1e-4
PARAM_LR = 5e-4
BATCH_SIZE = 64
TEST_EVERY = 100
N_TRAIN_ITERS = 2700

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
DEPTHS = [5, 10, 20,40]
ACT_FNS = ["relu"]
