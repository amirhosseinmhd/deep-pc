#!/bin/bash
# Run all baseline (plain MLP, no skip) experiments
# Usage: bash exper_baseline/run_baseline.sh
# Assumes conda environment 'jax_env' is active

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== Baseline Experiment 1: MLP (no skip) Performance ==="
python exper_baseline/exp1_performance.py

echo ""
echo "=== Baseline Experiment 2: Condition Numbers ==="
python exper_baseline/exp2_condition.py

echo ""
echo "=== Baseline Experiment 3: Weight Updates ==="
python exper_baseline/exp3_weight_updates.py

echo ""
echo "=== Baseline Experiment 4: Latent Norms ==="
python exper_baseline/exp4_latent_norms.py

echo ""
echo "================================================"
echo "All baseline experiments complete! Results in exper_baseline/results/"
echo "================================================"
