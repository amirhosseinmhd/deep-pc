#!/bin/bash
# Run all μPC experiments
# Usage: bash exper/run_all.sh
# Assumes conda environment 'jax_env' is active

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== Experiment 1: SP ResNet Performance ==="
python exper/exp1_performance.py

echo ""
echo "=== Experiment 2: Condition Numbers ==="
python exper/exp2_condition.py

echo ""
echo "=== Experiment 3: Weight Updates ==="
python exper/exp3_weight_updates.py

echo ""
echo "=== Experiment 4: Latent Norms ==="
python exper/exp4_latent_norms.py

echo ""
echo "================================================"
echo "All experiments complete! Results in exper/results/"
echo "================================================"
