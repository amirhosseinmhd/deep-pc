#!/bin/bash
# Run all DyT (Dynamic Tanh) experiments
# Usage: bash exper_dyt/run_dyt.sh
# Assumes conda environment 'jax_env' is active

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== Dynamic Tanh (DyT) Experiments ==="
python exper_dyt/exp_dyt_all.py

echo ""
echo "================================================"
echo "All DyT experiments complete! Results in exper_dyt/results/"
echo "================================================"
