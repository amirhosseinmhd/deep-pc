#!/bin/bash
# Run all DyT v2 (Dynamic Tanh) experiments
# Usage: conda activate jax_env && bash exper_dyt_v2/run_dyt_v2.sh

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== Dynamic Tanh v2 (DyT-v2) Experiments ==="
echo "=== DyT placement: act_fn -> DyT(linear) + skip ==="
python exper_dyt_v2/exp_dyt_v2_all.py

echo ""
echo "================================================"
echo "All DyT-v2 experiments complete! Results in exper_dyt_v2/results/"
echo "================================================"
