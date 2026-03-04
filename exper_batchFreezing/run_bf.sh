#!/bin/bash
# Run all BatchNorm Freezing experiments
# Usage: bash exper_batchFreezing/run_bf.sh
# Assumes conda environment 'jax_env' is active

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== BatchNorm Freezing Experiments ==="
python exper_batchFreezing/exp_bf_all.py

echo ""
echo "================================================"
echo "All BF experiments complete! Results in exper_batchFreezing/results/"
echo "================================================"
