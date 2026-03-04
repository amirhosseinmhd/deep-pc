#!/bin/bash
# Run all BatchNorm Freezing v2 experiments
# Usage: conda activate jax_env && bash exper_batchFreezing_v2/run_bf_v2.sh

set -e
cd "$(dirname "$0")/.."

echo "Working directory: $(pwd)"
echo "================================================"

echo ""
echo "=== BatchNorm Freezing v2 Experiments ==="
echo "=== BN placement: act_fn -> BN(linear) + skip ==="
python exper_batchFreezing_v2/exp_bf_v2_all.py

echo ""
echo "================================================"
echo "All BF-v2 experiments complete! Results in exper_batchFreezing_v2/results/"
echo "================================================"
