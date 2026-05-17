#!/usr/bin/env bash
# Clipping-mechanism sweep — MACHINE B (this box, RTX 5090, 1 GPU).
# Sweeps --global-clip-norm (Frobenius-norm clip on delta_W).
# 5 cells in 3 waves of {2, 2, 1} cells in parallel on the single GPU.
# Reproject-c stays at its 0 default.
#
# Wall-clock estimate: 6 k iters at ~2.0 s/iter when 2 cells share the 5090
#   → ~3.3 h/cell. 3 waves × 3.3 h ≈ 9.9 h total.
#
# Launch under tmux/nohup:
#   tmux new -d -s clipB 'bash /home/amirmhd/Documents/deep-pc/run_clip_sweep_machineB.sh'

set +e   # one cell failing must not kill the parent

REPO=/home/amirmhd/Documents/deep-pc
TS=$(date +%Y%m%d_%H%M)
LOGDIR="$REPO/logs/clip_sweep_${TS}"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"

# ── Environment ────────────────────────────────────────────────────────────
source /home/amirmhd/anaconda3/etc/profile.d/conda.sh
conda activate jax_env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$LOGDIR/jit_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$REPO"

MEM_FRAC=0.50            # 2 runs sharing the single 5090
PER_RUN_TIMEOUT=15000    # ~4.2 h, ~20% headroom over 3.3 h estimate
WANDB_PROJECT=ablation_clipping

# ── Preflight ──────────────────────────────────────────────────────────────
python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" \
  || { echo 'ERROR: wandb not logged in. Run `wandb login` first.'; exit 1; }
python -c "from common.data import get_dataloaders; get_dataloaders('MNIST', 128)" \
  > "$LOGDIR/preflight_mnist.log" 2>&1 \
  || { echo 'ERROR: MNIST preflight failed. See preflight_mnist.log'; exit 1; }

printf 'cell_id\texit_code\tduration_s\tlog\n' > "$SUMMARY"

# Base flags shared by every cell (matches approach345 a0_control recipe,
# n-iters bumped to 6 k per user request).
BASE=(
  --variant res_error_net
  --dataset MNIST
  --depths 25
  --act-fns relu
  --n-iters 6000
  --test-every 20
  --wandb-project "$WANDB_PROJECT"
  --res-loss ce
)

# ── Single-cell launcher (background; single GPU 0) ────────────────────────
launch_cell() {
  local CELL_ID=$1; shift
  local LOG="$LOGDIR/${CELL_ID}.log"
  (
    local START=$SECONDS
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
      timeout "$PER_RUN_TIMEOUT" python -u run_training.py \
        "${BASE[@]}" --wandb-run-name "$CELL_ID" "$@" \
      > "$LOG" 2>&1
    local CODE=$?
    local DUR=$(( SECONDS - START ))
    printf '%s\t%d\t%d\t%s\n' "$CELL_ID" "$CODE" "$DUR" "$LOG" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] <<< $CELL_ID  exit=$CODE dur=${DUR}s"
  ) &
}

run_wave() {
  local WAVE_NAME=$1
  echo ""
  echo "[$(date +%H:%M:%S)] === $WAVE_NAME starting ==="
}

# ── Wave 1: gc1, gc2 (tight clips) ────────────────────────────────────────
run_wave "WAVE 1 (global_clip_norm 0.01, 0.1)"
launch_cell gc1_0p01    --global-clip-norm 0.01
sleep 2
launch_cell gc2_0p1     --global-clip-norm 0.1
wait
echo "[$(date +%H:%M:%S)] === WAVE 1 done ==="

# ── Wave 2: gc3, gc4 (mid clips) ──────────────────────────────────────────
run_wave "WAVE 2 (global_clip_norm 1.0, 10.0)"
launch_cell gc3_1       --global-clip-norm 1.0
sleep 2
launch_cell gc4_10      --global-clip-norm 10.0
wait
echo "[$(date +%H:%M:%S)] === WAVE 2 done ==="

# ── Wave 3: gc5 (loose clip; solo) ────────────────────────────────────────
run_wave "WAVE 3 (global_clip_norm 100.0)"
launch_cell gc5_100     --global-clip-norm 100.0
wait
echo "[$(date +%H:%M:%S)] === WAVE 3 done ==="

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Summary: $SUMMARY"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
