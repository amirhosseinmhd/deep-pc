#!/usr/bin/env bash
# Clipping-mechanism sweep — MACHINE A (2× 24 GiB 3090s).
# Sweeps --reproject-c (per-leaf Gaussian-ball update radius).
# 5 cells in 2 waves (4 parallel + 1 solo). Both flags off elsewhere
# (global_clip_norm stays at its 0 default).
#
# Wall-clock estimate: 6 k iters at ~3.2 s/iter when 2 cells share a 3090
#   → ~5.25 h/cell. Wave 1 = 4 cells × 5.25 h; Wave 2 = 1 × 5.25 h.
#   Total ≈ 10.5 h.
#
# Place at /home/amirmhd/deep-pc/run_clip_sweep_machineA.sh on machine A.
# Launch under tmux/nohup:
#   tmux new -d -s clipA 'bash /home/amirmhd/deep-pc/run_clip_sweep_machineA.sh'

set +e   # one cell failing must not kill the parent

REPO=/home/amirmhd/deep-pc
TS=$(date +%Y%m%d_%H%M)
LOGDIR="$REPO/logs/clip_sweep_${TS}"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"

# ── Environment ────────────────────────────────────────────────────────────
source /home/amirmhd/miniforge3/etc/profile.d/conda.sh
conda activate jax_env
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$LOGDIR/jit_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$REPO"

MEM_FRAC=0.40            # 2 runs/GPU → ~40% memory each
PER_RUN_TIMEOUT=22500    # ~6.25 h, ~20% headroom over 5.25 h estimate
WANDB_PROJECT=ablation_clipping

# ── Preflight ──────────────────────────────────────────────────────────────
python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" \
  || { echo 'ERROR: wandb not logged in. Run `wandb login` first.'; exit 1; }
python -c "from common.data import get_dataloaders; get_dataloaders('MNIST', 128)" \
  > "$LOGDIR/preflight_mnist.log" 2>&1 \
  || { echo 'ERROR: MNIST preflight failed. See preflight_mnist.log'; exit 1; }

printf 'cell_id\tgpu\texit_code\tduration_s\tlog\n' > "$SUMMARY"

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

# ── Single-cell launcher (background) ──────────────────────────────────────
# Args: CELL_ID  GPU  [extra CLI flags…]
launch_cell() {
  local CELL_ID=$1; shift
  local GPU=$1;     shift
  local LOG="$LOGDIR/${CELL_ID}.log"
  (
    local START=$SECONDS
    CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC" \
      timeout "$PER_RUN_TIMEOUT" python -u run_training.py \
        "${BASE[@]}" --wandb-run-name "$CELL_ID" "$@" \
      > "$LOG" 2>&1
    local CODE=$?
    local DUR=$(( SECONDS - START ))
    printf '%s\t%s\t%d\t%d\t%s\n' \
      "$CELL_ID" "$GPU" "$CODE" "$DUR" "$LOG" >> "$SUMMARY"
    echo "[$(date +%H:%M:%S)] <<< $CELL_ID  gpu=$GPU exit=$CODE dur=${DUR}s"
  ) &
}

run_wave() {
  local WAVE_NAME=$1
  echo ""
  echo "[$(date +%H:%M:%S)] === $WAVE_NAME starting ==="
  wait    # ensure no leftover background jobs
}

# ── Wave 1: rp1..rp4 on GPUs 0,0,1,1 ──────────────────────────────────────
run_wave "WAVE 1 (reproject_c 0.001..1.0)"
launch_cell rp1_0p001   0   --reproject-c 0.001
sleep 2
launch_cell rp2_0p01    0   --reproject-c 0.01
sleep 2
launch_cell rp3_0p1     1   --reproject-c 0.1
sleep 2
launch_cell rp4_1       1   --reproject-c 1.0
wait
echo "[$(date +%H:%M:%S)] === WAVE 1 done ==="

# ── Wave 2: rp5 on GPU 0 (solo; the GPU is fully free now) ────────────────
run_wave "WAVE 2 (reproject_c 10.0)"
launch_cell rp5_10      0   --reproject-c 10.0
wait
echo "[$(date +%H:%M:%S)] === WAVE 2 done ==="

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Summary: $SUMMARY"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
