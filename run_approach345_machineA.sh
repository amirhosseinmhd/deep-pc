#!/usr/bin/env bash
# Approach 3/4/5 ablation sweep — MACHINE A (2× 24 GiB GPUs).
# Runs 8 cells in 2 waves of 4 parallel cells (one per GPU × 2 cells/GPU).
#
# Wall-clock estimate: 2 waves × ~3 h each ≈ 6 h (machineB.sh is the
# bound at ~8.8 h; this one finishes earlier).
#
# Place at /home/amirmhd/deep-pc/run_approach345_machineA.sh on machine A.
# Launch under tmux/nohup:
#   tmux new -d -s ablA 'bash /home/amirmhd/deep-pc/run_approach345_machineA.sh'

set +e   # one cell failing must not kill the parent

REPO=/home/amirmhd/deep-pc
TS=$(date +%Y%m%d_%H%M)
LOGDIR="$REPO/logs/approach345_${TS}"
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
PER_RUN_TIMEOUT=15000    # ~4.2 h, ~20% headroom over 3.5 h estimate
WANDB_PROJECT=ablation_alpha_v_lr

# ── Preflight ──────────────────────────────────────────────────────────────
python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" \
  || { echo 'ERROR: wandb not logged in. Run `wandb login` first.'; exit 1; }
python -c "from common.data import get_dataloaders; get_dataloaders('MNIST', 128)" \
  > "$LOGDIR/preflight_mnist.log" 2>&1 \
  || { echo 'ERROR: MNIST preflight failed. See preflight_mnist.log'; exit 1; }

printf 'cell_id\tgpu\texit_code\tduration_s\tlog\n' > "$SUMMARY"

# Base flags shared by every cell.
BASE=(
  --variant res_error_net
  --dataset MNIST
  --depths 25
  --act-fns relu
  --n-iters 4000
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

# ── Wave 1: a0, a1, a2, a3 on GPUs 0,0,1,1 ────────────────────────────────
run_wave "WAVE 1 (alpha 0..3)"
launch_cell a0_control      0
sleep 2
launch_cell a1_alpha_0p3    0   --res-alpha 0.3
sleep 2
launch_cell a2_alpha_0p1    1   --res-alpha 0.1
sleep 2
launch_cell a3_alpha_0p03   1   --res-alpha 0.03
wait
echo "[$(date +%H:%M:%S)] === WAVE 1 done ==="

# ── Wave 2: a4, a5, b1, b2 on GPUs 0,0,1,1 ────────────────────────────────
run_wave "WAVE 2 (alpha 4-5 + V learnable)"
launch_cell a4_alpha_0p01   0   --res-alpha 0.01
sleep 2
launch_cell a5_alpha_cos    0   --res-alpha 1 --res-alpha-schedule cosine --res-alpha-min 0.01
sleep 2
launch_cell b1_vreg_low     1   --res-v-reg 0.01 --res-v-lr 1e-4
sleep 2
launch_cell b2_vreg_zero    1   --res-v-reg 0    --res-v-lr 1e-4
wait
echo "[$(date +%H:%M:%S)] === WAVE 2 done ==="

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Summary: $SUMMARY"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
