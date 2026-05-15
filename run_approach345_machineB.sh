#!/usr/bin/env bash
# Approach 3/4/5 ablation sweep — MACHINE B (this box, RTX 5090, 1 GPU).
# Runs 8 cells in 4 waves of 2 parallel cells (both on the single GPU).
#
# Wall-clock estimate per user: 10 iter = 20 s with 2 cells in parallel
# → 2 s/iter/cell → 4000 iters ≈ 2.22 h/cell.
# 4 waves × 2.22 h ≈ 8.88 h total.
#
# Launch under tmux/nohup:
#   tmux new -d -s ablB 'bash /home/amirmhd/Documents/deep-pc/run_approach345_machineB.sh'

set +e   # one cell failing must not kill the parent

REPO=/home/amirmhd/Documents/deep-pc
TS=$(date +%Y%m%d_%H%M)
LOGDIR="$REPO/logs/approach345_${TS}"
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
PER_RUN_TIMEOUT=10000    # ~2.8 h, ~20% headroom over 2.22 h estimate
WANDB_PROJECT=ablation_alpha_v_lr

# ── Preflight ──────────────────────────────────────────────────────────────
python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" \
  || { echo 'ERROR: wandb not logged in. Run `wandb login` first.'; exit 1; }
python -c "from common.data import get_dataloaders; get_dataloaders('MNIST', 128)" \
  > "$LOGDIR/preflight_mnist.log" 2>&1 \
  || { echo 'ERROR: MNIST preflight failed. See preflight_mnist.log'; exit 1; }

printf 'cell_id\texit_code\tduration_s\tlog\n' > "$SUMMARY"

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

# ── Single-cell launcher (background; single GPU 0) ────────────────────────
# Args: CELL_ID  [extra CLI flags…]
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

# ── Wave 1: V frozen + cosine LR (default start) ──────────────────────────
run_wave "WAVE 1 (b3 frozen-V + c1 cosine 1e-4→1e-5)"
launch_cell b3_vfrozen_init0p05  --res-v-frozen --res-v-init-scale 0.05
sleep 2
launch_cell c1_lr_cos_1e4        --param-lr 1e-4 --param-lr-schedule cosine --param-lr-min 1e-5
wait
echo "[$(date +%H:%M:%S)] === WAVE 1 done ==="

# ── Wave 2: cosine LR (lower starts) ──────────────────────────────────────
run_wave "WAVE 2 (c2/c3 cosine LR variants)"
launch_cell c2_lr_cos_5e5        --param-lr 5e-5 --param-lr-schedule cosine --param-lr-min 5e-6
sleep 2
launch_cell c3_lr_cos_3e5        --param-lr 3e-5 --param-lr-schedule cosine --param-lr-min 3e-6
wait
echo "[$(date +%H:%M:%S)] === WAVE 2 done ==="

# ── Wave 3: highway sparsity 2, 3 ─────────────────────────────────────────
run_wave "WAVE 3 (d1/d2 highway every_k=2,3)"
launch_cell d1_every_k_2         --res-highway-every-k 2
sleep 2
launch_cell d2_every_k_3         --res-highway-every-k 3
wait
echo "[$(date +%H:%M:%S)] === WAVE 3 done ==="

# ── Wave 4: highway sparsity 5, 10 ────────────────────────────────────────
run_wave "WAVE 4 (d3/d4 highway every_k=5,10)"
launch_cell d3_every_k_5         --res-highway-every-k 5
sleep 2
launch_cell d4_every_k_10        --res-highway-every-k 10
wait
echo "[$(date +%H:%M:%S)] === WAVE 4 done ==="

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Summary: $SUMMARY"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
