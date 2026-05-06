#!/usr/bin/env bash
# Overnight ablation around `--res-dyt-norm post` (best config so far) — Machine A.
# Focus: DyT mechanism (placement, alpha) + depth scaling.
# Companion: run_dyt_ablation_machineB.sh (highway/V structure + inference).
#
# Held constant on every cell:
#   --variant res_error_net --dataset MNIST --act-fns relu
#   --n-iters 2000 --test-every 20
# Default `--res-dyt-norm post` unless the cell varies it.
#
# All runs land in W&B project `dyt_post_ablation`, named `machineA_<cell_id>`,
# so Machine A and Machine B aggregate in one place.
#
# 8 cells × ~70 min/cell ≈ 9.3 h, with timeout 5400s (90 min) per cell.
#
# Launch under tmux so SSH drops don't kill it:
#   tmux new -d -s dyt_a 'bash /home/amirmhd/Documents/deep-pc/run_dyt_ablation_machineA.sh'

REPO=/home/amirmhd/Documents/deep-pc
MACHINE_TAG=machineA
LOGDIR="$REPO/logs/dyt_ablation_${MACHINE_TAG}_$(date +%Y%m%d_%H%M)"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"

# ── Environment ────────────────────────────────────────────────────────────
source /home/amirmhd/anaconda3/etc/profile.d/conda.sh
conda activate jax_env
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$REPO"

# ── Preflight: wandb logged in? ────────────────────────────────────────────
python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" \
  || { echo 'ERROR: wandb not logged in. Run `wandb login` first.'; exit 1; }

# ── Preflight: pre-warm MNIST ──────────────────────────────────────────────
python -c "from common.data import get_dataloaders; get_dataloaders('MNIST', 128)" \
  > "$LOGDIR/preflight_mnist.log" 2>&1 \
  || { echo 'ERROR: MNIST preflight failed. See preflight_mnist.log'; exit 1; }

# ── Held-constant flags (mirror your dyt_post command) ─────────────────────
WANDB_PROJECT=dyt_post_ablation
COMMON=(
  --variant res_error_net
  --dataset MNIST
  --act-fns relu
  --n-iters 2000
  --test-every 20
  --wandb-project "$WANDB_PROJECT"
)

# ── Summary header ─────────────────────────────────────────────────────────
printf 'cell_id\tvaried\texit_code\tduration_s\tbest_test_acc\tlog\n' > "$SUMMARY"

# ── One-cell runner ────────────────────────────────────────────────────────
# Args: cell_id  varied_label  [extra CLI flags…]
run_cell() {
  local CELL_ID=$1; shift
  local VARIED=$1;  shift
  local LOG="$LOGDIR/${CELL_ID}.log"
  local NAME="${MACHINE_TAG}_${CELL_ID}"
  local START=$SECONDS

  echo "[$(date +%H:%M:%S)] >>> $CELL_ID  varied=$VARIED"

  timeout 5400 python -u run_training.py \
    "${COMMON[@]}" --wandb-run-name "$NAME" "$@" \
    > "$LOG" 2>&1
  local CODE=$?
  local DUR=$(( SECONDS - START ))

  # Best test acc grepped from "Iter N, loss=..., train acc=..., test acc=NN.NN"
  local BEST
  BEST=$(grep -oE 'test acc=[0-9]+\.[0-9]+' "$LOG" \
         | awk -F= '{print $2}' | sort -gr | head -1)
  [[ -z "$BEST" ]] && BEST="NA"

  printf '%s\t%s\t%d\t%d\t%s\t%s\n' \
    "$CELL_ID" "$VARIED" "$CODE" "$DUR" "$BEST" "$LOG" >> "$SUMMARY"
  echo "[$(date +%H:%M:%S)] <<< $CELL_ID  exit=$CODE dur=${DUR}s best_test_acc=$BEST"
}

set +e   # do not abort on a single-cell failure

# ── DyT mechanism (placement, alpha) ───────────────────────────────────────
run_cell "a01_baseline"          "dyt_post_d18_anchor"      --depths 18 --res-dyt-norm post
run_cell "a02_dyt_off"           "dyt_off"                  --depths 18 --res-dyt-norm off
run_cell "a03_dyt_pre"           "dyt_pre"                  --depths 18 --res-dyt-norm pre
run_cell "a04_dyt_alpha_0p1"     "dyt_post_alpha=0.1"       --depths 18 --res-dyt-norm post --res-dyt-init-alpha 0.1
run_cell "a05_dyt_alpha_2p0"     "dyt_post_alpha=2.0"       --depths 18 --res-dyt-norm post --res-dyt-init-alpha 2.0

# ── Depth scaling under post-DyT (and one no-DyT control) ──────────────────
run_cell "a06_depth_12"          "depth=12_dyt_post"        --depths 12 --res-dyt-norm post
run_cell "a07_depth_24"          "depth=24_dyt_post"        --depths 24 --res-dyt-norm post
run_cell "a08_depth_24_dyt_off"  "depth=24_dyt_off_control" --depths 24 --res-dyt-norm off

echo ""
echo "=== MACHINE A DONE ($(date)) ==="
echo "Logs:    $LOGDIR"
echo "Summary: $SUMMARY"
echo "W&B:     project=$WANDB_PROJECT  (filter on names starting with '${MACHINE_TAG}_')"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
