#!/usr/bin/env bash
# Overnight ablation around `--res-dyt-norm post` (best config so far) — Machine B.
# Focus: highway / V structure + inference dynamics, all under post-DyT.
# Companion: run_dyt_ablation_machineA.sh (DyT mechanism + depth scaling).
#
# Held constant on every cell:
#   --variant res_error_net --dataset MNIST --act-fns relu
#   --depths 18 --res-dyt-norm post
#   --n-iters 2000 --test-every 20
# Each cell varies ONE knob.
#
# All runs land in W&B project `dyt_post_ablation`, named `machineB_<cell_id>`,
# so Machine A and Machine B aggregate in one place.
#
# 8 cells × ~70 min/cell ≈ 9.3 h, with timeout 5400s (90 min) per cell.
#
# Launch under tmux so SSH drops don't kill it:
#   tmux new -d -s dyt_b 'bash /home/amirmhd/Documents/deep-pc/run_dyt_ablation_machineB.sh'

REPO=/home/amirmhd/Documents/deep-pc
MACHINE_TAG=machineB
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

# ── Held-constant flags (mirror your dyt_post command + post-DyT) ──────────
WANDB_PROJECT=dyt_post_ablation
COMMON=(
  --variant res_error_net
  --dataset MNIST
  --act-fns relu
  --depths 18
  --n-iters 2000
  --test-every 20
  --res-dyt-norm post
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

# ── Highway / V structure (under post-DyT) ─────────────────────────────────
run_cell "b01_k_2"            "highway_every_k=2"          --res-highway-every-k 2
run_cell "b02_k_4"            "highway_every_k=4"          --res-highway-every-k 4
run_cell "b03_smode_dense"    "s_mode=dense"               --res-highway-s-mode dense
run_cell "b04_vreg_0"         "v_reg=0"                    --res-v-reg 0
run_cell "b05_vreg_10"        "v_reg=10"                   --res-v-reg 10
run_cell "b06_v_frozen"       "v_frozen_DFA"               --res-v-frozen

# ── Inference dynamics (under post-DyT) ────────────────────────────────────
run_cell "b07_T_15"           "inference_T=15"             --res-inference-T 15
run_cell "b08_T_60"           "inference_T=60"             --res-inference-T 60

echo ""
echo "=== MACHINE B DONE ($(date)) ==="
echo "Logs:    $LOGDIR"
echo "Summary: $SUMMARY"
echo "W&B:     project=$WANDB_PROJECT  (filter on names starting with '${MACHINE_TAG}_')"
echo ""
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
