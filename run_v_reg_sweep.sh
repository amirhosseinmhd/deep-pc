#!/usr/bin/env bash
# V-matrix study for res_error_net (MLP, MNIST, depth=18, tanh).
#
# Question: how much L2 penalty ρ should we put on V_{L→i}?
#   F_aug += (ρ/2)·Σ‖V‖²   ⇒   ΔV gets a +ρ·V shrinkage term.
#   ρ=0   : V drifts freely (can blow up, F_aug unbounded below in V).
#   ρ=1   : current default in config.py.
#   ρ→∞   : V pinned at init (≈ DFA-style frozen baseline).
#
# Each row holds everything else fixed and varies one V knob.
#
# Run under tmux so it survives terminal close:
#   tmux new -d -s vreg 'bash /home/amirmhd/Documents/deep-pc/run_v_reg_sweep.sh'

REPO=/home/amirmhd/Documents/deep-pc
LOGDIR="$REPO/logs/v_reg_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

# ── Environment ────────────────────────────────────────────────────────────
source /home/amirmhd/anaconda3/etc/profile.d/conda.sh
conda activate jax_env
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$REPO"

# ── Held-constant settings (mirror your reference command) ─────────────────
WANDB_PROJECT=V_REC_NEW
COMMON=(
  --variant res_error_net
  --dataset MNIST
  --depths 18
  --act-fns tanh
  --test-every 50
  --wandb-project "$WANDB_PROJECT"
  --n-iters 2000
)

# ── One-cell runner ────────────────────────────────────────────────────────
# Args: run_name  [extra CLI flags…]
run_cell() {
  local NAME=$1; shift
  local LOG="$LOGDIR/${NAME}.log"
  local START=$SECONDS
  echo "[$(date +%H:%M:%S)] >>> $NAME"
  timeout 2400 python -u run_training.py \
    "${COMMON[@]}" --wandb-run-name "$NAME" "$@" \
    > "$LOG" 2>&1
  local CODE=$?
  echo "[$(date +%H:%M:%S)] <<< $NAME  exit=$CODE  dur=$((SECONDS-START))s  log=$LOG"
}

set +e   # do not abort on a single-cell failure

# ── Sweep 1: V regularization ρ (the main question) ────────────────────────
# Span 4 decades around the current default of 1.
for RHO in 0 0.01 0.1 1 10 100; do
  run_cell "vreg_rho${RHO}" --res-v-reg "$RHO"
done

# ── Sweep 2: controls — does V learning matter at all? ─────────────────────
# Frozen V is the DFA-style baseline; if it matches the best ρ, V learning
# is doing no work.
run_cell "vreg_frozen"  --res-v-frozen
run_cell "vreg_no_v_lr" --res-v-lr 0          # learnable shape but zero step

# ── Sweep 3: init scale (does ρ interact with how big V starts?) ───────────
# Only meaningful if Sweep 1 shows ρ matters. Init=0.01 is the default.
for SCALE in 0.001 0.01 0.1; do
  run_cell "vreg_init${SCALE}" --res-v-init-scale "$SCALE"
done

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Logs: $LOGDIR"
echo "Compare runs in W&B project: $WANDB_PROJECT"
