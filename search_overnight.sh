#!/usr/bin/env bash
# search_overnight.sh — launch autonomous CNN-rec-LRA architecture search
#
# Usage:
#   ./search_overnight.sh              # shell loop, foreground (Ctrl-C to stop)
#   ./search_overnight.sh --tmux       # detached tmux session (survives terminal close)
#
# How it works:
#   A bash while-loop repeatedly calls `claude -p` (non-interactive / no-TTY mode).
#   Each call reads the current results_cnn.tsv, runs one batch of up to 3 parallel
#   experiments, appends results, then exits.  The loop immediately starts the next round.
#   `claude -p` is used because it doesn't require a real TTY (no Ink UI).
#
# Stop: Ctrl-C in foreground, or `tmux kill-session -t cnn_search_<tag>` for tmux.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_INIT="$HOME/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="jax_env"
PROGRAM_FILE="$REPO_DIR/program_cnn.md"
RESULTS_FILE="$REPO_DIR/results_cnn.tsv"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: conda not found at $CONDA_INIT"; exit 1
fi
if [ ! -f "$PROGRAM_FILE" ]; then
    echo "ERROR: $PROGRAM_FILE not found."; exit 1
fi
if ! command -v claude &>/dev/null; then
    echo "ERROR: 'claude' CLI not found in PATH."
    echo "Install Claude Code: https://claude.ai/code"
    exit 1
fi

# ── Activate conda ─────────────────────────────────────────────────────────────
# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate "$ENV_NAME"
echo "conda env: $(conda info --envs | grep '\*' | awk '{print $1}')"
python -c "import jax; print(f'JAX {jax.__version__} — devices: {jax.devices()}')"

# ── Git branch ─────────────────────────────────────────────────────────────────
TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')
BRANCH="hypersearch/cnn_${TAG}"
cd "$REPO_DIR"
if git rev-parse --verify "$BRANCH" &>/dev/null; then
    echo "Branch $BRANCH already exists — resuming."
    git checkout "$BRANCH"
else
    git checkout -b "$BRANCH"
    echo "Created branch: $BRANCH"
fi

# ── Results file ───────────────────────────────────────────────────────────────
if [ ! -f "$RESULTS_FILE" ]; then
    printf 'experiment\ttest_acc\ttrain_acc\tloss\tn_iters\tparam_lr\te_lr\tbeta\tcnn_channels\tn_fc_hidden\tfwd_skip\terr_skip\te_update\toptim\tstatus\tnotes\n' \
        > "$RESULTS_FILE"
    echo "Initialised: $RESULTS_FILE"
else
    N=$(( $(wc -l < "$RESULTS_FILE") - 1 ))
    echo "Resuming — $N experiment(s) already logged."
fi

# ── The search loop ────────────────────────────────────────────────────────────
# Each iteration calls `claude -p` (no-TTY print mode).
# The prompt = full program spec + current results + "do ONE round, then exit".
# Claude runs up to 3 experiments in parallel via Bash tool calls, then returns.

run_search_loop() {
    local ROUND=1
    echo ""
    echo "=== CNN architecture search loop started ($(date)) ==="
    echo "Program spec : $PROGRAM_FILE"
    echo "Results log  : $RESULTS_FILE"
    echo "Press Ctrl-C to stop."
    echo ""

    while true; do
        echo "--- Round $ROUND started at $(date '+%H:%M:%S') ---"

        # Build the current-state context
        CURRENT_RESULTS=$(cat "$RESULTS_FILE" 2>/dev/null || echo "(none yet)")
        NEXT_EXP=$(( $(wc -l < "$RESULTS_FILE") ))   # header counts as 1, so rows = lines-1+1

        PROMPT="$(cat "$PROGRAM_FILE")

---
## Current State (Round ${ROUND} — $(date '+%Y-%m-%d %H:%M'))

results_cnn.tsv (all experiments so far):
${CURRENT_RESULTS}

Next experiment number to assign: ${NEXT_EXP}

---
## Instructions for this round

Run ONE batch of up to 3 parallel experiments.  Choose configurations based on
the current results above (follow the phase logic in the program spec).
After ALL runs in this batch finish:
  1. Append each result as a new row to results_cnn.tsv.
  2. Print a one-line summary: 'Round ${ROUND} complete. Best so far: XX.XX%'
  3. Exit (do not start another batch — the outer loop handles iteration).

Use --no-weight-updates --no-activity-norms --no-grad-norms --no-layer-energy
on ALL runs to maximise speed (we only care about test accuracy here).
Always use --test-every 20 and --no-wandb.
"
        claude --dangerously-skip-permissions -p "$PROMPT"

        echo "--- Round $ROUND finished at $(date '+%H:%M:%S') ---"
        ROUND=$(( ROUND + 1 ))
        # Small pause so the filesystem flushes before the next round
        sleep 2
    done
}

# ── Entry point ────────────────────────────────────────────────────────────────
USE_TMUX=false
for arg in "$@"; do
    [[ "$arg" == "--tmux" ]] && USE_TMUX=true
done

if $USE_TMUX; then
    SESSION="cnn_search_${TAG}"
    if ! command -v tmux &>/dev/null; then
        echo "WARNING: tmux not installed. Running in foreground instead."
        run_search_loop
        exit 0
    fi

    # Export variables needed inside the tmux session
    export REPO_DIR CONDA_INIT ENV_NAME PROGRAM_FILE RESULTS_FILE TAG

    # Launch: tmux provides a real PTY so Ink would work if needed,
    # but we use claude -p anyway (no-TTY loop).
    tmux new-session -d -s "$SESSION" \
        "bash -c 'source $CONDA_INIT && conda activate $ENV_NAME && cd $REPO_DIR && bash $REPO_DIR/search_overnight.sh; exec bash'"

    echo ""
    echo "Search running in tmux session: $SESSION"
    echo "  Attach  : tmux attach -t $SESSION"
    echo "  Kill    : tmux kill-session -t $SESSION"
    echo "  Watch   : watch -n5 tail -n5 $RESULTS_FILE"
else
    run_search_loop
fi
