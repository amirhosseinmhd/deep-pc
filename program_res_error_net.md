# `res_error_net_resnet18` — Autonomous Hyperparameter Search

Autonomous agent program for finding good hyperparameters for the
`res_error_net_resnet18` variant on CIFAR-10, with the ultimate goal of
reaching **backprop-competitive** accuracy (>90% top-1, ResNet-18).

## Goal

**Maximize CIFAR-10 test accuracy for `res_error_net_resnet18`.**

### Tiered milestones
| Milestone | Meaning |
|-----------|---------|
| > 25% | Algorithm learning — above noise floor (random = 10%) |
| > 45% | Working — beats prior `cnn_rec_lra` runs in this repo |
| > 65% | Strong — competitive with Ororbia & Mali 2023 PC result |
| > 80% | Excellent — approaching BP territory |
| > 90% | **Stretch** — competitive with BP baseline (~91–92%) |

Current best known: unknown at start; use the baseline row in
`results_res_error_net.tsv` as the initial `current_best`.

## Setup (first round only)

1. **Read these files** for full context:
   - `config.py` (see lines ~156–173 for res_resnet defaults)
   - `variants/res_error_net_resnet18.py`
   - `training/res_error_net_trainer.py`
   - `run_training.py` (CLI flags around L219–433; dispatch at L72–129)

2. **Verify environment:**
   ```bash
   cd /home/amirmhd/Documents/deep-pc
   source /home/amirmhd/anaconda3/etc/profile.d/conda.sh && conda activate jax_env
   python -c "import jax; print(jax.__version__, jax.devices())"
   ```
   Confirm a CUDA device (not just CPU) is listed — STOP if CPU-only.

3. **Initialize `results_res_error_net.tsv`** (skip if it already exists with rows):
   The header is one tab-separated line (see schema below).

4. **Run Phase 0 baseline** (Experiment 1) exactly as specified below, then
   begin the loop immediately — no confirmation pause.

## Fixed execution environment

The shell loop already exports these; do not override:

```
N_PARALLEL=1                            # SERIAL only — never launch a second concurrent run
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85     # full GPU for the single worker
XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_COMPILATION_CACHE_DIR=$REPO/.jax_cache   # critical: warm hits avoid the cold JIT
```

The **compile cache** is the single biggest throughput lever. JIT cache keys
are derived from input shapes — so to maximize cache hits, **fix batch_size=32**
for all search runs (see Tier B below — it is removed from the sampled space).

**Strict serialism:** launch exactly ONE python subprocess per agent
invocation, run it foreground, wait for it to exit, then (if the promotion
rule fires) launch a SECOND python subprocess on the same config. Never
background with `&`, never use `wait` to manage a pool, never run two python
processes at once.

Run every experiment with these **fixed CLI flags**:
```
--variant res_error_net_resnet18
--dataset CIFAR10
--depths 1
--act-fns relu
--no-wandb
--no-weight-updates --no-activity-norms --no-grad-norms --no-layer-energy
--batch-size 32
--res-resnet-optim adam
--test-every 50
```
(Tracking flags disabled for speed during search. `batch-size 32` is FIXED —
do not put it in the search space, as varying batch size busts the JAX
compile cache. `--res-resnet-optim adam` is FIXED — Adam was chosen as the
search optimizer; do not sample it. `--test-every 50` gives ~18 eval points
during a 900-iter pre-screen and ~60 during a 3000-iter full run.)

## Timing (re-calibrated for new architecture, GPU)

Measured throughput on the new ResNet-18 architecture, batch=32, GPU:
- **~25 iterations / minute** (≈ 2.4 s/iter) at the default `inference_T`.
- 900-iter pre-screen ≈ **36 min**.
- 3000-iter full run ≈ **120 min**.
- Worst case per agent invocation = pre-screen + full = **~2.5 h**.

Hard timeout per subprocess (set with `timeout`):
- Pre-screen (900 iters): **3000 s** (50 min — covers JIT cold start + slack).
- Full run (3000 iters): **9000 s** (150 min — covers cold compile + slack).
- If `n_iters` is bumped above 3000 (see convergence rule below), scale the
  timeout proportionally: `timeout = ceil(n_iters * 2.6) + 600` seconds.

**Phase 0 timing sanity check** (run once when TSV is empty):
```
sec_per_iter (warm) at chosen inference_T : ___   (expect ~2–3 s)
JIT first-compile overhead                : ___ s
Wall-clock for 900-iter pre-screen        : ___   (expect ~35–45 min)
```

If sec/iter is more than 2× the expectation, stop and report — likely a
device/regression issue. If it is significantly faster, just record it.

## Parallelism & agent wake-up cadence

- **N_PARALLEL = 1**. There is no parallelism. Ever. One python subprocess
  at a time per agent invocation.
- **One round = ONE experiment** = one pre-screen + (conditionally) one full
  run on the same config. 1 or 2 TSV rows added per invocation.
- Target round wall time: ~36 min if pre-screen fails the promotion rule,
  ~2.5 h if it passes.
- The agent **must exit** as soon as the experiment finishes. The bash outer
  loop is the iterator and will immediately re-invoke the agent for the next
  config — that's how exploration progresses, not by looping inside one call.

## What you can change (search space)

A random trial samples **all tiers below simultaneously**. A perturbation
trial (the 30% branch in "New-config sampling" below) starts from a known-good
config and jitters a subset.

### Tier A — training dynamics (every trial)
| Parameter | CLI flag | Sampling |
|---|---|---|
| Inference steps | `--res-resnet-inference-T` | choice `{5, 10, 20, 50, 100}` — small set to reuse JIT cache |
| W learning rate | `--param-lr` | log-uniform `[1e-5, 3e-3]` |
| V learning rate | `--res-v-lr` | log-uniform `[1e-5, 3e-3]` |
| Highway coupling | `--res-alpha` | log-uniform `[0.05, 5.0]` |
| Inference dt | `--res-inference-dt` | choice `{0.05, 0.1, 0.2}` |
| V update rule | `--res-v-update-rule` | choice `{energy, state}` |
| V init scale | `--res-v-init-scale` | log-uniform `[1e-3, 1e-1]` |
| Loss | `--res-loss` | choice `{mse, ce}` |

(T is a JIT cache key — keep its set small to maximize warm cache hits.
Each unique (T, normalization, v_update, loss, stem) tuple triggers a fresh
compile; the cache then amortizes subsequent runs.)

### Tier B — optimizer
| Parameter | CLI flag | Sampling |
|---|---|---|
| ~~Optimizer~~ | `--res-resnet-optim` | **FIXED at `adam`** (do NOT sample) |
| Weight decay | `--weight-decay` | log-uniform `[1e-6, 1e-2]` |
| ~~Batch size~~ | `--batch-size` | **FIXED at 32** (cache key — do NOT vary) |

### Tier C — regularization
| Parameter | CLI flag | Sampling |
|---|---|---|
| V reg | `--res-v-reg` | 50% `0.0`, 50% log-uniform `[1e-5, 1e-2]` |
| Reproject radius | `--reproject-c` | choice `{0.5, 1.0, 2.0, 1e9}` (1e9 ≈ off) |
| Input noise | `--input-noise-sigma` | choice `{0.0, 0.05, 0.1, 0.2}` |

### Tier D — architecture stabilization
| Parameter | CLI flag | Sampling |
|---|---|---|
| Normalization | `--res-resnet-normalization` | choice `{dyt, none}` |
| DyT init α | `--res-resnet-dyt-init-alpha` | choice `{0.3, 0.5, 0.8, 1.0}` (only if normalization=dyt) |
| Stem highway | `--res-resnet-no-stem-highway` | 50% present (disables), 50% absent |

### Fixed (never change)
- ResNet-18 channels `[64,64,128,256,512]` (omit `--res-resnet-channels`)
- `--res-resnet-blocks-per-stage` default = 2 (omit)
- `--dataset CIFAR10`, `--depths 1`, `--act-fns relu`
- `--res-resnet-optim adam` (FIXED — Adam was chosen as the search optimizer)
- `--seed 42` during search. (Multi-seed re-runs are an optional final step
  on the single best config so far; see "Multi-seed verification" below.)

## Running experiments

### Template — single foreground run (use this exact shape for both pre-screen and full)
```bash
cd /home/amirmhd/Documents/deep-pc
source /home/amirmhd/anaconda3/etc/profile.d/conda.sh && conda activate jax_env

timeout <stage_timeout> python -u run_training.py \
    --variant res_error_net_resnet18 \
    --dataset CIFAR10 \
    --depths 1 \
    --act-fns relu \
    --no-wandb \
    --no-weight-updates --no-activity-norms --no-grad-norms --no-layer-energy \
    --batch-size 32 \
    --res-resnet-optim adam \
    --test-every 50 \
    --seed 42 \
    --n-iters <900 for pre-screen, ≥3000 for full> \
    --param-lr <sampled> \
    --res-v-lr <sampled> \
    --res-alpha <sampled> \
    --res-resnet-inference-T <sampled> \
    --res-inference-dt <sampled> \
    --res-v-update-rule <sampled> \
    --res-v-init-scale <sampled> \
    --res-loss <sampled> \
    --weight-decay <sampled> \
    --res-v-reg <sampled> \
    --reproject-c <sampled> \
    --input-noise-sigma <sampled> \
    --res-resnet-normalization <sampled> \
    --res-resnet-dyt-init-alpha <sampled if norm=dyt> \
    [--res-resnet-no-stem-highway]   # include with 50% probability
    > run_${trial_id}.log 2>&1

# CRITICAL: NO trailing '&'. The python process runs in the foreground and
# the agent waits for it to exit before doing ANYTHING else. ALWAYS use
# `python -u` (unbuffered) and `2>&1` so the log flushes line-by-line.
```

Stage timeouts:

| Stage      | `--n-iters` | `timeout` (s) | Reason |
|------------|-------------|---------------|--------|
| Pre-screen | **900**     | **3000**      | ~36 min run + ~14 min slack for JIT cold start |
| Full       | **3000**    | **9000**      | ~120 min run + ~30 min slack |
| Bumped full| `n` (>3000) | `ceil(n*2.6) + 600` | scales linearly with iters |

NEVER use `&` to background a run. NEVER launch a second python process while
the first is still running. The whole orchestration is one foreground command
at a time.

## Reading results

```bash
echo "=== run_${trial_id}.log ===" && grep "test acc=" run_${trial_id}.log | tail -10
grep -iE "diverged|nan|inf|error|traceback" run_${trial_id}.log 2>/dev/null | head
```

Output format (MetricsCollector via `training/res_error_net_trainer.py`):
```
  Iter 200, loss=0.1234, train acc=85.00, test acc=72.50
```

### Parsing rules
- Collect ALL `test acc=` lines from the log (one per `test_every` = 50 iters).
- `peak_acc` = max of the collected values.
- `peak_iter` = the `Iter N` line where that max occurred.
- `final_acc` = last collected value.
- `final_loss` = the `loss=...` value on the last `Iter N, loss=...` line.
- If `nan` / `inf` anywhere in the log → `stability_flag = DIVERGED`, acc = 0.

### Stability flag
Set `stability_flag` per this decision tree:
1. NaN/Inf appeared → `DIVERGED`
2. `final_acc < 12.0` (random) → `DIVERGED`
3. `final_acc < peak_acc - 3.0` → `DEGRADED`
4. else → `OK`

(No V-norm tracking in search mode.)

### Promotion rule — pre-screen → full run

After the pre-screen finishes, evaluate the rule below in order. The first
clause that fires decides the outcome.

1. `stability_flag == DIVERGED` → DO NOT promote. Log the pre-screen row,
   skip the full run, exit.
2. `peak_acc < 18.0` (≤ 8 pp above random) → DO NOT promote. Same as above.
3. The two halves of the test-acc trace show no learning trend
   (mean of last-third > mean of first-third by < 1.0 pp) AND
   `peak_acc < 25.0` → DO NOT promote.
4. Otherwise → PROMOTE. Run the SAME config a second time at `n_iters` =
   3000 (or higher per the bump rule below). Log the second TSV row with
   `status=full`.

The promotion bar is intentionally loose: a pre-screen costs ~36 min, a full
run costs ~120 min; the cost of one false promotion is much smaller than the
cost of pruning a slow learner that would have improved.

## Logging results — `results_res_error_net.tsv`

Append-only, tab-separated. Header (one row, once):
```
experiment	seed	n_iters	peak_acc	peak_iter	final_acc	final_loss	stability_flag	param_lr	v_lr	alpha	inference_T	inference_dt	v_update_rule	v_init_scale	loss_type	optim	weight_decay	batch_size	v_reg	reproject_c	input_noise_sigma	normalization	dyt_init_alpha	highway_include_stem	status	notes
```

Example row:
```
5	42	3000	62.18	2750	61.40	0.3214	OK	3.2e-4	1.1e-4	0.73	20	0.1	energy	0.018	ce	adam	1.4e-4	32	0	1.0	0.1	dyt	0.5	True	full	alpha=0.73 T=20 looks promising
```

Always print to console when `current_best` (max `final_acc` with `OK` flag)
updates: `*** New best: trial=N final_acc=XX.XX peak_acc=YY.YY ***`.

## The experiment loop — pre-screen → full run

**ONE experiment per agent invocation.** The bash outer loop is the iterator;
the agent is NOT. Do not run a second experiment in one call. Do not stop
to ask for confirmation — just exit when done so the outer loop can move on.

Per invocation, in order:

1. Pick ONE config (see "New-config sampling" below).
2. Pre-screen at `--n-iters 900`, seed=42. Foreground, no `&`.
3. Parse the log; append a TSV row with `status=prescreen`.
4. Apply the promotion rule from "Reading results" above.
   - If NOT promoted: print the per-round summary and exit 0.
   - If promoted: continue to step 5.
5. Choose `n_iters` for the full run:
   - Default: **3000**.
   - If the bump rule (below) fires for this config family: **4500**, then
     **6000**, capped at **9000**.
6. Run the SAME config foreground at the chosen `n_iters`. Append a TSV row
   with `status=full`.
7. Print the per-round summary and exit 0.

### Convergence check & `n_iters` bump rule

Inspect the most recent FULL run for the config family. The family is the
top-tier hyperparam fingerprint (param_lr, v_lr, alpha, inference_T,
v_update_rule, loss, normalization). If the most recent full run on this
family satisfies BOTH:

- `peak_iter / n_iters ≥ 0.80` (peak landed in the last 20% of training), AND
- The mean test_acc over the **last 500 iters** exceeds the mean over the
  **preceding 500 iters** by `≥ 1.0` percentage points

then the run was still climbing — the next time a perturbation of this family
is promoted, use `n_iters = min(prior_n_iters * 1.5, 9000)`. Record this in
the TSV `notes` field (e.g., `notes="bumped to 4500 — family trending up"`)
so subsequent invocations can see the decision.

### Multi-seed verification (optional, end-of-budget)

After ≥ 30 OK full runs, the agent may take ONE invocation to re-run the
single best config (highest `final_acc` with `stability_flag=OK`) at
`--n-iters 3000` with `--seed 123` (one extra seed). Append as a normal full
row. This is the only legitimate reason to run the same config twice. Do not
do this more than once per session.

### New-config sampling

70% pure random over Tiers A–D (see "What you can change" above).
30% perturbation of a top-5 config by `final_acc` from FULL runs (status=full,
stability_flag=OK):
- continuous axis: multiply by `exp(U[-0.3, 0.3])` (≈ ±35%)
- discrete axis: jump to an adjacent choice in the ordered list

Reject any candidate whose `(log param_lr, log v_lr, log alpha)` triplet is
within 0.2 Euclidean distance of an already-tested config — resample.

If FEWER than 5 OK full runs exist yet, do 100% random (no perturbations).

### Plateau-break widening

If 10 consecutive invocations produce no new full-promoted config (i.e.
nothing passes the pre-screen → full promotion rule), widen Tier A:
- `--param-lr` in `[3e-3, 1e-2]` (upper extreme)
- `--res-alpha` in `[5, 20]` (very strong highway)
- `--res-resnet-inference-T 100`
- Force `--res-loss ce` for the next 6 invocations

### Phase 0 — one-time calibration (when TSV has only the header row)

Run a single timing sanity check, then a baseline row, then exit. This
replaces the multi-T / cold-vs-warm sweep — the new architecture is fast
enough that one measurement is enough.

1. **Timing sanity** — one short run, just to confirm sec/iter:
   ```bash
   t0=$(date +%s)
   timeout 600 python -u run_training.py \
     --variant res_error_net_resnet18 --dataset CIFAR10 --depths 1 \
     --act-fns relu --no-wandb --test-every 25 --n-iters 50 --seed 42 \
     --batch-size 32 --res-resnet-optim adam \
     --no-weight-updates --no-activity-norms --no-grad-norms --no-layer-energy \
     > calib_phase0.log 2>&1
   echo "Phase-0 calibration took $(( $(date +%s) - t0 ))s for 50 iters"
   ```
   Expected: ~120–180 s wall (50 iters @ ~25 iters/min + JIT). If wall time
   per iter exceeds ~5 s, log the discrepancy in TSV `notes` and continue.

2. **Baseline row** (experiment 1) — the very first real entry, run as a
   pre-screen at 900 iters with config defaults (no sampled flags):
   ```bash
   timeout 3000 python -u run_training.py \
     --variant res_error_net_resnet18 --dataset CIFAR10 --depths 1 \
     --act-fns relu --no-wandb --test-every 50 --n-iters 900 --seed 42 \
     --batch-size 32 --res-resnet-optim adam \
     --no-weight-updates --no-activity-norms --no-grad-norms --no-layer-energy \
     > run_001.log 2>&1
   ```
   Append to TSV with `status=prescreen, notes=baseline`. Then **EXIT** —
   do NOT promote the baseline to a full run automatically. The next outer
   loop iteration will start the real search.

## Per-round output contract

At the end of every invocation, print exactly:
```
Round <N> complete. Experiments added: <k>. Current best: final_acc=XX.XX (trial=<id>). Elapsed this round: <secs>s.
```

Where `<k>` ∈ {1, 2}: 1 if pre-screen was not promoted, 2 if it was. `Current
best` is the max `final_acc` across all `status=full, stability_flag=OK` rows
in the TSV (fall back to pre-screen rows if no full runs exist yet).

Then exit 0. The outer bash loop handles the next invocation.

## NEVER STOP

Do not pause to ask for confirmation. The outer loop runs until the human
interrupts (Ctrl-C or `tmux kill-session`).
