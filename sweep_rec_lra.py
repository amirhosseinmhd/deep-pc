#!/usr/bin/env python
"""Random hyperparameter sweep for rec-LRA over beta, gamma_E, e_lr."""

import subprocess
import random
import math
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_TRIALS = 20
N_ITERS = 100
N_WORKERS = 6  # number of parallel experiments
LOG_FILE = "sweep_results.csv"

# Search ranges (log-uniform where appropriate)
SEARCH_SPACE = {
    "beta":    (0.01, 10.0,  "log"),   # nudging strength
    "gamma_e": (0.001, 1.0,  "log"),   # E learning rate scale
    "e_lr":    (1e-4, 1e-1,  "log"),   # E learning rate
}

def sample_param(lo, hi, scale):
    if scale == "log":
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    return random.uniform(lo, hi)

def run_trial(trial_id, beta, gamma_e, e_lr):
    cmd = [
        "python", "run_training.py",
        "--variant", "rec_lra",
        "--depths", "10",
        "--n-iters", str(N_ITERS),
        "--act-fns", "tanh",
        "--rec-lra-e-update", "grad",
        "--forward-skip-every", "5",
        "--error-skip-every", "5",
        "--beta", f"{beta:.6f}",
        "--gamma-e", f"{gamma_e:.6f}",
        "--e-lr", f"{e_lr:.6f}",
        "--no-wandb",
    ]
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}/{NUM_TRIALS}: beta={beta:.4f} gamma_e={gamma_e:.4f} e_lr={e_lr:.6f}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout

        # Parse final test accuracy from output
        # Format: "Iter X, loss=Y, train acc=Z, test acc=W"
        test_acc = None
        final_loss = None
        for line in output.split("\n"):
            if "test acc=" in line:
                try:
                    test_acc = float(line.split("test acc=")[-1].strip())
                except ValueError:
                    pass
            if "loss=" in line:
                try:
                    final_loss = float(line.split("loss=")[1].split(",")[0])
                except (ValueError, IndexError):
                    pass

        if test_acc is None:
            print(f"  -> Could not parse test_acc. Final loss: {final_loss}")
            print(f"  Last 5 lines of output:")
            for line in output.strip().split("\n")[-5:]:
                print(f"     {line}")
            return {"test_acc": None, "final_loss": final_loss}

        print(f"  -> Test acc: {test_acc}")
        return {"test_acc": test_acc, "final_loss": None}

    except subprocess.TimeoutExpired:
        print(f"  -> TIMEOUT")
        return {"test_acc": None, "final_loss": None}
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"test_acc": None, "final_loss": None}


def run_trial_wrapper(args):
    """Wrapper for parallel execution (ProcessPoolExecutor needs picklable args)."""
    trial_id, beta, gamma_e, e_lr = args
    result = run_trial(trial_id, beta, gamma_e, e_lr)
    return trial_id, beta, gamma_e, e_lr, result


def main():
    random.seed(42)

    # Write CSV header
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["trial", "beta", "gamma_e", "e_lr", "test_acc", "final_loss"])

    # Pre-generate all trial configs
    trials = []
    for i in range(1, NUM_TRIALS + 1):
        beta = sample_param(*SEARCH_SPACE["beta"])
        gamma_e = sample_param(*SEARCH_SPACE["gamma_e"])
        e_lr = sample_param(*SEARCH_SPACE["e_lr"])
        trials.append((i, beta, gamma_e, e_lr))

    best_acc = -1
    best_params = {}

    print(f"Running {NUM_TRIALS} trials with {N_WORKERS} parallel workers...")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_trial_wrapper, t): t for t in trials}

        for future in as_completed(futures):
            trial_id, beta, gamma_e, e_lr, result = future.result()

            # Log to CSV
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([trial_id, beta, gamma_e, e_lr, result["test_acc"], result["final_loss"]])

            if result["test_acc"] is not None and result["test_acc"] > best_acc:
                best_acc = result["test_acc"]
                best_params = {"beta": beta, "gamma_e": gamma_e, "e_lr": e_lr}
                print(f"  *** New best! acc={best_acc:.2f} with {best_params}")

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Best params: {best_params}")
    print(f"Full results saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
