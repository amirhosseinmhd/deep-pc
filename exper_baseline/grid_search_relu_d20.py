"""Grid search over (param_lr, activity_lr) for baseline MLP (no skip) relu depth=20."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exper'))

import jax.random as jr
import numpy as np
from config import INPUT_DIM, OUTPUT_DIM, WIDTH, SEED, BATCH_SIZE, TEST_EVERY
from common import create_model, train_and_record, ensure_dir, set_seed

save_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "results", "grid_search"))
act_fn, depth = "relu", 20
n_train_iters = 200

param_lrs = [1e-2, 5e-3, 1e-3, 5e-4]
activity_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]

results = {}
best_acc = -1
best_config = None

total = len(param_lrs) * len(activity_lrs)
run = 0

for eta in param_lrs:
    for beta in activity_lrs:
        run += 1
        print(f"\n[{run}/{total}] η={eta:.0e}, β={beta:.0e}")
        set_seed(SEED)
        key = jr.PRNGKey(SEED)
        model, skip_model = create_model(
            key, depth=depth, width=WIDTH,
            act_fn=act_fn, param_type="sp", use_skips=False
        )
        try:
            res = train_and_record(
                seed=SEED, model=model, skip_model=skip_model,
                param_type="sp", activity_lr=beta, param_lr=eta,
                batch_size=BATCH_SIZE, n_train_iters=n_train_iters,
                test_every=n_train_iters,  # evaluate only at the end
                act_fn=act_fn,
            )
            final_acc = res["test_accs"][-1] if res["test_accs"] else 0.0
            final_loss = res["train_losses"][-1] if res["train_losses"] else float("inf")
            diverged = np.isnan(final_loss) or np.isinf(final_loss)
        except Exception as e:
            print(f"  FAILED: {e}")
            final_acc = 0.0
            final_loss = float("inf")
            diverged = True

        results[(eta, beta)] = {
            "acc": final_acc,
            "loss": final_loss,
            "diverged": diverged,
        }

        status = "DIVERGED" if diverged else f"acc={final_acc:.2f}%, loss={final_loss:.4f}"
        print(f"  -> {status}")

        if final_acc > best_acc and not diverged:
            best_acc = final_acc
            best_config = (eta, beta)

# ================================================================
# Summary table
# ================================================================
print("\n" + "=" * 70)
print("GRID SEARCH RESULTS — Baseline MLP (no skip) relu depth=20")
print("=" * 70)

header = f"{'η (param)':>12} | {'β (activity)':>12} | {'Test Acc':>10} | {'Loss':>10} | {'Status':>8}"
print(header)
print("-" * len(header))

for eta in param_lrs:
    for beta in activity_lrs:
        r = results[(eta, beta)]
        status = "DIV" if r["diverged"] else "OK"
        acc_str = f"{r['acc']:.2f}%" if not r["diverged"] else "—"
        loss_str = f"{r['loss']:.4f}" if not r["diverged"] else "—"
        marker = " <-- BEST" if (eta, beta) == best_config else ""
        print(f"{eta:>12.0e} | {beta:>12.0e} | {acc_str:>10} | {loss_str:>10} | {status:>8}{marker}")

if best_config:
    print(f"\nBest config: η={best_config[0]:.0e}, β={best_config[1]:.0e} -> {best_acc:.2f}%")
else:
    print("\nNo successful config found.")

# Save results
np.save(os.path.join(save_dir, "grid_results_relu_d20.npy"), results)
print(f"Results saved to {save_dir}")
