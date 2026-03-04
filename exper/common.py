"""Shared utilities for μPC experiments."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    INPUT_DIM, OUTPUT_DIM, WIDTH, SEED,
    ACTIVITY_LR, PARAM_LR, BATCH_SIZE, TEST_EVERY, N_TRAIN_ITERS,
    DEPTHS, ACT_FNS, RESULTS_DIR,
)

import jpc

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.tree_util import tree_leaves

import equinox as eqx
import optax

import math
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings
warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class MNIST(datasets.MNIST):
    def __init__(self, train, normalise=True, save_dir="data"):
        if normalise:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label)
        return img, label


def one_hot(labels, n_classes=10):
    return torch.eye(n_classes)[labels]


def get_mnist_loaders(batch_size):
    train_data = MNIST(train=True, normalise=True)
    test_data = MNIST(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_model(key, depth, width=WIDTH, act_fn="tanh",
                 param_type="sp", use_skips=True):
    """Create a model and optional skip model.

    Returns (model, skip_model).
    """
    model = jpc.make_mlp(
        key=key,
        input_dim=INPUT_DIM,
        width=width,
        depth=depth,
        output_dim=OUTPUT_DIM,
        act_fn=act_fn,
        use_bias=False,
        param_type=param_type,
    )
    skip_model = jpc.make_skip_model(depth) if use_skips else None
    return model, skip_model


# ---------------------------------------------------------------------------
# Orthogonal init (from mupc_paper/utils.py)
# ---------------------------------------------------------------------------
def orthogonal_init(key, weight, gain=1.0):
    out_f, in_f = weight.shape
    shape = (max(out_f, in_f), min(out_f, in_f))
    M = jr.normal(key, shape=shape)
    Q, R = jnp.linalg.qr(M)
    Q *= jnp.sign(jnp.diag(R))
    if out_f < in_f:
        Q = Q.T
    return gain * Q[:out_f, :in_f]


def init_weights_orthogonal(key, model, act_fn="tanh"):
    """Re-initialise all weights with orthogonal init."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight for x in tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    weights = get_weights(model)
    gain = 1.05 if act_fn == "tanh" else 1.0
    subkeys = jr.split(key, len(weights))
    new_weights = [orthogonal_init(sk, w, gain) for sk, w in zip(subkeys, weights)]
    return eqx.tree_at(get_weights, model, new_weights)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, skip_model, test_loader, param_type):
    avg_test_acc = 0.0
    for _, (img_batch, label_batch) in enumerate(test_loader):
        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()
        _, test_acc = jpc.test_discriminative_pc(
            model=model, output=label_batch, input=img_batch,
            skip_model=skip_model, param_type=param_type
        )
        avg_test_acc += float(test_acc)
    return avg_test_acc / len(test_loader)


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------
def _get_weight_list(model, act_fn):
    """Return a flat list of weight arrays from the model."""
    all_params = tree_leaves(model)
    if act_fn not in ("linear",):
        # activation functions are stored as leaves too; filter to arrays only
        all_params = [p for p in all_params if isinstance(p, jnp.ndarray) and p.ndim >= 2]
    return all_params


def _selected_layer_indices(depth):
    """Return indices for layers: 1, L/4, L/2, 3L/4, L."""
    L = depth
    idxs = sorted(set([
        0,
        max(0, int(L / 4) - 1),
        max(0, int(L / 2) - 1),
        max(0, int(L * 3 / 4) - 1),
        L - 1,
    ]))
    return idxs


# ---------------------------------------------------------------------------
# Training loop with optional metric tracking
# ---------------------------------------------------------------------------
def train_and_record(
    seed, model, skip_model, param_type,
    activity_lr=ACTIVITY_LR, param_lr=PARAM_LR,
    batch_size=BATCH_SIZE, n_train_iters=N_TRAIN_ITERS,
    test_every=TEST_EVERY, act_fn="tanh",
    track_weight_updates=False,
    track_activity_norms=False,
):
    """Train a PCN and record metrics.

    Returns a dict with keys:
        train_losses, test_accs, test_iters,
        weight_update_norms (if requested),
        activity_norms_init (if requested),
        activity_norms_post (if requested),
    """
    set_seed(seed)

    depth = len(model)

    activity_optim = optax.sgd(activity_lr)
    param_optim = optax.adam(param_lr)
    param_opt_state = param_optim.init(
        (eqx.filter(model, eqx.is_array), skip_model)
    )
    train_loader, test_loader = get_mnist_loaders(batch_size)

    train_losses = []
    test_accs = []
    test_iters = []

    layer_idxs = _selected_layer_indices(depth)
    n_tracked = len(layer_idxs)

    # Weight update tracking
    weight_update_norms = [] if track_weight_updates else None

    # Activity norm tracking
    activity_norms_init = [] if track_activity_norms else None
    activity_norms_post = [] if track_activity_norms else None

    data_iter = iter(train_loader)
    for iter_num in range(n_train_iters):
        try:
            img_batch, label_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            img_batch, label_batch = next(data_iter)

        img_batch, label_batch = img_batch.numpy(), label_batch.numpy()

        # Init activities via forward pass
        activities = jpc.init_activities_with_ffwd(
            model=model, input=img_batch,
            skip_model=skip_model, param_type=param_type
        )
        activity_opt_state = activity_optim.init(activities)
        train_loss = float(jpc.mse_loss(activities[-1], label_batch))
        train_losses.append(train_loss)

        # Record activity norms at init (pre-inference)
        if track_activity_norms:
            norms = []
            for a in activities:
                norms.append(float(jnp.linalg.norm(a, axis=1, ord=2).mean()))
            activity_norms_init.append(norms)

        # Snapshot weights before update (for weight-update tracking)
        if track_weight_updates:
            old_weights = _get_weight_list(model, act_fn)
            old_weights = [jnp.array(w) for w in old_weights]

        # Inference
        for t in range(depth):
            result = jpc.update_pc_activities(
                params=(model, skip_model), activities=activities,
                optim=activity_optim, opt_state=activity_opt_state,
                output=label_batch, input=img_batch,
                param_type=param_type
            )
            activities = result["activities"]
            activity_opt_state = result["opt_state"]

        # Record activity norms post-inference
        if track_activity_norms:
            norms = []
            for a in activities:
                norms.append(float(jnp.linalg.norm(a, axis=1, ord=2).mean()))
            activity_norms_post.append(norms)

        # Learning
        result = jpc.update_pc_params(
            params=(model, skip_model), activities=activities,
            optim=param_optim, opt_state=param_opt_state,
            output=label_batch, input=img_batch,
            param_type=param_type
        )
        model = result["model"]
        skip_model = result["skip_model"]
        param_opt_state = result["opt_state"]

        # Weight update norms
        if track_weight_updates:
            new_weights = _get_weight_list(model, act_fn)
            update_norms = []
            for w_old, w_new in zip(old_weights, new_weights):
                update_norms.append(
                    float(jnp.linalg.norm(jnp.ravel(w_new - w_old)))
                )
            weight_update_norms.append(update_norms)

        if np.isinf(train_loss) or np.isnan(train_loss):
            print(f"  Diverged at iter {iter_num}, loss={train_loss}")
            break

        if ((iter_num + 1) % test_every) == 0:
            avg_acc = evaluate(model, skip_model, test_loader, param_type)
            test_accs.append(avg_acc)
            test_iters.append(iter_num + 1)
            print(f"  Iter {iter_num+1}, loss={train_loss:.4f}, "
                  f"test acc={avg_acc:.2f}")

    out = {
        "train_losses": train_losses,
        "test_accs": test_accs,
        "test_iters": test_iters,
    }
    if track_weight_updates:
        out["weight_update_norms"] = np.array(weight_update_norms)
    if track_activity_norms:
        out["activity_norms_init"] = np.array(activity_norms_init)
        out["activity_norms_post"] = np.array(activity_norms_post)
    return out


# ---------------------------------------------------------------------------
# Hessian / condition number
# ---------------------------------------------------------------------------
def unwrap_hessian_pytree(hessian_pytree, activities):
    """Convert Hessian pytree to dense matrix."""
    activities = activities[:-1]
    hessian_pytree = hessian_pytree[:-1]
    widths = [a.shape[1] for a in activities]
    N = sum(widths)
    hessian_matrix = jnp.zeros((N, N))

    start_row = 0
    for l, pytree_l in enumerate(hessian_pytree):
        start_col = 0
        for k, pytree_k in enumerate(pytree_l[:-1]):
            block = pytree_k[0, :, 0].reshape(widths[l], widths[k])
            hessian_matrix = hessian_matrix.at[
                start_row:start_row + widths[l],
                start_col:start_col + widths[k]
            ].set(block)
            start_col += widths[k]
        start_row += widths[l]
    return hessian_matrix


def compute_condition_number(model, skip_model, param_type, x, y):
    """Compute condition number κ(H_z) of the activity Hessian."""
    activities = jpc.init_activities_with_ffwd(
        model=model, input=x,
        skip_model=skip_model, param_type=param_type
    )
    hessian_pytree = jax.hessian(jpc.pc_energy_fn, argnums=1)(
        (model, skip_model), activities, y,
        x=x, param_type=param_type
    )
    H = unwrap_hessian_pytree(hessian_pytree, activities)
    eigenvals = jnp.linalg.eigvalsh(H)
    lam_max = jnp.abs(eigenvals[-1])
    lam_min = jnp.abs(eigenvals[0])
    cond = float(lam_max / jnp.maximum(lam_min, 1e-30))
    return cond, eigenvals


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
