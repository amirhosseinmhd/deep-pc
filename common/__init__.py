"""Shared utilities for unified PC experiments."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from common.data import set_seed, MNIST, one_hot, get_mnist_loaders
from common.utils import (
    ensure_dir, selected_layer_indices, get_weight_list,
    orthogonal_init, init_weights_orthogonal,
)
from common.hessian import unwrap_hessian_pytree
