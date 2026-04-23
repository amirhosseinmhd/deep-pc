"""Shared utilities for unified PC experiments."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from common.data import (
    set_seed, MNIST, FashionMNIST, CIFAR10, one_hot,
    get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders,
    get_dataloaders, get_input_dim,
)
from common.utils import (
    ensure_dir, selected_layer_indices, get_weight_list,
    orthogonal_init, init_weights_orthogonal,
)
from common.hessian import unwrap_hessian_pytree
