"""Data loading and seeding utilities."""

import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 GCN + ZCA preprocessing (Ororbia & Mali 2023, p.15). The
# whitening matrix is computed once on the *training* set and cached.
_ZCA_CACHE_PATH = "data/cifar10_zca_cache.npz"


def _global_contrast_normalize(x, scale=55.0, eps=1e-8):
    """Per-image GCN: subtract mean, divide by L2 norm. x: float32 [..., D]."""
    x = x - x.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
    return scale * x / np.maximum(norm, eps)


def _compute_zca(flat_train, eps=1e-2):
    """Compute the ZCA whitening matrix and per-feature mean of GCN'd data."""
    mean = flat_train.mean(axis=0, keepdims=True)
    centered = flat_train - mean
    cov = (centered.T @ centered) / centered.shape[0]
    U, S, _ = np.linalg.svd(cov.astype(np.float64))
    W = (U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T).astype(np.float32)
    return W, mean.astype(np.float32)


def _load_or_build_cifar10_zca():
    """Compute (or load cached) ZCA whitening params from CIFAR-10 train."""
    if os.path.exists(_ZCA_CACHE_PATH):
        npz = np.load(_ZCA_CACHE_PATH)
        return npz["W"], npz["mean"]

    raw = datasets.CIFAR10("data/CIFAR10", train=True, download=True)
    imgs = np.asarray(raw.data, dtype=np.float32) / 255.0
    flat = imgs.reshape(imgs.shape[0], -1)
    flat = _global_contrast_normalize(flat)
    W, mean = _compute_zca(flat)

    os.makedirs(os.path.dirname(_ZCA_CACHE_PATH), exist_ok=True)
    np.savez(_ZCA_CACHE_PATH, W=W, mean=mean)
    return W, mean


class _GCNZCATransform:
    """Per-sample GCN followed by ZCA whitening using cached train stats."""

    def __init__(self):
        self._W, self._mean = _load_or_build_cifar10_zca()

    def __call__(self, img_tensor):
        x = img_tensor.numpy().astype(np.float32)
        flat = x.reshape(-1)
        flat = _global_contrast_normalize(flat[None, :])[0]
        flat = (flat - self._mean[0]) @ self._W
        return torch.from_numpy(flat.reshape(x.shape))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(labels, n_classes=10):
    return torch.eye(n_classes)[labels]


# ---------------------------------------------------------------------------
# Dataset dimensions
# ---------------------------------------------------------------------------
DATASET_INPUT_DIMS = {
    "MNIST": 28 * 28,          # 784
    "FashionMNIST": 28 * 28,   # 784
    "CIFAR10": 32 * 32 * 3,    # 3072
}


def get_input_dim(dataset):
    """Return input dimensionality for a given dataset name."""
    if dataset not in DATASET_INPUT_DIMS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Options: {list(DATASET_INPUT_DIMS.keys())}"
        )
    return DATASET_INPUT_DIMS[dataset]


# ---------------------------------------------------------------------------
# MNIST
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


# ---------------------------------------------------------------------------
# Fashion-MNIST
# ---------------------------------------------------------------------------
class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, normalise=True, save_dir="data"):
        if normalise:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.2860,), std=(0.3530,))
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label)
        return img, label


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------
class CIFAR10(datasets.CIFAR10):
    """CIFAR-10 with paper-spec preprocessing: GCN + ZCA whitening."""

    def __init__(self, train, normalise=True, save_dir="data/CIFAR10",
                 use_zca=True):
        if normalise and use_zca:
            transform = transforms.Compose([
                transforms.ToTensor(),
                _GCNZCATransform(),
            ])
        elif normalise:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(save_dir, download=True, train=train, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = torch.flatten(img)
        label = one_hot(label)
        return img, label


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
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


def get_fashion_mnist_loaders(batch_size):
    train_data = FashionMNIST(train=True, normalise=True)
    test_data = FashionMNIST(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    return train_loader, test_loader


def get_cifar10_loaders(batch_size, use_zca=True):
    train_data = CIFAR10(train=True, normalise=True, use_zca=use_zca)
    test_data = CIFAR10(train=False, normalise=True, use_zca=use_zca)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    return train_loader, test_loader


def get_dataloaders(dataset, batch_size, use_zca=True):
    """Return (train_loader, test_loader) for the given dataset name."""
    if dataset == "MNIST":
        return get_mnist_loaders(batch_size)
    elif dataset == "FashionMNIST":
        return get_fashion_mnist_loaders(batch_size)
    elif dataset == "CIFAR10":
        return get_cifar10_loaders(batch_size, use_zca=use_zca)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Options: MNIST, FashionMNIST, CIFAR10"
        )
