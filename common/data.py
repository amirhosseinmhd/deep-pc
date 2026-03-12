"""Data loading and seeding utilities."""

import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
# CIFAR-10
# ---------------------------------------------------------------------------
class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, normalise=True, save_dir="data/CIFAR10"):
        if normalise:
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                    ),
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


def get_cifar10_loaders(batch_size):
    train_data = CIFAR10(train=True, normalise=True)
    test_data = CIFAR10(train=False, normalise=True)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    return train_loader, test_loader


def get_dataloaders(dataset, batch_size):
    """Return (train_loader, test_loader) for the given dataset name."""
    if dataset == "MNIST":
        return get_mnist_loaders(batch_size)
    elif dataset == "CIFAR10":
        return get_cifar10_loaders(batch_size)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Options: MNIST, CIFAR10"
        )
