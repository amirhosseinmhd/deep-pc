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
