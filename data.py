"""MNIST loading, augmentation and device helpers.

Rule of the assignment: train on the TRAIN split, test on the TEST split.
We additionally hold out the last 5,000 training images as a validation set
for picking the best epoch, so the test set is never used for any decision.
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
VAL_SIZE = 5000

normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

# Plain pipeline: tensor + normalize.
plain_tf = transforms.Compose([transforms.ToTensor(), normalize])

# Augmentations that keep a digit the same digit.
# (No horizontal/vertical flips: a flipped 2 or 7 is not a 2 or 7.)
augment_tf = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.12, 0.12),
                            scale=(0.88, 1.12), shear=8),
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0),
])

# "Hard" test pipeline: the real test images, randomly rotated/shifted/scaled.
# Used only to measure robustness; the standard score is on the clean test set.
perturbed_test_tf = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    normalize,
])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_val_loaders(data_dir="data", augment=False, batch_size=128, workers=2):
    train_full = datasets.MNIST(data_dir, train=True, download=True,
                                transform=augment_tf if augment else plain_tf)
    val_full = datasets.MNIST(data_dir, train=True, download=True, transform=plain_tf)
    n = len(train_full)                                  # 60,000
    train_idx = list(range(n - VAL_SIZE))
    val_idx = list(range(n - VAL_SIZE, n))
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)                   # never augmented
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def test_loader(data_dir="data", perturbed=False, batch_size=512, workers=2):
    tf = perturbed_test_tf if perturbed else plain_tf
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
