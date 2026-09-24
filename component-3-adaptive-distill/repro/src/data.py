"""Reused verbatim from component-1-verifier-reward/repro/src/data.py
(frozen upstream code/checkpoints; do not modify semantics here).
"""
"""MNIST-16x16 dataset loading, with a synthetic fallback if download is blocked.

Spec (DECISION.md): MNIST downsampled to 16x16 grayscale, pixel values scaled to
roughly [-1, 1].
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _make_synthetic_mnist(n_samples: int, seed: int = 0):
    """Synthetic MNIST-like substitute: crude digit-shaped blobs on a 28x28 canvas
    per class, with per-sample random affine jitter and noise, labeled 0-9.

    Documented fallback only — used if the real MNIST download is unreachable.
    Not a claim of visual fidelity to real handwritten digits, only a labeled,
    learnable 10-class image dataset of the same shape/range as MNIST.
    """
    rng = np.random.RandomState(seed)
    imgs = np.zeros((n_samples, 28, 28), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int64)
    yy, xx = np.mgrid[0:28, 0:28]
    for i in range(n_samples):
        digit = i % 10
        labels[i] = digit
        cx, cy = 14 + rng.uniform(-2, 2), 14 + rng.uniform(-2, 2)
        img = np.zeros((28, 28), dtype=np.float32)
        # each digit class gets a distinct simple geometric signature so a small
        # CNN can actually learn to separate classes
        if digit == 0:
            r = 9 + rng.uniform(-1, 1)
            ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r)
            img = np.exp(-(ring ** 2) / 4.0)
        elif digit == 1:
            img = np.exp(-((xx - cx) ** 2) / 4.0) * (np.abs(yy - cy) < 10)
        else:
            n_strokes = 2 + (digit % 4)
            for s in range(n_strokes):
                ang = rng.uniform(0, np.pi) + digit * 0.3
                sx = cx + rng.uniform(-6, 6)
                sy = cy + rng.uniform(-6, 6)
                dx, dy = np.cos(ang), np.sin(ang)
                length = rng.uniform(6, 10)
                for t in np.linspace(-length / 2, length / 2, 20):
                    px, py = sx + t * dx, sy + t * dy
                    img += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / 3.0)
            img = np.clip(img, 0, 1)
        img += rng.normal(0, 0.03, size=img.shape).astype(np.float32)
        imgs[i] = np.clip(img, 0, 1)
    return imgs, labels


def _resize_28_to_16(x: torch.Tensor) -> torch.Tensor:
    # x: [N, 1, 28, 28] in [0, 1]
    return torch.nn.functional.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)


def load_mnist16(n_train: int = 10000, n_test: int = 2000, seed: int = 0):
    """Returns (train_dataset, test_dataset) of (image, label) pairs.
    Images are float tensors [1, 16, 16] scaled to roughly [-1, 1].
    Tries real MNIST via torchvision first; falls back to a synthetic dataset,
    logging which path was used to stdout.
    """
    used_real = False
    try:
        from torchvision import datasets, transforms

        os.makedirs(DATA_DIR, exist_ok=True)
        tfm = transforms.Compose([transforms.ToTensor()])
        train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
        test_full = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
        used_real = True
        print(f"[data] loaded REAL MNIST (train={len(train_full)}, test={len(test_full)})", flush=True)

        g = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_full), generator=g)[:n_train]
        test_idx = torch.randperm(len(test_full), generator=g)[:n_test]

        train_imgs = torch.stack([train_full[i][0] for i in train_idx])  # [N,1,28,28] in [0,1]
        train_labels = torch.tensor([train_full[i][1] for i in train_idx])
        test_imgs = torch.stack([test_full[i][0] for i in test_idx])
        test_labels = torch.tensor([test_full[i][1] for i in test_idx])
    except Exception as e:
        print(f"[data] REAL MNIST download failed ({e!r}); using SYNTHETIC MNIST-like fallback", flush=True)
        train_arr, train_lab = _make_synthetic_mnist(n_train, seed=seed)
        test_arr, test_lab = _make_synthetic_mnist(n_test, seed=seed + 1)
        train_imgs = torch.from_numpy(train_arr).unsqueeze(1)  # [N,1,28,28] in [0,1]
        train_labels = torch.from_numpy(train_lab)
        test_imgs = torch.from_numpy(test_arr).unsqueeze(1)
        test_labels = torch.from_numpy(test_lab)

    train_imgs16 = _resize_28_to_16(train_imgs)
    test_imgs16 = _resize_28_to_16(test_imgs)
    # scale [0,1] -> [-1,1]
    train_imgs16 = train_imgs16 * 2.0 - 1.0
    test_imgs16 = test_imgs16 * 2.0 - 1.0

    train_ds = TensorDataset(train_imgs16, train_labels)
    test_ds = TensorDataset(test_imgs16, test_labels)
    return train_ds, test_ds, used_real
