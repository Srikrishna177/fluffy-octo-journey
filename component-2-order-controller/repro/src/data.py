"""MNIST-16x16 dataset loading, quantized to 4 gray levels (2 bits/pixel).

Spec (DECISION.md): MNIST downsampled to 16x16, pixel values quantized to 4
gray levels; each of the 256 pixel positions is a "token" over vocab
{0,1,2,3}, with a 5th MASK token id (4) added for the MaskGIT-style objective.

Adapted from component-1's repro/src/data.py (same downsampling approach),
made self-contained here per program convention.
"""
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

VOCAB_SIZE = 4          # gray levels 0..3
MASK_TOKEN = 4           # special mask id
SEQ_LEN = 256            # 16x16


def _resize_28_to_16(x: torch.Tensor) -> torch.Tensor:
    # x: [N, 1, 28, 28] in [0, 1]
    return torch.nn.functional.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)


def quantize_to_tokens(imgs01: torch.Tensor) -> torch.Tensor:
    """imgs01: [N,1,16,16] in [0,1]. Returns LongTensor [N,256] token ids in {0,1,2,3}."""
    levels = torch.clamp((imgs01 * VOCAB_SIZE).long(), max=VOCAB_SIZE - 1)  # 0..3
    return levels.view(imgs01.shape[0], -1)


def dequantize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: [...,256] long in {0,1,2,3}. Returns float in roughly [-1,1] (MNIST DDPM scale
    used by component-1's classifier), matching (level+0.5)/VOCAB_SIZE mapped to [0,1] then [-1,1].
    """
    frac01 = (tokens.float() + 0.5) / VOCAB_SIZE  # center of each quantization bin, [0,1]
    return frac01 * 2.0 - 1.0


def load_mnist16_tokens(n_train: int = 10000, n_test: int = 2000, seed: int = 0):
    """Returns (train_tokens, train_labels, test_tokens, test_labels, used_real).
    train_tokens/test_tokens: LongTensor [N, 256] in {0,1,2,3}.
    labels: LongTensor [N] MNIST digit labels (0-9), kept for the classifier-judge eval.
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
        raise RuntimeError(
            f"REAL MNIST download failed ({e!r}). Component-2's spec requires real MNIST "
            "(DECISION.md); no synthetic fallback is implemented here since component-1 already "
            "confirmed real MNIST is reachable in this sandbox."
        )

    train_imgs16 = _resize_28_to_16(train_imgs)  # [N,1,16,16] in [0,1]
    test_imgs16 = _resize_28_to_16(test_imgs)

    train_tokens = quantize_to_tokens(train_imgs16)
    test_tokens = quantize_to_tokens(test_imgs16)

    return train_tokens, train_labels, test_tokens, test_labels, used_real
