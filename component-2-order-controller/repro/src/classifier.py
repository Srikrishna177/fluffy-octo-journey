"""Frozen MNIST classifier architecture, copied from component-1
(component-1-verifier-reward/repro/src/classifier.py) so this component stays
self-contained. Only the architecture is duplicated; the actual weights are
loaded read-only from component-1's checkpoint (see inference.py) -- no
retraining happens here.
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x8x8
            nn.Conv2d(16, 32, 3, padding=1),  # 32x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x4x4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.net(x))
