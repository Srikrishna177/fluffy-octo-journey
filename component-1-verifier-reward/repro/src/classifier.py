"""Small frozen CNN classifier used as the DDPO reward model.

Spec: ~50-150k params, trained on MNIST-16x16. Reward = softmax confidence for a
single fixed target digit class.
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


def reward_fn(classifier: nn.Module, images: torch.Tensor, target_class: int) -> torch.Tensor:
    """images: [B,1,16,16] in roughly [-1,1]. Returns [B] softmax confidence for target_class."""
    with torch.no_grad():
        logits = classifier(images)
        probs = torch.softmax(logits, dim=-1)
    return probs[:, target_class]
