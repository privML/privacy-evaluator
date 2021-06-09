import torch
from torch import nn
from typing import Tuple, Optional


def FCNeuralNet(
    num_classes: int = 2, dropout: float = 0, input_shape: Tuple[int, ...] = (32, 32, 3)
) -> Optional[nn.Module]:
    """A simple fully-connected network for multi-classification.

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate in the fully-connected layer.
        input_shape: the shape of one data point,
            for CIFAR: [32, 32, 3], for MNIST [28, 28]
    """
    if input_shape == (32, 32, 3):
        return FCNeuralNet32X32X3(num_classes, dropout)
    elif input_shape in [(28, 28), (28, 28,), (28, 28, 1),]:
        return FCNeuralNet28X28X1(num_classes, dropout)


class FCNeuralNet32X32X3(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0,
    ):
        super(FCNeuralNet32X32X3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),  # TODO: retrieve later
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc(out)
        return out


class FCNeuralNet28X28X1(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0,
    ):
        super(FCNeuralNet28X28X1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28 * 1, 512),  # TODO: retrieve later
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc(out)
        return out
