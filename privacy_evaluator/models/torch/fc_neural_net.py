from torch import nn
from typing import Tuple
import math

class FCNeuralNet(nn.Module):
    """A simple fully-connected network for multi-classification.

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate in the fully-connected layer.
    """

    def __init__(
            self,
            num_classes: int = 2,
            dropout: float = 0,
            input_shape: Tuple[int, ...] = (32, 32, 3)
    ):
        super(FCNeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.length_after_flatten = math.prod(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(length_after_flatten, 512),
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
