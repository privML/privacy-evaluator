import torch
from torch import nn
from typing import Tuple, Optional


def FCNeuralNet(
    num_classes: int = 2, dropout: float = 0, input_shape: Tuple[int, ...] = (32, 32, 3)
) -> Optional[nn.Module]:
    """Provide a simple fully-connected network for multi-classification in PyTorch.

    Note: This method is just aimed at fetching a model for developers' test when 
    a target model is required. Since only `MNIST` and `CIFAR10` datasets are 
    our concern, this method is compatible only with these two corresponding 
    image sizes (28*28 and 32*32*3)

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate between the fully-connected layers.
        input_shape: the shape of one data point,
            for CIFAR: [32, 32, 3], for MNIST [28, 28]

    Returns:
        An 28X28X1-classifier if `input_shape` corresponds to 28*28 or a 
        32X32X3-classifier if corresponds to 32*32*3. Otherwise raise an Error.
    """
    if input_shape == (32, 32, 3):
        return FCNeuralNet32X32X3(num_classes, dropout)
    elif input_shape in [
        (28, 28),
        (
            28,
            28,
        ),
        (28, 28, 1),
    ]:
        return FCNeuralNet28X28X1(num_classes, dropout)
    else:
        raise ValueError(
            "The input_shape must be one of the followings: "
            "(1, 28, 28), (28, 28, 1), (28, 28), (32, 32, 3)"
        )


class FCNeuralNet32X32X3(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0,
    ):
        super(FCNeuralNet32X32X3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
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
            nn.Linear(28 * 28 * 1, 512),
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
