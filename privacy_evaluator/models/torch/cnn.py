import torch
from torch import nn
from typing import Tuple, Union, Optional


def ConvNet(
    num_classes: int = 2, input_shape: Tuple[int, ...] = (1, 28, 28)
) -> nn.Module:
    if input_shape in [(1, 28, 28), (28, 28, 1)]:
        input_shape = (1, 28, 28)
        num_channels = (1, 64, 128)
        return ConvNetMNIST(num_classes, input_shape, num_channels)

    elif input_shape in [(3, 32, 32), (32, 32, 3)]:
        input_shape = (3, 32, 32)
        num_channels = (3, 64, 128)
        return ConvNetCIFAR10(num_classes, input_shape, num_channels)

    else:
        raise ValueError(
            "The input_shape must be one of the followings: "
            "(1, 28, 28), (28, 28, 1), (3, 32, 32), (32, 32, 3)"
        )


class ConvNetMNIST(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        num_channels: Tuple[int, ...] = (1, 64, 128),
    ):
        super(ConvNetMNIST, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_channels = num_channels

        # define the architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=(3, 3),
            ),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=(3, 3),
            ),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # if the input is in the form "BHWC", make channel the second dimension
        if x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        return out


class ConvNetCIFAR10(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        num_channels: Tuple[int, ...] = (3, 64, 128),
    ):
        super(ConvNetCIFAR10, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_channels = num_channels

        # define the architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=(3, 3),
            ),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=(3, 3),
            ),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # if the input is in the form "BHWC", make channel the second dimension
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)
        return out