import torch
from torch import nn
from typing import Tuple, Union, Optional


def ConvNet(
    num_classes: int = 2,
    input_shape: Tuple[int, ...] = (1, 28, 28),
    num_channels: Tuple[int, ...] = (1, 32, 64, 128),
) -> nn.Module:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert num_channels[0] in [input_shape[0], input_shape[-1]], \
        "Argument num_channels must have the same value on the first or last dimension as argument input_shape!"
    assert len(num_channels) == 4, "Argument num_channels must have length 4."

    if input_shape in [(1, 28, 28), (28, 28, 1)]:
        input_shape = (1, 28, 28)
        return ConvNetMNIST(num_classes, input_shape, num_channels).to(device)

    elif input_shape in [(3, 32, 32), (32, 32, 3)]:
        input_shape = (3, 32, 32)
        return ConvNetCIFAR10(num_classes, input_shape, num_channels).to(device)

    else:
        raise ValueError(
            "The input_shape must be one of the followings: "
            "(1, 28, 28), (28, 28, 1), (3, 32, 32), (32, 32, 3)"
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvNetMNIST(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        num_channels: Tuple[int, ...] = (1, 16, 32, 64),
    ):
        super(ConvNetMNIST, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_channels = num_channels

        # define the architecture
        self.conv1 = ConvBlock(in_channels=num_channels[0], out_channels=num_channels[1])
        self.conv2 = ConvBlock(in_channels=num_channels[1], out_channels=num_channels[2])
        self.conv3 = ConvBlock(in_channels=num_channels[2], out_channels=num_channels[3])

        self.flatten = nn.Flatten()
        self.fc = self.generate_fc()

    def generate_fc(self):
        # Dry run in order to generate the fully connected layer with the correct input shape
        x = self.convs(torch.randn((2, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        x = self.flatten(x)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1),
        )

    def convs(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

    def forward(self, x):
        # if the input is in the form "BHWC", make channel the second dimension
        if x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        out = self.convs(x)
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
        self.conv1 = ConvBlock(in_channels=num_channels[0], out_channels=num_channels[1])
        self.conv2 = ConvBlock(in_channels=num_channels[1], out_channels=num_channels[2])
        self.conv3 = ConvBlock(in_channels=num_channels[2], out_channels=num_channels[3])

        self.flatten = nn.Flatten()
        self.fc = self.generate_fc()

    def generate_fc(self):
        # Dry run in order to generate the fully connected layer with the correct input shape
        x = self.convs(torch.randn((2, self.input_shape[0], self.input_shape[1], self.input_shape[2])))
        x = self.flatten(x)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(x.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1),
        )

    def convs(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


def forward(self, x):
        # if the input is in the form "BHWC", make channel the second dimension
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        out = self.convs(x)
        out = self.fc(out)
        return out

