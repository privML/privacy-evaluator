import os

import torch
import torch.nn as nn


__all__ = ["DCTI", "load_dcti"]


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)


class DCTI(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            Block(3, 64),
            Block(64, 64),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            Block(64, 128),
            Block(128, 128),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            Block(128, 256),
            Block(256, 256),
            Block(256, 256),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, 2),
            Block(256, 512),
            Block(512, 512),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, 2),
            Block(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x)


def load_dcti(pretrained: bool = True) -> DCTI:
    """
    DCTI model from
    `"Lightweight Deep Convolutional Network for Tiny Object Recognition"
    <https://www.scitepress.org/Papers/2018/67520/67520.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR-10
    """
    model = DCTI()
    if pretrained:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "dcti", "model.pth")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)

    return model
