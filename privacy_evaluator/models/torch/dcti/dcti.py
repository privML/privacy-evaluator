import os
import torch
import torch.nn as nn


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
    """DCTI model architecture from `"Lightweight Deep Convolutional Network for Tiny Object Recognition"
    <https://www.scitepress.org/Papers/2018/67520/67520.pdf>`.
    """

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


def load_dcti(
    pretrained: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> DCTI:
    """Loads a PyTorch DCTI model.

    :param pretrained: If True, returns a model pre-trained on CIFAR-10.
    :param device: Device on which the model is loaded. Either cpu or gpu.
    :return: Loaded PyTorch DCTI model.
    """
    model = DCTI()
    if pretrained:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "model", "model.pth"
                ),
                map_location=device,
            )
        )
    return model
