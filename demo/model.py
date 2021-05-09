from torch import nn
import torchvision


class ResNet(nn.Module):
    """An adapted residual network for multi-classification.
    
    The backbone is the pretrained `resne18` model. We freeze all the convolutional
    layers and only change the last fully connected layer to do classification.

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate in the last fully-connected layer.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0):
        super(ResNet, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.resnet(x)
        return out
