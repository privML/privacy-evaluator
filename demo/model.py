from torch import nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0):
        super(ResNet, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,64),
            nn.ReLU(), 
            nn.Linear(64, num_classes))

    def forward(self, x):
        out = self.resnet(x)
        return out