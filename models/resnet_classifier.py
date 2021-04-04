import torch
import torch.nn as nn

from torchvision import models


class ResNetClassifier(nn.Module):

    def __init__(self, num_classes=8, pretrained=True, layers=50):
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes

        if layers == 18:
            self.net = models.resnet18(pretrained=pretrained)
        elif layers == 34:
            self.net = models.resnet34(pretrained=pretrained)
        elif layers == 50:
            self.net = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            self.net = models.resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

    def forward(self, x):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)
