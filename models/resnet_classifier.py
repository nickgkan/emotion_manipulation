"""Flexible class for ResNet backbone."""

import torch.nn as nn

from torchvision import models

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class ResNetClassifier(nn.Module):

    def __init__(self, num_classes=8, pretrained=True, freeze_backbone=False, layers=50):
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

        if freeze_backbone:
            # first set all to False
            requires_grad(list(self.net.parameters()), False)

            # set the last conv block and fc layer to True
            requires_grad(list(self.net.layer4.parameters()), True)
            requires_grad(list(self.net.fc.parameters()), True)

    def forward(self, x):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)
