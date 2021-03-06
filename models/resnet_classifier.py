"""Flexible class for ResNet backbone."""

import torch.nn as nn

from torchvision import models


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


class ResNetClassifier(nn.Module):

    def __init__(self, num_classes=9, pretrained=True, freeze_backbone=False,
                 layers=50):
        super(ResNetClassifier, self).__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

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

        '''
        self.net.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
        '''
        self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

        self.enable_grads()

    def forward(self, x):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)

    def train(self, mode=True):
        """Override train to control batch-norm layers."""
        nn.Module.train(self, mode and not self.freeze_backbone)
        self.net.fc.train(mode=mode)

    def enable_grads(self):
        """Enable gradients for trainable modules."""
        if self.freeze_backbone:
            # first set all to False
            requires_grad(list(self.net.parameters()), False)

            # set the last conv block and fc layer to True
            requires_grad(list(self.net.layer4.parameters()), True)
            requires_grad(list(self.net.fc.parameters()), True)
        else:
            self.enable_all_grads()

    def enable_all_grads(self):
        """Enable gradients for all modules."""
        requires_grad(list(self.net.parameters()), True)

    def disable_all_grads(self):
        """Disable gradients for all modules."""
        requires_grad(list(self.net.parameters()), False)
