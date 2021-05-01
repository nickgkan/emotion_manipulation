import torch.nn as nn

from torchvision import models


class ResNetEBM(nn.Module):

    def __init__(self, pretrained=True, freeze_backbone=False, layers=50):
        super(ResNetEBM, self).__init__()
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

        # scoring layers
        self.net.fc = nn.Sequential(
            nn.Linear(self.net.fc.in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)

    def train(self, mode=True):
        """Override train to control batch-norm layers."""
        nn.Module.train(self, mode and not self.freeze_backbone)
        self.net.fc.train(mode=mode)
