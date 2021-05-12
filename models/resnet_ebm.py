import torch
import torch.nn as nn
from torchvision import models


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


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
            nn.Linear(self.net.fc.in_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            #nn.Sigmoid()
        )

        if freeze_backbone:
            # first set all to False
            requires_grad(self.net.parameters(), False)

            # set the last conv block and fc layer to True
            # requires_grad(list(self.net.layer4.parameters()), True)
            requires_grad(self.net.fc.parameters(), True)

    def forward(self, x):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)

    '''
    def train(self, mode=True):
        """Override train to control batch-norm layers."""
        nn.Module.train(self, mode and not self.freeze_backbone)
        self.net.fc.train(mode=mode)
    '''

    def enable_grads(self):
        """Enable gradients for trainable modules."""
        if self.freeze_backbone:
            # first set all to False
            requires_grad(self.net.parameters(), False)

            # set the last conv block and fc layer to True
            requires_grad(self.net.layer4.parameters(), True)
            requires_grad(self.net.fc.parameters(), True)
        else:
            self.enable_all_grads()

    def enable_all_grads(self):
        """Enable gradients for all modules."""
        requires_grad(self.net.parameters(), True)

    def disable_all_grads(self):
        """Disable gradients for all modules."""
        requires_grad(self.net.parameters(), False)

    def disable_batchnorm(self):
        self.net.apply(deactivate_batchnorm)
