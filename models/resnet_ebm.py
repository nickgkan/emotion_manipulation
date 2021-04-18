import torch.nn as nn

from torchvision import models


class ResNetEBM(nn.Module):

    def __init__(self, pretrained=True, layers=50):
        super(ResNetEBM, self).__init__()

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

        # scoring layer
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)
        
    def forward(self, cls, xyz, pad_mask):
        """Forward pass for an input image (B, 3, 224, 224)."""
        return self.net(x)

    def train(self, mode=True):
        """Override train to control batch-norm layers."""
        nn.Module.train(self, mode and not self.freeze_backbone)
