

import os
import sys


import torch
from torch.nn import Module
from torch.nn import Sequential, Dropout, Conv2d, ReLU, AdaptiveAvgPool2d, MaxPool2d

from torchvision.models.squeezenet import Fire
from torchvision.models import densenet161

# Setup pretrained-models.pytorch
assert 'PRETRAINED_MODELS' in os.environ
sys.path.append(os.environ['PRETRAINED_MODELS'])

import pretrainedmodels


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class IcebergSqueezeNet(Module):

    def __init__(self, input_n_channels):
        super(IcebergSqueezeNet, self).__init__()

        self.features = Sequential(
            Conv2d(input_n_channels, 64, kernel_size=3, stride=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently form the rest
        final_conv = Conv2d(512, 2, kernel_size=1)
        self.classifier = Sequential(
            Dropout(p=0.5),
            final_conv,
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten()
        )
        # init weights
        for m in self.modules():
            if isinstance(m, Conv2d):
                if m is final_conv:
                    torch.nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, a):
        x1, x2 = x[:, 0, :, :].unsqueeze(dim=1), x[:, 1, :, :].unsqueeze(dim=1)
        x1 = self.features(x1)
        y1 = self.classifier(x1)

        x2 = self.features(x2)
        y2 = self.classifier(x2)

        y = torch.max(y1, y2)
        return y

        
    



