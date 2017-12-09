

import os
import sys

import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool2d, Linear
from torch.nn import AdaptiveAvgPool2d

# Setup pretrained-models.pytorch
assert 'PRETRAINED_MODELS' in os.environ
sys.path.append(os.environ['PRETRAINED_MODELS'])

from torch.nn.init import xavier_normal, kaiming_normal
from torch.nn import Conv2d, BatchNorm2d, Linear
from torch.nn import ReLU

from pretrainedmodels.inceptionv4 import Inception_A, Inception_B, Reduction_A, Reduction_B, Inception_C, BasicConv2d
from . import Flatten


class _Mixed_3a(Module):
    def __init__(self):
        super(_Mixed_3a, self).__init__()
        self.maxpool = MaxPool2d(2, stride=2, padding=1)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class _Mixed_4a(Module):

    def __init__(self):
        super(_Mixed_4a, self).__init__()

        self.branch0 = Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class _Mixed_5a(Module):

    def __init__(self):
        super(_Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.maxpool = MaxPool2d(2, stride=2, padding=1)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class IcebergInceptionV4(Module):

    def __init__(self, input_n_channels):
        super(IcebergInceptionV4, self).__init__()

        # Specific stem
        self.stem = Sequential(
            BasicConv2d(input_n_channels, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 80, kernel_size=1, stride=1),
            BasicConv2d(80, 192, kernel_size=3, stride=1),
            BasicConv2d(192, 384, kernel_size=3, stride=2, padding=1),
        )

        # input channels : 384
        self.features = Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            AdaptiveAvgPool2d(1),
            Flatten()
        )

        # Linear layer
        self.classifier = Linear(1536, 2)

    def forward(self, x, a):
        x = self.stem(x)
        x = self.features(x)
        y = self.classifier(x)
        return y
    
    
class IcebergInceptionV5(Module):

    def __init__(self, input_n_channels, n_features=5):
        super(IcebergInceptionV5, self).__init__()

        # Specific stem
        self.stem = Sequential(
            BasicConv2d(input_n_channels, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2, padding=1),
        )

        # input channels : 384
        self.features = Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            AdaptiveAvgPool2d(1),
            Flatten()
        )

        # Linear layer
        self.classifier = Linear(1536, n_features)
        # Final classification layer
        self.final_classifier = Linear(n_features + 1, 2)

    def forward(self, x, a):
        x = self.stem(x)
        f0 = self.features(x)
        f1 = self.classifier(f0)
        if len(a.size()) == 1:
            a = a.unsqueeze(dim=1)
        f = torch.cat((f1, a), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergInceptionV6(Module):

    def __init__(self, input_n_channels, n_features=5, n_classes=1):
        super(IcebergInceptionV6, self).__init__()

        # Specific stem
        self.stem = Sequential(
            BasicConv2d(input_n_channels, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2, padding=1),
        )

        # input channels : 384
        self.features = Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            AdaptiveAvgPool2d(1),
            Flatten(),
            ReLU(inplace=True)
        )

        # Linear layer
        self.classifier = Linear(1536, n_features)
        # Final classification layer
        self.final_classifier = Linear(n_features + 1, n_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                xavier_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x, a):
        x = self.stem(x)
        f0 = self.features(x)
        f1 = self.classifier(f0)
        if len(a.size()) == 1:
            a = a.unsqueeze(dim=1)
        f = torch.cat((f1, a), dim=1)
        y = self.final_classifier(f)
        return y