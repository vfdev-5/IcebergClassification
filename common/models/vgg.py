import math

import torch
from torch.nn import Module
from torch.nn import Sequential, MaxPool2d, Linear, Conv2d, ReLU, Dropout
from torch.nn.init import xavier_uniform, xavier_normal, kaiming_normal
from . import Flatten


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def make_layers(in_channels, cfg):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return Sequential(*layers)


class IcebergVGG16(Module):

    def __init__(self, input_n_channels, n_features=5):
        super(IcebergVGG16, self).__init__()

        self.features = make_layers(input_n_channels, cfg['D'])
        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 512),
            ReLU(True),
            Dropout(),
            Linear(512, n_features),
        )
        self.final_classifier = Linear(n_features + 1, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                xavier_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, a):
        f0 = self.features(x)
        f1 = self.classifier(f0)
        if len(a.size()) == 1:
            a = a.unsqueeze(dim=1)
        f = torch.cat((f1, a), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergVGGv2(Module):

    def __init__(self, input_n_channels, n_features=5, n_classes=2, cfg_type='D'):
        super(IcebergVGGv2, self).__init__()

        self.features = make_layers(input_n_channels, cfg[cfg_type])
        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 512),
            ReLU(True),
            Dropout(),
            Linear(512, n_features),
            ReLU(inplace=True)
        )
        self.final_classifier = Linear(n_features + 1, n_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                xavier_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x, a):
        f0 = self.features(x)
        f1 = self.classifier(f0)
        if len(a.size()) == 1:
            a = a.unsqueeze(dim=1)
        f = torch.cat((f1, a), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergVGGv0(Module):

    def __init__(self, input_n_channels, n_classes=2, cfg_type='D'):
        super(IcebergVGGv0, self).__init__()

        self.features = make_layers(input_n_channels, cfg[cfg_type])
        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 256),
            ReLU(True),
            Dropout(),
            Linear(256, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                xavier_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y