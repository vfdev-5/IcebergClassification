
from torchvision.models.squeezenet import Fire

import torch
from torch.nn import Module
from torch.nn import Sequential, Dropout, Conv2d, ReLU, MaxPool2d, Linear
from torch.nn import AdaptiveMaxPool2d, AdaptiveAvgPool2d
from torch.nn.init import xavier_normal, kaiming_normal

from . import Flatten


def get_squeezenet_features(input_n_channels):
    return (Conv2d(input_n_channels, 64, kernel_size=3, stride=1),
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
            Fire(512, 64, 256, 256))


class IcebergSqueezeNet(Module):

    def __init__(self, input_n_channels):
        super(IcebergSqueezeNet, self).__init__()

        self.features = Sequential(*get_squeezenet_features(input_n_channels))

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


class IcebergSqueezeNet3(Module):

    def __init__(self, input_n_channels=2):
        super(IcebergSqueezeNet3, self).__init__()

        self.features = Sequential(*get_squeezenet_features(input_n_channels))

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
                    torch.nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, a):
        x = self.features(x)
        y = self.classifier(x)
        return y


class IcebergSqueezeNetMax(Module):

    def __init__(self, input_n_channels=2):
        super(IcebergSqueezeNetMax, self).__init__()

        self.features = Sequential(*get_squeezenet_features(input_n_channels))

        # Final convolution is initialized differently form the rest
        final_conv = Conv2d(512, 2, kernel_size=1)
        self.classifier = Sequential(
            Dropout(p=0.5),
            final_conv,
            ReLU(inplace=True),
            AdaptiveMaxPool2d(1),
            Flatten()
        )
        # init weights
        for m in self.modules():
            if isinstance(m, Conv2d):
                if m is final_conv:
                    torch.nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    torch.nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, a):
        x = self.features(x)
        y = self.classifier(x)
        return y



class IcebergSqueezeNet2(Module):

    def __init__(self, input_n_channels):
        super(IcebergSqueezeNet2, self).__init__()

        self.features1 = Sequential(*get_squeezenet_features(input_n_channels))
        self.features2 = Sequential(*get_squeezenet_features(input_n_channels))

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
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x = torch.cos(x1 - x2)
        y = self.classifier(x)
        return y


class IcebergSqueezeNet3b(Module):

    def __init__(self, input_n_channels, n_features=10):
        super(IcebergSqueezeNet3b, self).__init__()

        self.features = Sequential(*get_squeezenet_features(input_n_channels))

        # Final convolution is initialized differently form the rest
        final_conv = Conv2d(512, n_features, kernel_size=1)
        self.classifier = Sequential(
            Dropout(p=0.5),
            final_conv,
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.final_classifier = Linear(n_features + 1, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                xavier_normal(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, a):
        f0 = self.features(x)
        f1 = self.classifier(f0)
        if len(a.size()) == 1:
            a = a.unsqueeze(dim=1)
        f = torch.cat((f1, a), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergSqueezeNet3b1(Module):

    def __init__(self, input_n_channels, n_features=10):
        super(IcebergSqueezeNet3b1, self).__init__()

        self.features = Sequential(*get_squeezenet_features(input_n_channels))

        # Final convolution is initialized differently form the rest
        final_conv = Conv2d(512, n_features, kernel_size=1)
        self.classifier = Sequential(
            Dropout(p=0.5),
            final_conv,
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten(),
            ReLU(inplace=True)
        )
        self.final_classifier = Linear(n_features + 1, 1)
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
