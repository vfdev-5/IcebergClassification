

import os
import sys

from torchvision.models.densenet import OrderedDict, _DenseBlock, _Transition, densenet169
from torchvision.models.squeezenet import Fire

import torch
from torch.nn import Module
from torch.nn import Sequential, Dropout, Conv2d, ReLU, MaxPool2d, BatchNorm2d, Linear, AvgPool2d
from torch.nn import AdaptiveMaxPool2d, AdaptiveAvgPool2d

# Setup pretrained-models.pytorch
assert 'PRETRAINED_MODELS' in os.environ
sys.path.append(os.environ['PRETRAINED_MODELS'])

from pretrainedmodels.inceptionv4 import Inception_A, Inception_B, Reduction_A, Reduction_B, Inception_C, BasicConv2d


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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


class IcebergDenseNet(Module):

    def __init__(self, input_n_channels, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)):
        super(IcebergDenseNet, self).__init__()

        bn_size = 4
        drop_rate = 0

        # First convolution
        self.features = Sequential(OrderedDict([
            ('conv0',
             Conv2d(input_n_channels, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', BatchNorm2d(num_init_features)),
            ('relu0', ReLU(inplace=True)),
            ('pool0', MaxPool2d(kernel_size=3, stride=1, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', BatchNorm2d(num_features))

        # Linear layer
        self.classifier = Sequential(
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(num_features, 2)
        )

    def forward(self, x, a):
        x = self.features(x)
        y = self.classifier(x)
        return y


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