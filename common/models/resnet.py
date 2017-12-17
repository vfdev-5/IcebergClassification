
import math

import torch
from torch.nn import Module
from torch.nn import Sequential, Linear, Conv2d, ReLU, BatchNorm2d, AdaptiveAvgPool2d
from torchvision.models.resnet import Bottleneck

from . import Flatten


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class IcebergResNet50(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergResNet50, self).__init__()
        self.inplanes = 64

        self.stem = Sequential(
            Conv2d(input_n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )

        layers = [3, 4, 6, 3]
        block = Bottleneck

        self.features = Sequential(
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2)
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(512 * block.expansion, n_classes)
        )
        initialize_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def forward(self, x, a):
        x = self.stem(x)
        x = self.features(x)
        y = self.classifier(x)
        return y


class IcebergResNet(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergResNet, self).__init__()

        self.stem = Sequential(
            Conv2d(input_n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )

        self.inplanes = 64
        layers = [3, 4]
        block = Bottleneck

        self.features = Sequential(
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1])
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(128 * block.expansion, n_classes)
        )
        initialize_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def forward(self, x, a):
        x = self.stem(x)
        x = self.features(x)
        y = self.classifier(x)
        return y


class IcebergResNetThreePools(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergResNetThreePools, self).__init__()

        self.stem = Sequential(
            Conv2d(input_n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )

        self.inplanes = 64
        layers = [3, 4, 6]
        block = Bottleneck

        self.features = Sequential(
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2)
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(256 * block.expansion, n_classes)
        )
        initialize_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def forward(self, x, a):
        x = self.stem(x)
        x = self.features(x)
        y = self.classifier(x)
        return y


class IcebergResNet101(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergResNet101, self).__init__()
        self.inplanes = 64

        self.stem = Sequential(
            Conv2d(input_n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True)
        )

        layers = [3, 4, 23, 3]
        block = Bottleneck

        self.features = Sequential(
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2)
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(512 * block.expansion, n_classes)
        )
        initialize_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def forward(self, x, a):
        x = self.stem(x)
        x = self.features(x)
        y = self.classifier(x)
        return y
