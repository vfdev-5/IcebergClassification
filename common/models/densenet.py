

from torchvision.models.densenet import OrderedDict, _DenseBlock, _Transition

import torch
from torch.nn import Module
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, BatchNorm2d, Linear
from torch.nn import AdaptiveAvgPool2d
from torch.nn.functional import relu
from torch.nn.init import xavier_normal, kaiming_normal

from . import Flatten


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
            ('pool0', MaxPool2d(kernel_size=3, stride=2, padding=1)),
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

        # Linear layer
        self.classifier = Sequential(
            BatchNorm2d(num_features),
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(num_features, 2)
        )

    def forward(self, x, a):
        x = self.features(x)
        y = self.classifier(x)
        return y


class IcebergDenseNet2(Module):

    def __init__(self, input_n_channels, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)):
        super(IcebergDenseNet2, self).__init__()

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
        self.classifier_p1 = Sequential(
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.classifier_p2 = Linear(num_features + 2, 2)

    def forward(self, x, a):
        x1 = self.features(x)
        f1 = self.classifier_p1(x1)
        # Extract mean positive value from each band
        x_pos = relu(x)
        f2 = torch.mean(x_pos.view(x.size(0), x.size(1), -1), dim=2)
        # Join
        f = torch.cat((f1, f2), dim=1)
        # Classify
        y = self.classifier_p2(f)
        return y


class IcebergDenseNet3(Module):

    def __init__(self, input_n_channels, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)):
        super(IcebergDenseNet3, self).__init__()

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
        self.classifier_p1 = Sequential(
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(num_features, 4)
        )
        self.classifier_p2 = Linear(4 + 2, 2)

    def forward(self, x, a):
        x1 = self.features(x)
        f1 = self.classifier_p1(x1)
        # Extract mean positive value from each band
        x_pos = relu(x)
        f2 = torch.mean(x_pos.view(x.size(0), x.size(1), -1), dim=2)
        # Join
        f = torch.cat((f1, f2), dim=1)
        # Classify
        y = self.classifier_p2(f)
        return y


class IcebergDenseNet4(Module):

    def __init__(self, input_n_channels, n_features=10, n_classes=1,
                 num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)):
        super(IcebergDenseNet4, self).__init__()

        bn_size = 4
        drop_rate = 0

        # First convolution
        self.features = Sequential(OrderedDict([
            ('conv0', Conv2d(input_n_channels, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', BatchNorm2d(num_init_features)),
            ('relu0', ReLU(inplace=True)),
            ('pool0', MaxPool2d(kernel_size=3, stride=2, padding=1)),
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
        self.features.add_module('relu5', ReLU(inplace=True))

        # Linear layer
        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(num_features, n_features),
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
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
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


class IcebergDenseNet161(Module):

    def __init__(self, input_n_channels, n_classes=1,
                 num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)):
        super(IcebergDenseNet161, self).__init__()

        bn_size = 4
        drop_rate = 0

        # First convolution
        self.features = Sequential(
            Conv2d(input_n_channels, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(num_init_features),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

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
        self.features.add_module('relu5', ReLU(inplace=True))

        # Linear layer
        self.classifier = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(num_features, n_classes)
        )
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
        f = self.features(x)
        y = self.classifier(f)
        return y
