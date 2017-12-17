
import torch
from torch.nn import Module
from torch.nn import Sequential, Linear, Conv2d, ReLU, Dropout, BatchNorm2d, MaxPool2d
from torch.nn.init import xavier_uniform, xavier_normal, kaiming_normal
from . import Flatten

from torchvision.models.vgg import vgg16, vgg16_bn


def make_layers(in_channels, cfg, batch_norm=False, replace_maxpool=False):
    layers = []
    for v in cfg:
        if v == 'M':
            if replace_maxpool:
                conv2d_strided = Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
                if batch_norm:
                    layers += [conv2d_strided, BatchNorm2d(in_channels), ReLU(inplace=True)]
                else:
                    layers += [conv2d_strided, ReLU(inplace=True)]
            else:
                layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return Sequential(*layers)


cfg = {
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
'1Pool': [64, 64, 'M', 128, 128, 256, 256],
'2Pools': [64, 64, 'M', 128, 128, 'M', 256, 256, 256],
'3Pools': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512],
}


def initialize_weights(modules, conv_init=xavier_uniform, linear_init=xavier_uniform):
    for m in modules:
        if isinstance(m, Conv2d):
            conv_init(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            linear_init(m.weight.data)
            m.bias.data.zero_()


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
        initialize_weights(self.modules(), conv_init=xavier_uniform, linear_init=xavier_uniform)

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
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_uniform)

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
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergVGGTwoPools(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergVGGTwoPools, self).__init__()

        self.features = make_layers(input_n_channels, cfg['2Pools'])
        self.classifier = Sequential(
            Flatten(),
            Linear(256 * 18 * 18, 256),
            ReLU(True),
            Dropout(),
            Linear(256, 128),
            ReLU(True),
            Dropout(),
            Linear(128, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergVGGThreePools(Module):

    def __init__(self, input_n_channels, input_size=(75, 75), n_classes=2):
        super(IcebergVGGThreePools, self).__init__()
        feature_size = (input_size[0] // 2**3, input_size[1] // 2**3)
        self.features = make_layers(input_n_channels, cfg['3Pools'])
        self.classifier = Sequential(
            Flatten(),
            Linear(512 * feature_size[0] * feature_size[1], 256),
            ReLU(True),
            Dropout(),
            Linear(256, 128),
            ReLU(True),
            Dropout(),
            Linear(128, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergVGG13WithAnglesAndStats(Module):

    def __init__(self, input_n_channels, input_size=(75, 75), n_classes=2):
        super(IcebergVGG13WithAnglesAndStats, self).__init__()

        self.features = make_layers(input_n_channels, cfg['B'], replace_maxpool=True)
        self.features.add_module("%i" % len(self.features), Flatten())
        self.metadata_features = Sequential(
            Linear(7, 50),
            ReLU(True),
            Dropout(),
            Linear(50, 25),
            ReLU(True),
            Dropout()
        )

        feature_size = (int(input_size[0] / (2 ** 4) + 0.5), int(input_size[1] / (2 ** 4) + 0.5))
        self.classifier = Sequential(
            Linear(512 * feature_size[0] * feature_size[1], 512),
            ReLU(True),
            Dropout(),
            Linear(512, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 25)
        )
        self.final_classifier = Sequential(
            Linear(25 + 25, 50),
            ReLU(True),
            Dropout(),
            Linear(50, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, x_metadata):
        # x_metadata = [inc_angle, b1_min, b1_mean, b1_max, b2_min, b2_mean, b2_max]
        f1 = self.features(x)
        f1 = self.classifier(f1)
        f2 = self.metadata_features(x_metadata)
        f = torch.cat((f1, f2), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergVGGLike(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergVGGLike, self).__init__()
        # Inspired from https://www.kaggle.com/a45632/keras-starter-4l-added-performance-graph
        self.features = Sequential(
            # Layer 1
            Conv2d(input_n_channels, 64, kernel_size=3),
            ReLU(True),
            MaxPool2d(3, stride=2),
            Dropout(0.2),
            # Layer 2
            Conv2d(64, 128, kernel_size=3),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Dropout(0.2),
            # Layer 3
            Conv2d(128, 128, kernel_size=3),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Dropout(0.3),
            # Layer 4
            Conv2d(128, 64, kernel_size=3),
            ReLU(True),
            MaxPool2d(2, stride=2),
            Dropout(0.3)
        )

        self.classifier = Sequential(
            Flatten(),
            Linear(64 * 2 * 2, 512),
            ReLU(True),
            Dropout(0.2),
            Linear(512, 256),
            ReLU(True),
            Dropout(0.2),
            Linear(256, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergVGG13BNWithAnglesAndStats(Module):

    def __init__(self, input_n_channels, n_classes=2):
        super(IcebergVGG13BNWithAnglesAndStats, self).__init__()

        self.features = make_layers(input_n_channels, cfg['B'], batch_norm=True, replace_maxpool=True)
        self.features.add_module("%i" % len(self.features), Flatten())
        self.metadata_features = Sequential(
            Linear(7, 50),
            ReLU(True),
            Dropout(),
            Linear(50, 25),
            ReLU(True),
            Dropout()
        )
        self.classifier = Sequential(
            Linear(512 * 5 * 5, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 512),
            ReLU(True),
            Dropout(),
            Linear(512, 25)
        )
        self.final_classifier = Sequential(
            Linear(25 + 25, 50),
            ReLU(True),
            Dropout(),
            Linear(50, n_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, x_metadata):
        # x_metadata = [inc_angle, b1_min, b1_mean, b1_max, b2_min, b2_mean, b2_max]
        f1 = self.features(x)
        f1 = self.classifier(f1)
        f2 = self.metadata_features(x_metadata)
        f = torch.cat((f1, f2), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergPretrainedVGG16(Module):

    def __init__(self, input_n_channels, n_classes=2, pretrained=True, n_class_features=1024):
        super(IcebergPretrainedVGG16, self).__init__()

        model = vgg16(pretrained=pretrained)
        features = [f for f in model.features]
        if input_n_channels != 3:
            features[0] = Conv2d(input_n_channels, 64, kernel_size=3, padding=1)

        features = features[:-1]  # Remove the last max pooling
        self.features = Sequential(*features)

        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_classes)
        )
        self._initialize_weights(input_n_channels, pretrained)

    def _initialize_weights(self, input_n_channels, pretrained):

        if pretrained:
            for m in self.classifier:
                if isinstance(m, Linear):
                    xavier_normal(m.weight)
                    m.bias.data.zero_()

            if input_n_channels != 3:
                kaiming_normal(self.features[0].weight)
                if self.features[0].bias is not None:
                    self.features[0].bias.data.zero_()
        else:
            initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergPretrainedVGG16BN(Module):

    def __init__(self, input_n_channels, n_classes=2, pretrained=True):
        super(IcebergPretrainedVGG16BN, self).__init__()

        model = vgg16_bn(pretrained=pretrained)
        features = [f for f in model.features]
        if input_n_channels != 3:
            features[0] = Conv2d(input_n_channels, 64, kernel_size=3, padding=1)

        features = features[:-1]  # Remove the last max pooling
        self.features = Sequential(*features)

        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, 1024),
            ReLU(True),
            Dropout(),
            Linear(1024, 1024),
            ReLU(True),
            Dropout(),
            Linear(1024, n_classes)
        )
        self._initialize_weights(input_n_channels, pretrained)

    def _initialize_weights(self, input_n_channels, pretrained):

        if pretrained:
            for m in self.classifier:
                if isinstance(m, Linear):
                    xavier_normal(m.weight)
                    m.bias.data.zero_()

            if input_n_channels != 3:
                kaiming_normal(self.features[0].weight)
                if self.features[0].bias is not None:
                    self.features[0].bias.data.zero_()
        else:
            initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, a):
        f = self.features(x)
        y = self.classifier(f)
        return y


class IcebergPretrainedVGG16WithAngleAndStats(Module):

    def __init__(self, input_n_channels, n_classes=2, pretrained=True, n_class_features=1024):
        super(IcebergPretrainedVGG16WithAngleAndStats, self).__init__()

        model = vgg16(pretrained=pretrained)
        features = [f for f in model.features]
        if input_n_channels != 3:
            features[0] = Conv2d(input_n_channels, 64, kernel_size=3, padding=1)

        features = features[:-1]  # Remove the last max pooling
        self.features = Sequential(*features)

        self.classifier = Sequential(
            Flatten(),
            Linear(512 * 4 * 4, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_class_features),
            ReLU(True),
            Dropout(),
        )
        self.metadata_features = Sequential(
            Linear(7, 50),
            ReLU(True),
            Dropout(),
            Linear(50, 25),
            ReLU(True),
            Dropout()
        )
        self.final_classifier = Sequential(
            Linear(n_class_features + 25, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_classes)
        )
        self._initialize_weights(input_n_channels, pretrained)

    def _initialize_weights(self, input_n_channels, pretrained):
        if pretrained:
            modules = [self.classifier, self.metadata_features, self.final_classifier]
            for module in modules:
                for m in module:
                    if isinstance(m, Linear):
                        xavier_normal(m.weight)
                        m.bias.data.zero_()
                    elif isinstance(m, BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            if input_n_channels != 3:
                kaiming_normal(self.features[0].weight)
                if self.features[0].bias is not None:
                    self.features[0].bias.data.zero_()
        else:
            initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, x_metadata):
        # x_metadata = [inc_angle, b1_min, b1_mean, b1_max, b2_min, b2_mean, b2_max]
        f1 = self.features(x)
        f1 = self.classifier(f1)
        f2 = self.metadata_features(x_metadata)
        f = torch.cat((f1, f2), dim=1)
        y = self.final_classifier(f)
        return y


class IcebergPretrainedVGG16WithAngleAndStats2(Module):

    def __init__(self, input_n_channels, n_classes=2, pretrained=True, n_class_features=1024):
        super(IcebergPretrainedVGG16WithAngleAndStats2, self).__init__()

        model = vgg16(pretrained=pretrained)
        features = [f for f in model.features]
        if input_n_channels != 3:
            features[0] = Conv2d(input_n_channels, 64, kernel_size=3, padding=1)

        features = features[:-1]  # Remove the last max pooling
        self.features = Sequential(*features)
        self.features.add_module("%i" % len(self.features), Flatten())

        self.metadata_features = Sequential(
            Linear(7, 50),
            ReLU(True),
            Dropout(),
            Linear(50, 25),
            ReLU(True),
            Dropout()
        )

        self.classifier = Sequential(
            Linear(512 * 4 * 4 + 25, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_class_features),
            ReLU(True),
            Dropout(),
            Linear(n_class_features, n_classes)
        )
        self._initialize_weights(input_n_channels, pretrained)

    def _initialize_weights(self, input_n_channels, pretrained):
        if pretrained:
            modules = [self.classifier, self.metadata_features]
            for module in modules:
                for m in module:
                    if isinstance(m, Linear):
                        xavier_normal(m.weight)
                        m.bias.data.zero_()
                    elif isinstance(m, BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            if input_n_channels != 3:
                kaiming_normal(self.features[0].weight)
                if self.features[0].bias is not None:
                    self.features[0].bias.data.zero_()
        else:
            initialize_weights(self.modules(), conv_init=kaiming_normal, linear_init=xavier_normal)

    def forward(self, x, x_metadata):
        # x_metadata = [inc_angle, b1_min, b1_mean, b1_max, b2_min, b2_mean, b2_max]
        f1 = self.features(x)
        f2 = self.metadata_features(x_metadata)
        f = torch.cat((f1, f2), dim=1)
        y = self.classifier(f)
        return y
