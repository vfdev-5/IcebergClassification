
import torch
from torch.nn import Module
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, BatchNorm2d, Linear, Dropout
from torch.nn import AdaptiveAvgPool2d, ConvTranspose2d, Tanh, Sigmoid, LeakyReLU
from torch.nn.functional import relu
from torch.nn.init import xavier_normal, kaiming_normal, xavier_uniform

from . import Flatten

def initialize_weights(modules, 
                       conv_init=xavier_uniform, deconv_init=xavier_uniform, 
                       linear_init=xavier_uniform):
    for m in modules:
        if isinstance(m, Conv2d):
            conv_init(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, ConvTranspose2d):
            deconv_init(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            linear_init(m.weight.data)
            m.bias.data.zero_()


class IcebergShipGenerator(Module):

    @staticmethod
    def deconv(in_features, out_features, kernel_size, stride, padding):
        return Sequential(
            ConvTranspose2d(in_features, out_features, 
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False),
            BatchNorm2d(out_features),
            ReLU(True)
        )

    def __init__(self, nz, n_out_channels, n_features=64):
        """
        :param nz: integer, size of latent z vector (input for the generator)
        :param n_features: integer, number of features at the first layer
        """
        super(IcebergShipGenerator, self).__init__()

        self.generator = Sequential(
            self.deconv(nz, n_features * 8, 5, 1, 0),
            # Size : n_features * 8 x 5 x 5
            self.deconv(n_features * 8, n_features * 4, 5, 2, 0),
            # Size : n_features * 4 x 13 x 13
            self.deconv(n_features * 4, n_features * 4, 5, 2, 0),
            # Size : n_features * 4 x 29 x 29
            self.deconv(n_features * 4, n_features * 2, 5, 2, 0),
            # Size : n_features * 2 x 61 x 61

            self.deconv(n_features * 2, n_features * 2, 5, 1, 0),
            # Size : n_features * 2 x 65 x 65
            self.deconv(n_features * 2, n_features, 5, 1, 0),
            # Size : n_features x 69 x 69

            self.deconv(n_features, n_features, 3, 1, 0),
            # Size : n_features x 71 x 71
            self.deconv(n_features, n_features, 3, 1, 0),
            # Size : n_features x 73 x 73
            ConvTranspose2d(n_features, n_out_channels, 
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=False),            
            # Size : n_out_channels x 75 x 75
            Conv2d(2, 2, kernel_size=1)
        )

        initialize_weights(self.modules(), conv_init=xavier_normal, linear_init=xavier_normal)

    def forward(self, x):
        return self.generator(x)


class IcebergShipDescriminator(Module):

    @staticmethod
    def conv(in_features, out_features, kernel_size, stride, padding):
        return Sequential(
            Conv2d(in_features, out_features, 
                   kernel_size=kernel_size, 
                   stride=stride,
                   padding=padding,
                   bias=True),
            BatchNorm2d(out_features),
            LeakyReLU(0.2, inplace=True)            
        )

    def __init__(self, n_in_channels, n_features=16):
        super(IcebergShipDescriminator, self).__init__()

        self.descriminator = Sequential(
            
            Conv2d(n_in_channels, n_features, 
                   kernel_size=5,
                   stride=2,
                   padding=2,
                   bias=True),
            LeakyReLU(0.2, inplace=True),
            # Size: n_features x 38 x 38

            self.conv(n_features, n_features * 2, 3, 1, 1),
            MaxPool2d(2),
            # Size: n_features * 2 x 19 x 19

            self.conv(n_features * 2, n_features * 2, 3, 1, 1),
            MaxPool2d(2),
            # Size: n_features * 2 x 9 x 9

            self.conv(n_features * 2, n_features * 4, 3, 1, 1),
            self.conv(n_features * 4, n_features * 4, 3, 1, 1),
            MaxPool2d(2),
            # Size: n_features * 2 x 4 x 4

            self.conv(n_features * 4, n_features * 4, 3, 1, 1),
            self.conv(n_features * 4, n_features * 4, 3, 1, 1),
            MaxPool2d(2),
            # Size: n_features * 4 x 2 x 2

            self.conv(n_features * 4, n_features * 8, 3, 1, 1),
            self.conv(n_features * 8, n_features * 8, 3, 1, 1),            
            AdaptiveAvgPool2d(1),
            # Size: n_features * 8 x 1 x 1
            Flatten(),
            Linear(n_features * 8, 1)            
        )

        initialize_weights(self.modules(), conv_init=xavier_normal, linear_init=xavier_normal)

    def forward(self, x):
        return self.descriminator(x)


