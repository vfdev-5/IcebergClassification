
from torch.nn import Module


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


from .densenet import *
from .inceptionv4 import *
from .squeezenet import *
from .vgg import *


