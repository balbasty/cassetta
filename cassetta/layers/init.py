__all__ = [
    'InitWeightsKaiming',
]
import torch
from torch import nn
from .conv import _ConvBlockBase


class InitWeightsBase:
    """Base class for weights initializers"""

    def __init__(self):
        self.initializers = {}

    @torch.no_grad()
    def __call__(self, module):
        for klass, init in self.initializers.items():
            if isinstance(module, klass):
                init(module)


class InitWeightsKaiming(InitWeightsBase):
    """Init ConvBlocks using Kaiming He's method."""

    def __init__(self, neg_slope=1e-2):
        super().__init__()
        self.neg_slope = neg_slope
        self.initializers[_ConvBlockBase] = self.init_conv

    def init_conv(self, module):
        module = module.getattr('conv', None)
        if module:
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=self.neg_slope)
            if module.bias:
                module.bias = nn.init.constant_(module.bias, 0)
