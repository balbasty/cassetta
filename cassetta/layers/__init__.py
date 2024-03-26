"""
## Overview

Layers are relatively simple modules or sequences of modules. They
should be used as basic blocks for building [**backbones**](../backbones).

Modules
-------
activations
    Activation functions
attention
    Attention layers (squeeze & excite, dot-product, multi-head, ...)
conv
    Basic N-dimensional convolution layers
convblocks
    Building blocks for convolutional networks
dropout
    N-dimensional dropout layers
interpol
    N-dimensional interpolation and grid sampling
linear
    Linear layer (slightly more practical than PyTorch's)
simple
    A bunch of embarassingly simple layers (Cat, Sum, ...)
updown
    Different ways to upsample and downsample
"""
__all__ = []
from cassetta.core.utils import import_submodules

import_submodules([
    'activations',
    'attention',
    'conv',
    'convblocks',
    'dropout',
    'interpol',
    'linear',
    'norm',
    'simple',
    'updown',
], __name__, __all__, True)
