"""
## Overview

Backbones are complex/deep architectures, made of a bunch of layers.

In our philosophy, backbones are task-independant, and do not care
about the number of input and output channels of the problem at hand.
Instead they typically map an input feature space to an output feature
space.

[**Models**](../models) would wrap a backbone between a
feature-extraction layer (a single convolution without activation,
possibly with a somewhat large kernel size) and a feature-mapping layer
(a single 1x1x1 convolution, possibly followed by a task-specific
activation like a SoftMax).

Modules
-------
fcn
    Fully convolutional encoders and decoders
unet
    U-Nets: autoencoder wirth skip connections
atrous
    Networks that use dilated convolutions
"""
__all__ = []
from cassetta.core.utils import import_submodules

import_submodules([
    'fcn',
    'unet',
    'atrous',
], __name__, __all__, True)
