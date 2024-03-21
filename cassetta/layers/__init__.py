"""
This module implements a variety of very simple `nn.Module` that can be
though of as "layers" in a network.

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

from . import activations   # noqa: F401
from . import attention     # noqa: F401
from . import conv          # noqa: F401
from . import convblocks    # noqa: F401
from . import dropout       # noqa: F401
from . import interpol      # noqa: F401
from . import linear        # noqa: F401
from . import norm          # noqa: F401
from . import simple        # noqa: F401
from . import updown        # noqa: F401

from .activations import *  # noqa: F401, F403
from .attention import *    # noqa: F401, F403
from .conv import *         # noqa: F401, F403
from .convblocks import *   # noqa: F401, F403
from .dropout import *      # noqa: F401, F403
from .interpol import *     # noqa: F401, F403
from .linear import *       # noqa: F401, F403
from .norm import *         # noqa: F401, F403
from .simple import *       # noqa: F401, F403
from .updown import *       # noqa: F401, F403
