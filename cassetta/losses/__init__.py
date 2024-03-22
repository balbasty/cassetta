"""
## Overview

Losses are modules that return a scalar tensor and must be differentiable.

Modules
-------
base
    Base class for all losses
segmentation
    Losses for semantic segmentation
"""

from . import base              # noqa: F401
from . import segmentation      # noqa: F401

from .base import *             # noqa: F401, F403
from .segmentation import *     # noqa: F401, F403
