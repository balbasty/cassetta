"""
## Overview

Models are end-to-end, task-specific networks. They usually rely on
a more generic [**backbone**](../backbones) architecture.

Modules
-------
segmentation
    Models for semantic segmentation
registration
    Models for image registration
"""

from . import segmentation      # noqa: F401
from . import registration      # noqa: F401

from .segmentation import *     # noqa: F401, F403
from .registration import *     # noqa: F401, F403
