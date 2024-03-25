"""
## Overview

Models are end-to-end, task-specific networks. They usually rely on
a more generic [**backbone**](../backbones) architecture.

Modules
-------
segmentation
    Models for semantic segmentation
"""

from . import segmentation      # noqa: F401
from .segmentation import *     # noqa: F401, F403
