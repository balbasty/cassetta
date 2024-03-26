"""
## Overview

Thos module contains routines for data input/output.

It also defines mixins and utilities for making load/unloading of
modules and checkpoints easier than in pure PyTorch.

Modules
-------
modules
    Advanced loading/unloading of modules & models
"""
__all__ = []

from cassetta.core.utils import import_submodules

import_submodules([
    'modules',
    'utils',
], __name__, __all__, True)
