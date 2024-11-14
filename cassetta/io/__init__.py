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
__all__ = ["make_loadable"]

import torch
from cassetta.core.utils import import_submodules
from cassetta.core.typing import Type
from .loadable import DynamicLoadableMixin
from .optim import DynamicLoadableOptimizerMixin
from . import loadable

import_submodules([
    'loadable',
    'modules',
    'optim',
    'utils',
], __name__, __all__, True)


def make_loadable(klass, save_args: bool = True) -> Type[DynamicLoadableMixin]:
    """
    Create a loadable variant of an existing
    [`Module`][torch.nn.Module] or [`Optimizer`][torch.optim.Optimizer]
    subclass.

    Example
    -------
    ```python
    mymodel = make_loadable(nn.Sequential)(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    ```

    Parameters
    ----------
    klass : type
        A [`Module`][torch.nn.Module] or [`Optimizer`][torch.optim.Optimizer]
        subclass

    Returns
    -------
    loadable_klass : type
        A `(DynamicLoadableMixin, klass)` subclass.
    """
    if issubclass(klass, torch.optim.Optimizer):
        class Kls(DynamicLoadableOptimizerMixin, klass, save_args=save_args):
            ...
    else:
        class Kls(DynamicLoadableMixin, klass, save_args=save_args):
            ...

    Kls.__name__ = "Loadable" + klass.__name__
    Kls.__qualname__ = "Loadable" + klass.__name__
    return Kls


# monkey-patch make_loadable
# This is so make_loadable works on both optimizers (which have a special
# serialize/load logic) and modules.
loadable.make_loadable = make_loadable
