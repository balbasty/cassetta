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
__all__ = ['make_loss']
from cassetta.io.utils import import_fullname
from cassetta.core.utils import import_submodules
from cassetta.core.typing import LossType
from torch.nn import Module

import_submodules([
    'base',
    'segmentation',
], __name__, __all__, True)


def make_loss(loss: LossType, *args, **kwargs):
    """
    Instantiate a loss

    A loss can be:

    - the name of a cassetta loss, such as `"DiceLoss"`;
    - the fully qualified path to a model, such as
    `"cassetta.losses.DiceLoss"`, or `"monai.losses.GeneralizedDiceLoss"`;
    - a [`nn.Module`][torch.nn.Module] subclass, such as
    [`DiceLoss`][cassetta.losses.DiceLoss];
    - an already instantiated [`nn.Module`][torch.nn.Module], such as
    [`DiceLoss()`][cassetta.losses.DiceLoss].

    Parameters
    ----------
    loss : LossType
        Instantiated or non-instantiated loss
    *args : tuple
        Positional arguments pass to the loss constructor
    **kwargs : dict
        Keyword arguments pass to the loss constructor

    Returns
    -------
    loss : nn.Module
        Instantiated loss
    """
    reentrant = kwargs.pop('__reentrant', False)
    if isinstance(loss, str):
        if not reentrant:
            kwargs['__reentrant'] = True
            for prefix in ('', 'cassetta.', 'cassetta.losses.', 'torch.nn.'):
                try:
                    return make_loss(prefix + loss, *args, **kwargs)
                except Exception:
                    pass
        loss = import_fullname(loss)
    if not isinstance(loss, Module):
        loss = loss(*args, **kwargs)
    if not isinstance(loss, Module):
        raise ValueError('Instantiated object is not a Module')
    return loss
