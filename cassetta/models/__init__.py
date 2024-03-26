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
__all__ = ['make_model']
from torch.nn import Module
from cassetta.io.utils import import_fullname
from cassetta.core.utils import import_submodules
from cassetta.core.typing import ModelType


import_submodules(['segmentation', 'registration'], __name__, __all__, True)


def make_model(model: ModelType, *args, **kwargs):
    """
    Instantiate a model

    A model can be:

    - the name of a cassetta model, such as `"SegNet"`;
    - the fully qualified path to a model, such as
    `"cassetta.models.ElasticRegNet"`, or `"monai.networks.nets.ResNet"`;
    - a [`nn.Module`][torch.nn.Module] subclass, such as
    [`SegNet`][cassetta.models.SegNet];
    - an already instantiated [`nn.Module`][torch.nn.Module], such as
    [`SegNet(3, 1, 5)`][cassetta.models.SegNet].

    Parameters
    ----------
    model : ModelType
        Instantiated or non-instantiated model
    *args : tuple
        Positional arguments pass to the model constructor
    **kwargs : dict
        Keyword arguments pass to the model constructor

    Returns
    -------
    model : nn.Module
        Instantiated model
    """
    reentrant = kwargs.pop('__reentrant', False)
    if isinstance(model, str):
        if not reentrant:
            kwargs['__reentrant'] = True
            for prefix in ('', 'cassetta.', 'cassetta.models.', 'torch.nn.'):
                try:
                    return make_model(prefix + model, *args, **kwargs)
                except Exception:
                    pass
        model = import_fullname(model)
    if not isinstance(model, Module):
        model = model(*args, **kwargs)
    if not isinstance(model, Module):
        raise ValueError('Instantiated object is not a Module')
    return model
