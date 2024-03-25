__all__ = [
    'LoadableMixin',
    'load_module',
    'make_loadable',
]
import torch
from torch import nn
from typing import Union, IO
from warnings import warn
from inspect import signature
from importlib import import_module


def load_module(model_state: Union[str, IO]) -> nn.Module:
    """
    Load and build a model/module form a file.

    Parameters
    ----------
    model_state : file_like or dict
        Model state, or path to model file, or opened file object.

    Returns
    -------
    model : nn.Module
        Instantiated model
    """
    return LoadableMixin.load_from(model_state)


def make_loadable(klass):
    """
    Create a loadable variant of an existing `nn.Module` subclass.

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
        A `nn.Module` subclass

    Returns
    -------
    loadable_klass : type
        A `(LoadableMixin, klass)` subclass.
    """
    return type(
        'Loadable' + klass.__name__, (LoadableMixin, klass), {}
    )


class LoadableMixin:
    """
    A mixin to make a Module loadable.

    Example
    -------
    ```python
    class MyModel(LoadableMixin, nn.Sequential):

        def __init__(self):
            super().__init__(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

    # Save model
    model = MyModel()
    model.save('path/to/modelfile.tar')

    # Build and load model
    model = load_module('path/to/modelfile.tar')

    # Build and load model (alternative)
    model = LoadableMixin.load_from('path/to/modelfile.tar')

    # Build and load model with type hint
    model = MyModel.load_from('path/to/modelfile.tar')

    # Load weights in instantiated model
    model = MyModel()
    model.load('path/to/modelfile.tar')
    ```
    """

    __version__ = '1.0'
    """Current version of the LoadableMixin format"""

    @staticmethod
    def _nested_serialize(obj):
        if isinstance(obj, nn.Module):
            if not isinstance(obj, LoadableMixin):
                raise TypeError(
                    f'Only loadable modules (which inherit from '
                    f'LoadableMixin) can be serialized. Type {type(obj)} '
                    f'does not.')
            return obj.serialize()
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(LoadableMixin._nested_serialize, obj))
        if isinstance(obj, dict):
            return type(obj)({
                key: LoadableMixin._nested_serialize(value)
                for key, value in obj.items()
            })
        return obj

    @staticmethod
    def _nested_unserialize(obj, *, klass=None):
        if isinstance(obj, dict):
            if '__LoadableMixing__' in obj:
                state = obj
                if klass is None:
                    try:
                        klass = import_module(state['module'])
                        strklass = list(state['class'].split('.'))
                        while strklass:
                            klass = getattr(klass, strklass.pop(0))
                    except Exception:
                        raise ImportError(
                            'Could not import type', state['class'],
                            'from module', state['module'])
                args = LoadableMixin._nested_unserialize(state['args'])
                kwargs = LoadableMixin._nested_unserialize(state['kwargs'])
                obj = klass(*args, **kwargs)
                obj.load(state['state'])
                return obj
            else:
                return type(obj)({
                    key: LoadableMixin._nested_unserialize(value)
                    for key, value in obj.items()
                })
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(LoadableMixin._nested_unserialize, obj))
        return obj

    @classmethod
    def save_args(cls, init):
        """
        A decorator for `__init__` methods, that saves their arguments.

        Example
        -------
        ```python
        class MyConv(LoadableMixin, nn.Module):

            @LoadableMixin.save_args
            def __init__(self, inp_channels, out_channels):
                self.conv = nn.Conv3d(inp_channels, out_channels, 3)
        ```
        """
        def wrapper(self, *args, **kwargs):
            self._args = cls._nested_serialize(args)
            self._kwargs = cls._nested_serialize(kwargs)
            return init(self, *args, **kwargs)
        return wrapper

    def serialize(self) -> dict:
        """
        Return the object as a serialized dictionary from which the
        object can be reconstructed.

        Returns
        -------
        loadable_state : dict
            The module state, along with its constructor's parameters.
        """
        return {
            '__LoadableMixin__': LoadableMixin.__version__,
            'module': type(self).__module__,
            'qualname': type(self).__qualname__,
            'args': getattr(self, '_args', tuple()),
            'kwargs': getattr(self, '_kwargs', dict()),
            'state': self.state_dict(),
        }

    def save(self, path: Union[str, IO]) -> None:
        """
        Save the model (and everything needed to build it) in a file.

        Parameters
        ----------
        path : file_like
            Path to output module file, or opened file object.
        """
        if not hasattr(self, '_args') and \
                len(signature(self.__init__).parameters) > 1:
            warn(f"Object of type {type(self).__name__} does not have "
                 f"saved arguments. Did you decorate __init__ with "
                 f"@save_args ?")
        torch.save(self.serialize(), path)

    @classmethod
    def load_from(cls, loadable_state: Union[str, IO]) -> nn.Module:
        """
        Load and build a model/module form a file.

        Parameters
        ----------
        loadable_state : file_like or dict
            Module state, or path to model file, or opened file object.

        Returns
        -------
        module : nn.Module
            An instantiated [`nn.Module`][torch.nn.Module].
        """
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        hint = cls if cls is not LoadableMixin else None
        return cls._nested_unserialize(loadable_state, klass=hint)
