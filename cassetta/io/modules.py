__all__ = [
    "LoadableMixin",
    "LoadableModule",
    "LoadableModuleList",
    "LoadableModuleDict",
    "LoadableSequential",
    "DynamicLoadableMixin",
    "load_module",
    "make_loadable",
    "validate_loadable_module",
    "validate_loadable_modules",
]
import yaml
import torch
import dataclasses
from torch import nn
from pathlib import Path
from typing import Union, IO
from warnings import warn
from inspect import signature
from .utils import import_qualname


def validate_loadable_module(module):
    """
    Validate if a single module is an instance of LoadableMixin.

    Parameters
    ----------
    module : nn.Module
        The module to check.

    Raises
    ------
    TypeError
        If the module is not an instance of LoadableMixin.
    """
    if not isinstance(module, LoadableMixin):
        raise TypeError("Only Loadable modules can be added.")


def validate_loadable_modules(modules):
    """
    Check if all modules in a list are instances of LoadableMixin.

    Parameters
    ----------
    modules : iterable of nn.Module
        The collection of modules to check.

    Raises
    ------
    TypeError
        If any module is not an instance of LoadableMixin.
    """
    for module in modules:
        validate_loadable_module(module)


def load_module(model_state: Union[str, IO]) -> nn.Module:
    """
    Load and build a model/module from a file.

    Parameters
    ----------
    model_state : file_like or dict
        Model state, or path to model file, or opened file object.

    Returns
    -------
    model : nn.Module
        Instantiated model
    """
    return LoadableMixin.load(model_state)


def make_loadable(klass):
    """
    Create a loadable variant of an existing
    [`nn.Module`][torch.nn.Module] subclass.

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
        A [`nn.Module`][torch.nn.Module] subclass

    Returns
    -------
    loadable_klass : type
        A `(LoadableMixin, klass)` subclass.
    """

    class DynamicModule(DynamicLoadableMixin, klass):
        @DynamicLoadableMixin.save_args
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    DynamicModule.__name__ = "Loadable" + klass.__name__
    DynamicModule.__qualname__ = "Loadable" + klass.__name__
    return DynamicModule


class LoadableMixin:
    """
    A mixin to make a 'torch.nn.Module' loadable and savable.

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

    __version__ = "1.0"
    """Current version of the LoadableMixin format"""

    @staticmethod
    def _nested_serialize(obj):
        if isinstance(obj, nn.Module):
            if not isinstance(obj, LoadableMixin):
                raise TypeError(
                    f"Only loadable modules (which inherit from "
                    f"LoadableMixin) can be serialized. Type {type(obj)} "
                    f"does not."
                )
            return obj.serialize()
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(LoadableMixin._nested_serialize, obj))
        if isinstance(obj, dict):
            return type(obj)(
                {
                    key: LoadableMixin._nested_serialize(value)
                    for key, value in obj.items()
                }
            )
        return obj

    @staticmethod
    def _nested_unserialize(obj, *, klass=None):
        if isinstance(obj, dict):
            if "cassetta.LoadableState" in obj:
                state = obj
                if klass is None:
                    try:
                        klass = import_qualname(
                            sdddtate["module"],
                            state["qualname"],
                        )
                    except Exception:
                        raise ImportError(
                            "Could not import type",
                            state["qualname"],
                            "from module",
                            state["module"],
                        )
                if not isinstance(klass, LoadableMixin):
                    klass = make_loadable(klass)
                args = LoadableMixin._nested_unserialize(state["args"])
                kwargs = LoadableMixin._nested_unserialize(state["kwargs"])
                obj = klass(*args, **kwargs)
                obj.load_state_dict(state["state"])
                return obj
            else:
                return type(obj)(
                    {
                        key: LoadableMixin._nested_unserialize(value)
                        for key, value in obj.items()
                    }
                )
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
            if not hasattr(self, "_args"):
                # we only save parameters once, so that in the case where
                # a Loadable module is subclassed, it is the parameters of
                # the leaf class that are saved, not those passed to
                # any of the parents' __init__
                self._args = cls._nested_serialize(args)
                self._kwargs = cls._nested_serialize(kwargs)
            init(self, *args, **kwargs)

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
        klass = type(self)
        if DynamicLoadableMixin in klass.__bases__:
            klass = klass.__bases__[-1]
        return {
            "cassetta.LoadableState": LoadableMixin.__version__,
            "module": klass.__module__,
            "qualname": klass.__qualname__,
            "args": getattr(self, "_args", tuple()),
            "kwargs": getattr(self, "_kwargs", dict()),
            "state": self.state_dict(),
        }

    def save(self, path: Union[str, IO]) -> None:
        """
        Save the model (and everything needed to build it) in a file.

        Parameters
        ----------
        path : file_like
            Path to output module file, or opened file object.
        """
        num_params = len(signature(self.__init__).parameters)
        if not hasattr(self, "_args") and num_params > 1:
            warn(
                f"Object of type {type(self).__name__} does not have "
                f"saved arguments. Did you decorate __init__ with "
                f"@save_args ?"
            )
        torch.save(self.serialize(), path)

    @classmethod
    def load(cls, loadable_state: Union[str, IO]) -> nn.Module:
        """
        Load and build a model/module from a file.

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


class StateMixin:
    """
    A mixin to handle saving and loading the state of an object to/from a
    dictionary or a YAML file.
    """

    def state_dict(self) -> dict:
        """
        Return the state of the object as a dictionary.

        Returns
        -------
        state : dict
            A dictionary representing the current state of the object.
        """
        return dataclasses.asdict(self)

    def load_state_dict(self, state: Union[dict, str, Path]) -> "StateMixin":
        """
        Load the state of the object from a dictionary or a YAML file.

        Parameters
        ----------
        state : dict or str or Path
            The state dictionary or path to the YAML file containing the state.

        Returns
        -------
        self : StateMixin
            The instance with updated state.
        """
        if isinstance(state, (str, Path)):
            with open(state, "r") as f:
                state = yaml.load(f, Loader=yaml.Loader)  # Use full loader
        for key, value in state.items():
            setattr(self, key, value)
        return self

    @classmethod
    def from_state_dict(cls, state: Union[dict, str, Path]) -> "StateMixin":
        """
        Create a new instance of the class from a state dictionary or YAML
        file.

        Parameters
        ----------
        state : dict or str or Path
            The state dictionary or path to the YAML file containing the state.

        Returns
        -------
        instance : StateMixin
            A new instance of the class initialized with the given state.
        """
        if isinstance(state, (str, Path)):
            with open(state, "r") as f:
                state = yaml.safe_load(f)
        return cls(**state)

    def save_state_dict(self, path: Union[str, Path]) -> None:
        """
        Save the current state of the object to a YAML file.

        Parameters
        ----------
        path : str or Path
            The path of the YAML file to be saved.
        """
        state = self.state_dict()
        with open(path, "w") as f:
            yaml.dump(state, f)

    def serialize(self):
        """
        Serialize the object.
        """
        return self.state_dict()


class DynamicLoadableMixin(LoadableMixin):
    """
    A mixin for non-static types generated by
    [`make_loadable`](cassetta.io.modules.make_loadable)
    """

    pass


class LoadableModule(LoadableMixin, nn.Module):
    """A Loadable variant of [`nn.Module`][torch.nn.Module]"""

    @LoadableMixin.save_args
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LoadableSequential(LoadableMixin, nn.Sequential):
    """A Loadable variant of nn.Sequential"""

    def __init__(self, *args):
        super().__init__(*args)
        validate_loadable_modules(self)

    def append(self, module):
        validate_loadable_module(module)
        super().append(module)

    def extend(self, modules):
        validate_loadable_modules(modules)
        super().extend(modules)

    def insert(self, index, module):
        validate_loadable_module(module)
        super().insert(index, module)

    def serialize(self):
        return {
            "cassetta.LoadableState": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "modules": [module.serialize() for module in self],
            "args": getattr(self, "_args", tuple()),
            "kwargs": getattr(self, "_kwargs", dict()),
            "state": self.state_dict(),
        }

    @classmethod
    def load(cls, loadable_state):
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        modules = []
        for module_state in loadable_state["modules"]:
            modules.append(LoadableMixin.load(module_state))
        instance = cls(*modules)
        return instance


# Example:
# class LoadableLinear(LoadableMixin, nn.Linear):
#     @LoadableMixin.save_args
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__(in_features, out_features, bias)
# model = LoadableSequential(module1, make_loadable(nn.ReLU)())
# model.append(module2)
# Saving:
# model.save('loadable_sequential.pth')
# Loading:
# loaded_model = LoadableSequential.load('loadable_sequential.pth')


class LoadableModuleList(LoadableMixin, nn.ModuleList):
    """A Loadable variant of nn.ModuleList"""

    def __init__(self, modules=None):
        super().__init__(modules)
        validate_loadable_modules(self)

    def append(self, module):
        validate_loadable_module(module)
        super().append(module)

    def extend(self, modules):
        validate_loadable_modules(modules)
        super().extend(modules)

    def insert(self, index, module):
        validate_loadable_module(module)
        super().insert(index, module)

    def serialize(self):
        return {
            "cassetta.LoadableState": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "modules": [module.serialize() for module in self],
            "args": getattr(self, "_args", tuple()),
            "kwargs": getattr(self, "_kwargs", dict()),
            "state": self.state_dict(),
        }

    @classmethod
    def load(cls, loadable_state):
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        modules = []
        for module_state in loadable_state["modules"]:
            modules.append(LoadableMixin.load(module_state))
        instance = cls(modules)
        return instance


class LoadableModuleDict(LoadableMixin, nn.ModuleDict):
    """A Loadable variant of nn.ModuleDict"""

    def __init__(self, modules=None):
        super().__init__(modules)
        # Ensure all modules are loadable
        for key, module in self.items():
            if not isinstance(module, LoadableMixin):
                raise TypeError(f"Module '{key}' must be Loadable")

    def __setitem__(self, key, module):
        validate_loadable_module(module)
        super().__setitem__(key, module)

    def serialize(self):
        return {
            "cassetta.LoadableState": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "modules": {
                key: module.serialize() for key, module in self.items()
                },
            "args": getattr(self, "_args", tuple()),
            "kwargs": getattr(self, "_kwargs", dict()),
            "state": self.state_dict(),
        }

    @classmethod
    def load(cls, loadable_state):
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        modules = {
            key: LoadableMixin.load(mod_state)
            for key, mod_state in loadable_state["modules"].items()
        }
        instance = cls(modules)
        return instance
