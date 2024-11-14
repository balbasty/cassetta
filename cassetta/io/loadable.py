__all__ = [
    "LoadableMixin",
    "DynamicLoadableMixin",
    "load",
    "make_loadable",
]
# stdlib
import dataclasses
from pathlib import Path
from warnings import warn
from inspect import signature

# externals
import torch
from torch import nn

# internals
from cassetta.io.utils import import_qualname
from cassetta.core.typing import Union, IO, Type


def _default_init(cls) -> None:
    # Default init method that calls parents init.
    #
    #   Let `class Child(Base1, Base2): ...`
    #   If `__init__` was not specifically implemented in `Child`, then
    #   `Child.__init__` returns a reference one of the parents `__init__`.
    #   However, it does not mean that only this specific parent's `__init__`
    #   is called when instantiating a `Child` object. Instead, both parent's
    #   `__init__` are called, sorted by method resolution order (MRO).
    #   This is similar to what happens if we had defined `Child.__init__` as
    #   calling `super().__init__`.
    #
    #   Now, say that we want to decorate the child's __init__, whether
    #   it is overloaded in th echild class _or_ it implictely inherits
    #   both parents `__init__`. If we're in the latter case and simply do
    #       `Child.__init__ = decorate(Child.__init__)`
    #   we end up never caling the other parent's `__init__`! Effectively
    #   we've stripped down the MRO. It's like calling Base1.__init__
    #   instead of super().__init__ (big mistake!)
    #
    #   Instead, we can check whether `__init__` is explicit or implicit
    #   by checking whether it is in `Child.__dict__`. If yes, it has
    #   been explicitly implemented in the `Child` class. Otherwise,
    #   it implicitelly calls `super().__init__`. In the latter case
    #   we must assign a function that explicitly calls `super()`, so
    #   that it can be decorated.

    def __init__(self, *args, **kwargs) -> None:
        super(cls, self).__init__(*args, **kwargs)

    return __init__


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

    def __init_subclass__(cls, /, save_args: bool = True, **kwargs):
        # Implement `save_args` type keyword
        super().__init_subclass__(**kwargs)
        if save_args:
            if "__init__" not in cls.__dict__:
                cls.__init__ = _default_init(cls)
            cls.__init__ = cls._save_args(cls.__init__)

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
        _nested_unserialize = LoadableMixin._nested_unserialize
        if isinstance(obj, dict):
            if "cassetta.Loadable" in obj:
                state = obj
                if klass is None:
                    try:
                        klass = import_qualname(
                            state["module"],
                            state["qualname"],
                        )
                    except Exception:
                        raise ImportError(
                            "Could not import type",
                            state["qualname"],
                            "from module",
                            state["module"],
                        )
                if not issubclass(klass, LoadableMixin):
                    klass = make_loadable(klass)
                args = _nested_unserialize(state.get("args", []))
                kwargs = _nested_unserialize(state.get("kwargs", {}))
                obj = klass(*args, **kwargs)
                # For certain circumstances such as LoadableModuleDict,
                # there will not be a "state" attribute, as we save the
                # modules in args.
                if "state" in state and hasattr(obj, "load_state_dict"):
                    obj.load_state_dict(state["state"])
                return obj
            else:
                return type(obj)(
                    {
                        key: _nested_unserialize(value)
                        for key, value in obj.items()
                    }
                )
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(LoadableMixin._nested_unserialize, obj))
        return obj

    @classmethod
    def _save_args(cls, init):
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
        loadable_state = {
            "cassetta.Loadable": LoadableMixin.__version__,
            "module": klass.__module__,
            "qualname": klass.__qualname__,
            "args": getattr(self, "_args", tuple()),
            "kwargs": getattr(self, "_kwargs", dict()),
        }
        if hasattr(self, "state_dict"):
            loadable_state["state"] = self.state_dict()
        return loadable_state

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
    def load(cls, loadable_state: Union[str, IO]) -> "LoadableMixin":
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


def load(model_state: Union[str, IO]) -> nn.Module:
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


class DynamicLoadableMixin(LoadableMixin):
    """
    A mixin for non-static types generated by
    [`make_loadable`](cassetta.io.modules.make_loadable)
    """

    pass


def make_loadable(klass, save_args: bool = True) -> Type[DynamicLoadableMixin]:
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

    class DynamicModule(DynamicLoadableMixin, klass, save_args=save_args):
        ...

    DynamicModule.__name__ = "Loadable" + klass.__name__
    DynamicModule.__qualname__ = "Loadable" + klass.__name__
    return DynamicModule


class StateMixin:
    """
    A mixin to handle saving and loading the state of an object in multiple
    file formats {`.yaml`, `.json`, `.pt`}.

    This mixin is designed to be used with simple data containers (e.g.,
    classes decorated with `@dataclass`) where the state consists of basic
    attributes that can be serialized into dictionaries.

    Notes
    -----
    - Classes that inherit from `StateMixin` are intended to be data containers
      and must be decorated with `@dataclass` to ensure their fields can be
      serialized and deserialized correctly.
    - User must instantiate correct class (or subclas) before loading
      the state.
    - This mixin does not save the type of the object in the file.

    Supported File Formats
    ----------------------
    - `.yaml`: YAML
    - `.json`: JSON
    - PyTorch's default file format

    Example
    -------
    ```python
    # Make data container subclass
    @dataclass
    class TrainingState(StateMixin):
        epochs : int = 25

    # Build state data container object
    state = TrainingState()

    # Save state data container object
    state.save_state_dict('path/to/state.pth')

    # Load state
    loaded_state = TrainingState()
    loaded_state.load('training_state.pth)
    ```
    """

    def serialize(self) -> dict:
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
        Load the state of the object from a dictionary or file.

        Parameters
        ----------
        state : dict or str or Path
            The state dictionary or path to the file containing the state.

        Returns
        -------
        self : StateMixin
            The instance with updated state.
        """
        if isinstance(state, (str, Path)):
            path = Path(state)
            if path.suffix == ".yaml":
                import yaml
                with open(state, "r") as f:
                    state = yaml.load(f, Loader=yaml.Loader)
            elif path.suffix == ".json":
                import json
                with open(state, "r") as f:
                    state = json.load(f)
            else:
                state = torch.load(state)
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
            import yaml
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
        path = Path(path)
        state = self.serialize()
        if path.suffix == ".yaml":
            import yaml
            with open(path, "w") as f:
                yaml.dump(state, f)
        elif path.suffix == ".json":
            import json
            with open(path, "w") as f:
                json.dump(state, f)
        else:
            torch.save(state, path)
