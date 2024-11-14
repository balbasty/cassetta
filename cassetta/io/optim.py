__all__ = [
    "LoadableOptimizerMixin",
    "LazyOptimizer",
    "LoadableOptimizer",
    "LoadableOptimizerDict",
]

# externals
import torch
from torch.nn import Module
from torch.optim import Optimizer

# internals
from cassetta.core.typing import (
    Union, IO, Iterable, Dict, Optional, Callable, Mapping, Tuple
)
from cassetta.io.loadable import (
    LoadableMixin, DynamicLoadableMixin,
    _validate_loadable, _validate_loadable_all
)


OptimizedParameters = Union[Iterable[Module], Dict[str, Iterable[Module]]]


class LoadableOptimizerMixin(LoadableMixin):
    """
    Mixin specific to optimizer that does not serialize the parameter list.
    """
    _abstract_loadable = True  # Hint that this is not a concrete class

    def __init_subclass__(cls, /, **kwargs):
        # Implement `save_args` type keyword
        super().__init_subclass__(**kwargs)
        if "_abstract_loadable" not in cls.__dict__:
            cls._abstract_loadable = False  # derived classes are concrete

    def serialize(self) -> dict:
        # Serialize as normal
        serialized_state = super().serialize()
        # Gather state dict as standard from pytorch optimizer
        serialized_state["state"] = self.state_dict()
        # Gather args and kwargs (to be manipulated)
        args = serialized_state.get("args", tuple())
        kwargs = serialized_state.get("kwargs", dict())
        # Remove params from args if present
        args = args[1:]
        # Replace args and kwargs
        serialized_state["args"] = args
        serialized_state["kwargs"] = kwargs
        return serialized_state

    @classmethod
    def load(
        cls,
        loadable_state: Union[str, IO],
        params: Optional[OptimizedParameters] = None,
    ) -> "LoadableOptimizer":
        """
        Parameters
        ----------
        loadable_state : file_like or dict
            Optimizer state, or path to model file, or opened file object.
        params : list[Module] | dict[str, list[Module]], optional
            Parameters to pass to the optimizer

        Returns
        -------
        optim : Optimizer | LazyOptimizer
            An instantiated optimizer, or a lazy optimizer if `params=None`.
        """
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        else:
            loadable_state = dict(loadable_state)
        if params is None:
            # Return a lazy object
            return LazyOptimizer(lambda p: cls.load(loadable_state, p))
        else:
            # Prepend params to args
            args = [params] + list(loadable_state.get("args", []))
            loadable_state["args"] = args
            return super().load(loadable_state)


class DynamicLoadableOptimizerMixin(
    LoadableOptimizerMixin, DynamicLoadableMixin
):
    _abstract_loadable = True  # Hint that this is not a concrete class

    def __init_subclass__(cls, /, **kwargs):
        # Implement `save_args` type keyword
        super().__init_subclass__(**kwargs)
        if "_abstract_loadable" not in cls.__dict__:
            cls._abstract_loadable = False  # derived classes are concrete


class LazyOptimizer:
    """A wrapper around an optimizer that awaits its parameters."""
    def __init__(
        self,
        init: Union[Callable[[OptimizedParameters], Optimizer]]
    ):
        """
        Parameters
        ----------
        init : callable
            A function that takes model parameters (iterable or dict)
            and returns an instantiated optimizer.
        """
        self.init = init

    def __call__(self, params: OptimizedParameters) -> Optimizer:
        return self.init(params)


class LoadableOptimizer(LoadableOptimizerMixin, Optimizer):
    """
    A loadable variant of [`optim.Optimizer`][torch.optim.Optimizer]

    This is a loadable mixin for optimizers **without** saving model params.
    """
    pass


class LoadableOptimizerDict(LoadableMixin, dict, save_args=False):
    """A dictionary of optimizers."""

    _ValidInput = Union[
        Mapping[str, Optimizer],
        Iterable[Tuple[str, Optimizer]]
    ]

    def __init__(self, modules: Optional[_ValidInput] = None) -> None:
        super().__init__(modules)
        _validate_loadable_all(self.values())
        _validate_loadable_all(self.values(), (Optimizer, LazyOptimizer))

    def __setitem__(self, key, optim) -> None:
        _validate_loadable(optim, LoadableOptimizerMixin)
        _validate_loadable(optim, (Optimizer, LazyOptimizer))
        super().__setitem__(key, optim)

    def update(self, other) -> None:
        _validate_loadable_all(other.values(), LoadableOptimizerMixin)
        _validate_loadable_all(other.values(), (Optimizer, LazyOptimizer))
        super().update(other)

    def serialize(self):
        return {
            "cassetta.Loadable": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "args": [{
                key: optim.serialize() for key, optim in self.items()
            }],
            "kwargs": getattr(self, "_kwargs", dict()),
        }

    @classmethod
    def load(
        cls,
        loadable_state: Union[str, IO],
        params: Optional[Dict[str, OptimizedParameters]] = None,
    ) -> "LoadableOptimizer":
        """
        Parameters
        ----------
        loadable_state : file_like or dict
            Optimizer state, or path to model file, or opened file object.
        params : dict[str, list[Module] | dict[str, list[Module]]], optional
            Dictionary of parameters to pass to the optimizers.

        Returns
        -------
        optim : LoadableOptimizerDict
            An instantiated optimizer, or a lazy optimizer if `params=None`.
        """
        if not isinstance(loadable_state, dict):
            loadable_state = torch.load(loadable_state)
        else:
            loadable_state = dict(loadable_state)
        if params is None:
            params = {}
        loadable_state["args"] = [{
            key: LoadableOptimizerMixin.load(state, params.get(key, None))
            for key, state in params["args"][0].keys()
        }]
        return super().load(loadable_state)
