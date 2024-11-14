__all__ = [
    "LoadableModule",
    "LoadableSequential",
    "LoadableModuleList",
    "LoadableModuleDict",
]
# stdlib

# externals
from torch import nn

# internals
from cassetta.core.typing import Iterable, Optional, Mapping, Tuple, Union, IO
from cassetta.io.loadable import (
    LoadableMixin, _validate_loadable, _validate_loadable_all
)


class LoadableModule(LoadableMixin, nn.Module):
    """A Loadable variant of [`nn.Module`][torch.nn.Module]"""

    @classmethod
    def load(cls, loadable_state: Union[str, IO]) -> "LoadableModule":
        """
        Load and build a model/module from a file.

        Parameters
        ----------
        loadable_state : file_like or dict
            Object state, or path to model file, or opened file object.

        Returns
        -------
        module : nn.Module
            An instantiated [`nn.Module`][torch.nn.Module].
        """
        # We only overload to specialize the docstring
        return super().load(loadable_state)


class LoadableSequential(LoadableMixin, nn.Sequential, save_args=False):
    """A Loadable variant of [`nn.Sequential`][torch.nn.Sequential]"""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        _validate_loadable_all(self)

    def append(self, module: nn.Module) -> "LoadableSequential":
        _validate_loadable(module)
        super().append(module)
        return self

    def extend(self, modules: Iterable[nn.Module]) -> "LoadableSequential":
        modules = list(modules)
        _validate_loadable_all(modules)
        super().extend(modules)
        return self

    def insert(self, index: int, module: nn.Module) -> "LoadableSequential":
        _validate_loadable(module)
        super().insert(index, module)
        return self

    def serialize(self) -> dict:
        return {
            "cassetta.Loadable": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "args": [module.serialize() for module in self],
            "kwargs": getattr(self, "_kwargs", dict()),
        }


class LoadableModuleList(LoadableMixin, nn.ModuleList, save_args=False):
    """A Loadable variant of [`nn.ModuleList`][torch.nn.ModuleList]"""

    def __init__(self, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__(modules)
        _validate_loadable_all(self.values())

    def append(self, module: nn.Module) -> "LoadableModuleList":
        _validate_loadable(module)
        super().append(module)
        return self

    def extend(self, modules: Iterable[nn.Module]) -> "LoadableModuleList":
        modules = list(modules)
        _validate_loadable_all(modules)
        super().extend(modules)
        return self

    def insert(self, index: int, module: nn.Module) -> "LoadableModuleList":
        _validate_loadable(module)
        super().insert(index, module)
        return self

    def serialize(self) -> dict:
        return {
            "cassetta.Loadable": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "args": [[module.serialize() for module in self]],
            "kwargs": getattr(self, "_kwargs", dict()),
        }


class LoadableModuleDict(LoadableMixin, nn.ModuleDict, save_args=False):
    """A Loadable variant of [`nn.ModuleDict`][torch.nn.ModuleDict]"""

    _ValidInput = Union[
        Mapping[str, nn.Module],
        Iterable[Tuple[str, nn.Module]]
    ]

    def __init__(self, modules: Optional[_ValidInput] = None) -> None:
        super().__init__(modules)
        _validate_loadable_all(self.values())

    def __setitem__(self, key, module) -> None:
        _validate_loadable(module)
        super().__setitem__(key, module)

    def update(self, other) -> None:
        _validate_loadable_all(other.values())
        super().update(other)

    def serialize(self):
        return {
            "cassetta.Loadable": LoadableMixin.__version__,
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "args": [{
                key: module.serialize() for key, module in self.items()
            }],
            "kwargs": getattr(self, "_kwargs", dict()),
        }
