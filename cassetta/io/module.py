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
from cassetta.core.typing import Iterable, Optional, Mapping, Tuple
from cassetta.io.loadable import LoadableMixin


def _validate_loadable_module(module: nn.Module) -> None:
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
        raise TypeError(
            "Only Loadable modules can be added."
            f" '{module.__class__.__name__}' is not a LoadableMixin."
            )


def _validate_loadable_modules(modules: Iterable[nn.Module]) -> None:
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
        _validate_loadable_module(module)


class LoadableModule(LoadableMixin, nn.Module):
    """A Loadable variant of [`nn.Module`][torch.nn.Module]"""
    ...


class LoadableSequential(LoadableMixin, nn.Sequential, save_args=False):
    """A Loadable variant of [`nn.Sequential`][torch.nn.Sequential]"""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        _validate_loadable_modules(self)

    def append(self, module: nn.Module) -> "LoadableSequential":
        _validate_loadable_module(module)
        super().append(module)
        return self

    def extend(self, modules: Iterable[nn.Module]) -> "LoadableSequential":
        modules = list(modules)
        _validate_loadable_modules(modules)
        super().extend(modules)
        return self

    def insert(self, index: int, module: nn.Module) -> "LoadableSequential":
        _validate_loadable_module(module)
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
        _validate_loadable_modules(self)

    def append(self, module: nn.Module) -> "LoadableModuleList":
        _validate_loadable_module(module)
        super().append(module)
        return self

    def extend(self, modules: Iterable[nn.Module]) -> "LoadableModuleList":
        modules = list(modules)
        _validate_loadable_modules(modules)
        super().extend(modules)
        return self

    def insert(self, index: int, module: nn.Module) -> "LoadableModuleList":
        _validate_loadable_module(module)
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

    _ValidInput = Mapping[str, nn.Module] | Iterable[Tuple[str, nn.Module]]

    def __init__(self, modules: Optional[_ValidInput] = None) -> None:
        super().__init__(modules)
        # Ensure all modules are loadable
        for key, module in self.items():
            if not isinstance(module, LoadableMixin):
                raise TypeError(f"Module '{key}' must be Loadable")

    def __setitem__(self, key, module):
        _validate_loadable_module(module)
        super().__setitem__(key, module)

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
