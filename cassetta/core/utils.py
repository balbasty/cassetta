"""
A set of various utility functions.
"""
__all__ = [
    'ensure_list',
    'ensure_tuple',
    'make_vector',
    'torch_version',
    'to_torch_dtype',
    'import_submodules',
    'refresh_experiment_dir',
    'delete_files_with_pattern',
    'find_checkpoint'
]
import glob
import os
import numbers
import numpy as np
import shutil
import torch
from torch import Tensor
from types import GeneratorType as generator
from typing import List, Tuple, Any, Optional
from importlib import import_module
from .typing import DeviceType


def ensure_list(
    x: Any,
    length: Optional[int] = None,
    crop: bool = True,
    **kwargs
) -> List:
    """
    Ensure that an object is a list

    The output list is of length at least `length`.
    When `crop` is `True`, its length is also at most `length`.
    If needed, the last value is replicated, unless `default` is provided.

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, range, or generator, it is converted to a list.
    Otherwise, it is placed inside a list.
    """
    if not isinstance(x, (list, tuple, range, generator)):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if length and len(x) < length:
        default = [kwargs.get('default', x[-1] if x else None)]
        x += default * (length - len(x))
    if length and crop:
        x = x[:length]
    return x


def ensure_tuple(
    x: Any,
    length: Optional[int] = None,
    crop: bool = True,
    **kwargs
) -> Tuple:
    """
    Ensure that an object is a tuple.

    See [`ensure_list`][cassetta.core.utils.ensure_list].
    """
    return tuple(ensure_list(x, length, crop, **kwargs))


def make_vector(
    input: Any,
    length: Optional[int] = None,
    crop: bool = True,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceType] = None,
    **kwargs
) -> Tensor:
    """
    Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    length : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.

    Keyword Parameters
    ------------------
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if length is None:
        return input
    if input.numel() >= length:
        return input[:length] if crop else input
    default = kwargs.get('default', input[-1] if input.numel() else 0)
    default = input.new_full([length - len(input)], default)
    return torch.cat([input, default])


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    version = ensure_list(version)
    return _compare_versions(current_version, mode, version)


_dtype_python2torch = {
    float: torch.float32,
    complex: torch.complex64,
    int: torch.int64,
    bool: torch.bool,
}
_dtype_numbers2torch = {
    numbers.Number: torch.float32,
    numbers.Rational: torch.float32,
    numbers.Real: torch.float32,
    numbers.Complex: torch.complex64,
    numbers.Integral: torch.int64,
}
_dtype_str2torch = {
    'float': torch.float32,
    'complex': torch.complex64,
}
_dtype_np2torch = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
_dtype_upcast2torch = {
    np.uint16: torch.int32,
    np.uint32: torch.int64,
}
if hasattr(np, 'float16'):
    if hasattr(torch, 'float16'):
        _dtype_np2torch[np.float16] = torch.float16
    else:
        _dtype_upcast2torch[np.float16] = torch.float32
_dtype_trunc2torch = {
    np.uint64: torch.int64
}
for dt in ('uint128', 'uint256', 'int128', 'int256'):
    if hasattr(np, dt):
        _dtype_trunc2torch[dt] = torch.int64
for dt in ('float80', 'float96', 'float128', 'float256'):
    if hasattr(np, dt):
        _dtype_trunc2torch[dt] = torch.float64
for dt in ('complex160', 'complex192', 'complex256', 'complex512'):
    if hasattr(np, dt):
        _dtype_trunc2torch[dt] = torch.complex128


def to_torch_dtype(dtype, upcast=False, trunc=False):
    """
    Transform a python or numpy dtype or dtype name to a torch dtype.

    !!! warning "Python -> PyTorch convention"
        We follow the PyTorch convention and convert `float` to
        `torch.float32`, `int` to `torch.long` and `complex` to
        `torch.complex64`.

    Parameters
    ----------
    dtype : str or type or np.dtype or torch.dtype
        Input data type
    upcast : bool
        Upcast to nearest torch dtype if input dtype cannot be represented
        exactly. Else, raise a `TypeError`.
    trunc : bool
        Trunc to nearest torch dtype if input dtype cannot be represented
        exactly. Else, raise a `TypeError`.

    Returns
    -------
    dtype : torch.dtype
        Torch data type
    """
    if not dtype:
        return None

    # PyTorch data types
    if isinstance(dtype, torch.dtype):
        return dtype

    # Python builtin data types
    if dtype in _dtype_python2torch:
        return _dtype_python2torch[dtype]

    # Python number types
    if dtype in _dtype_numbers2torch:
        return _dtype_numbers2torch[dtype]

    # Strings for which we do not follow Numpy
    if dtype in _dtype_str2torch:
        return _dtype_str2torch[dtype]

    # Numpy data types
    dtype = np.dtype(dtype).type
    if dtype in _dtype_np2torch:
        return _dtype_np2torch[dtype]

    if dtype in _dtype_upcast2torch:
        if not upcast:
            raise TypeError('Cannot represent dtype in torch (upcast needed)')
        return _dtype_upcast2torch[dtype]

    if dtype in _dtype_trunc2torch:
        if not upcast:
            raise TypeError('Cannot represent dtype in torch (trunc needed)')
        return _dtype_trunc2torch[dtype]

    raise TypeError('Unknown type:', dtype)


def import_submodules(submodules, module, all=None, import_into=False):
    """
    Pre-import submodules into parent module, so that we can do
    ```python
    import pck
    x = pck.submodule.subsubmodule.function(3)
    ```
    instead of
    ```python
    import pck.submodule.subsubmodule
    x = pck.submodule.subsubmodule.function(3)
    ```

    Parameters
    ----------
    submodules : list[str]
        Names of submodules to import
    module : str
        Path to parent module: `__name__`.
    all : list[str]
        Reference to the parent module's `__all__`, that then gets populated
    import_into : bool
        Also import all objects from the submodule into the parent module
    """
    parent_name = module
    parent = import_module(parent_name)
    for child_name in submodules:
        child = import_module('.' + child_name, parent_name)
        setattr(parent, child_name, child)
        if all is not None:
            all += [child_name]
        if import_into:
            for child_obj_name in child.__all__:
                setattr(parent, child_obj_name, getattr(child, child_obj_name))
                if all is not None:
                    all += [child_obj_name]


def refresh_experiment_dir(experiment_dir: str) -> None:
    """
    Check if the directory has contents, and if so, delete them recursively.

    Parameters
    ----------
    dir_path : str
        Path to the directory to be checked and cleared.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    """
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(
            f"The directory {experiment_dir} does not exist."
            )

    # Check if directory is not empty
    if os.listdir(experiment_dir):
        # Recursively remove all contents of the directory
        for item in os.listdir(experiment_dir):
            item_path = os.path.join(experiment_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
            else:
                os.remove(item_path)  # Remove file
        print(f"All contents of {experiment_dir} have been deleted.")
    else:
        print(f"The directory {experiment_dir} is already empty.")
    os.mkdir(f'{experiment_dir}/predictions')
    os.mkdir(f'{experiment_dir}/checkpoints')


def delete_files_with_pattern(directory, pattern):
    """
    Deletes all files in the specified directory that match the given pattern.

    Parameters
    ----------
    directory : str
        The path to the directory.
    pattern : str
        The glob pattern to match files.

    Example
    -------
    delete_files_with_pattern('/path/to/directory', '*last*')
    """
    # Construct the full search pattern
    search_pattern = os.path.join(directory, pattern)

    # Retrieve a list of files matching the pattern
    files_to_delete = glob.glob(search_pattern)

    if not files_to_delete:
        pass

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def find_files_with_pattern(directory, pattern):
    """
    Search for files in the specified directory that match the given grep
    pattern.

    Parameters
    ----------
    directory : str
        The path to the directory to search.
    pattern : str
        The grep pattern to search for.

    Returns
    -------
    paths : list
        A list of file paths that match the pattern.
    """

    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    # Check if the path is a directory
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"The path '{directory}' is not a directory.")

    # Construct the search pattern
    search_pattern = os.path.join(directory, pattern)

    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_pattern)

    return matching_files


def find_checkpoint(
    experiment_dir: str,
    checkpoint_type: str = "best"
) -> Optional[str]:
    """
    Find the checkpoint file in the specified experiment directory.

    Parameters
    ----------
    experiment_dir : str
        The path to the directory of the experiment.
    checkpoint_type : str, optional
        The type of checkpoint to find, either "last" or "best".
        Defaults to "best".

    Returns
    -------
    Optional[str]
        The full path to the checkpoint file if found, otherwise None.

    Raises
    ------
    ValueError
        If checkpoint_type is not "last" or "best".
    """
    if checkpoint_type not in {"last", "best"}:
        raise ValueError("checkpoint_type must be 'last' or 'best'")

    # Define the pattern for file searching based on checkpoint type
    pattern = f"{checkpoint_type}-*.pt"
    # Search for files in the folder matching the pattern
    matches = glob.glob(os.path.join(experiment_dir, 'checkpoints', pattern))

    # If a match is found, return the first (there should only be one)
    return matches[0] if matches else None
