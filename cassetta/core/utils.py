from types import GeneratorType as generator
from typing import List, Any, Optional
from .typing import Device
import torch
from torch import Tensor


def ensure_list(x: Any, length: Optional[int] = None, crop: bool = True,
                **kwargs) -> List:
    """Ensure that an object is a list

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


def make_vector(input: Any, length: Optional[int] = None, crop: bool = True, *,
                dtype: Optional[torch.dtype] = None,
                device: Optional[Device] = None,
                **kwargs) -> Tensor:
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

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
