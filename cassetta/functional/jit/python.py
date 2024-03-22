"""TorchScript compatible functions that act on Python bultins."""
__all__ = [
    'pad_list_int',
    'pad_list_float',
    'pad_list_str',
    'any_list_bool',
    'all_list_bool',
    'prod_list_int',
    'sum_list_int',
    'reverse_list_int',
    'cumprod_list_int',
]
import torch
from typing import List


@torch.jit.script
def pad_list_int(x: List[int], length: int) -> List[int]:
    """
    Pad/crop a list of int until it reaches a target length.

    !!! note
        If padding, the last element gets replicated.

    Parameters
    ----------
    x : list[int]
        List of int
    length : int
        Target length

    Returns
    -------
    x : (length,) list[int]
        List of length `length`.

    """
    if len(x) < length:
        x = x + x[-1:] * (length - len(x))
    if len(x) > length:
        x = x[:length]
    return x


@torch.jit.script
def pad_list_float(x: List[float], dim: int) -> List[float]:
    """
    See [`pad_list_int`][cassetta.functional.jit.pad_list_int].
    """
    if len(x) < dim:
        x = x + x[-1:] * (dim - len(x))
    if len(x) > dim:
        x = x[:dim]
    return x


@torch.jit.script
def pad_list_str(x: List[str], dim: int) -> List[str]:
    """
    See [`pad_list_int`][cassetta.functional.jit.pad_list_int].
    """
    if len(x) < dim:
        x = x + x[-1:] * (dim - len(x))
    if len(x) > dim:
        x = x[:dim]
    return x


_pad_list_help = (
r"""Pad/crop a list of {dtype} until it reaches a target length.

!!! note
    If padding, the last element gets replicated.

Parameters
----------
x : list[{dtype}]
    List of {dtype}
length : int
    Target length

Returns
-------
x : (length,) list[{dtype}]
    List of length `length`.

""")  # noqa: E122
pad_list_int.__doc__ = _pad_list_help.format(dtype="int")
pad_list_float.__doc__ = _pad_list_help.format(dtype="float")
pad_list_str.__doc__ = _pad_list_help.format(dtype="str")


@torch.jit.script
def any_list_bool(x: List[bool]) -> bool:
    """TorchScript equivalent to `any(x)`"""
    for elem in x:
        if elem:
            return True
    return False


@torch.jit.script
def all_list_bool(x: List[bool]) -> bool:
    """TorchScript equivalent to `all(x)`"""
    for elem in x:
        if not elem:
            return False
    return True


@torch.jit.script
def prod_list_int(x: List[int]) -> int:
    """Compute the product of elements in the list"""
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


@torch.jit.script
def sum_list_int(x: List[int]) -> int:
    """Compute the sum of elements in the list. Equivalent to `sum(x)`."""
    if len(x) == 0:
        return 1
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 + x1
    return x0


@torch.jit.script
def reverse_list_int(x: List[int]) -> List[int]:
    """TorchScript equivalent to `x[::-1]`"""
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@torch.jit.script
def cumprod_list_int(x: List[int], reverse: bool = False,
                     exclusive: bool = False) -> List[int]:
    """Cumulative product of elements in the list

    Parameters
    ----------
    x : list[int]
        List of integers
    reverse : bool
        Cumulative product from right to left.
        Else, cumulative product from left to right (default).
    exclusive : bool
        Start series from 1.
        Else start series from first element (default).

    Returns
    -------
    y : list[int]
        Cumulative product

    """
    if len(x) == 0:
        lx: List[int] = []
        return lx
    if reverse:
        x = reverse_list_int(x)

    x0 = 1 if exclusive else x[0]
    lx = [x0]
    all_x = x[:-1] if exclusive else x[1:]
    for x1 in all_x:
        x0 = x0 * x1
        lx.append(x0)
    if reverse:
        lx = reverse_list_int(lx)
    return lx
