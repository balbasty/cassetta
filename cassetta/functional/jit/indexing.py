__all__ = [
    'ind2sub',
    'sub2ind',
    'sub2ind_list',
]

import torch
from torch import Tensor
from typing import List
from .math import floor_div_int
from .python import cumprod_list_int


@torch.jit.script
def ind2sub(ind, shape: List[int]):
    """Convert linear indices into sub indices (i, j, k).

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    ind : tensor_like
        Linear indices
    shape : (D,) vector_like
        Size of each dimension.

    Returns
    -------
    subs : (D, ...) tensor
        Sub-indices.
    """
    stride = cumprod_list_int(shape, reverse=True, exclusive=True)
    sub = ind.new_empty([len(shape)] + ind.shape)
    sub.copy_(ind)
    for d in range(len(shape)):
        if d > 0:
            sub[d] = torch.remainder(sub[d], stride[d-1])
        sub[d] = floor_div_int(sub[d], stride[d])
    return sub


@torch.jit.script
def sub2ind(subs, shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D, ...) tensor
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    subs = subs.unbind(0)
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = cumprod_list_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind


@torch.jit.script
def sub2ind_list(subs: List[Tensor], shape: List[int]):
    """Convert sub indices (i, j, k) into linear indices.

    The rightmost dimension is the most rapidly changing one
    -> if shape == [D, H, W], the strides are therefore [H*W, W, 1]

    Parameters
    ----------
    subs : (D,) list[tensor]
        List of sub-indices. The first dimension is the number of dimension.
        Each element should have the same number of elements and shape.
    shape : (D,) list[int]
        Size of each dimension. Its length should be the same as the
        first dimension of ``subs``.

    Returns
    -------
    ind : (...) tensor
        Linear indices
    """
    ind = subs[-1]
    subs = subs[:-1]
    ind = ind.clone()
    stride = cumprod_list_int(shape[1:], reverse=True, exclusive=False)
    for i, s in zip(subs, stride):
        ind += i * s
    return ind
