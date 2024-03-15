__all__ = [
    'prod_list_tensor',
    'sum_list_tensor',
    'movedim',
]
import torch
from torch import Tensor
from typing import List


@torch.jit.script
def prod_list_tensor(x: List[Tensor]) -> Tensor:
    """Compute the product of tensors in the list."""
    if len(x) == 0:
        empty: List[int] = []
        return torch.ones(empty)
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 * x1
    return x0


@torch.jit.script
def sum_list_tensor(x: List[Tensor]) -> Tensor:
    """Compute the sum of tensors in the list. Equivalent to `sum(x)`."""
    if len(x) == 0:
        empty: List[int] = []
        return torch.ones(empty)
    x0 = x[0]
    for x1 in x[1:]:
        x0 = x0 + x1
    return x0


@torch.jit.script
def movedim(x, source: int, destination: int):
    """Backward compatible `torch.movedim`"""
    dim = x.dim()
    src, dst = source, destination
    src = dim + src if src < 0 else src
    dst = dim + dst if dst < 0 else dst
    permutation = [d for d in range(dim)]
    permutation = permutation[:src] + permutation[src+1:]
    permutation = permutation[:dst] + [src] + permutation[dst:]
    return x.permute(permutation)
