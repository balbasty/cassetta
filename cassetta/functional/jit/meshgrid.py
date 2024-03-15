r"""
# meshgrid

## Eager vs TorchScript

The signature of `torch.meshgrid` differs between eager and torchscript modes.

In eager mode, `meshgrid` takes an unpacked list of tensors as inputs:
```python
gx, gy = torch.meshgrid(coordx, coordy)
```

In torchscript mode, it takes a (packed) list of tensors instead:
```python
gx, gy = torch.meshgrid([coordx, coordy])
```

This makes writing code that works in both eager (by setting the
environment variable `PYTORCH_JIT=0`) and torchscript modes (with the
default `PYTORCH_JIT=1`) complicated. Instead, we define functions with
explicit names (`meshgrid_list_*`) that always take a (packed) list of
tensors as input.

## Backward compatibility

For `torch<1.10`, `torch.meshgrid` worked in `"ij"` mode, meaning that
the first tensor of coordinates was mapped to the first dimension of the
output grid and the second tensor of coordinates was mapped to the
second dimension of the output grid. Starting with `torch=1.10`,
the keyword argument `indexing`, which takes value `"ij"` or `"xy"`, was
introduced. Furthermore, the default behavior of the function when `indexing`
is not used will change from `"ij"` to `"xy"` in the future.

To make any code backward compatible, we define explicit functions
postfixed by either `_ij` or `_xy`.
"""
__all__ = [
    'meshgrid_list_ij',
    'meshgrid_list_xy',
]
import torch
import os
from typing import List


_help_intro = (
r"""Creates grids of coordinates specified by the 1D inputs in `tensors`.

This is helpful when you want to visualize data over some
range of inputs.

Given $N$ 1D tensors $T_0, \dots, T_{N-1}$ as inputs with
corresponding sizes $S_0, \dots, S_{N-1}$, this creates $N$
N-dimensional tensors $G_0, \dots, G_{N-1}$, each with shape
$(S_0, \dots, S_{N-1})$ where the output $G_i$ is constructed
by expanding $T_i$ to the result shape.

!!! note
    0D inputs are treated equivalently to 1D inputs of a
    single element.
""")  # noqa: E122

_help_prm = (
r"""
Parameters
----------
tensors : list[tensor]
    list of scalars or 1 dimensional tensors. Scalars will be
    treated as tensors of size $(1,)$ automatically

Returns
-------
seq : list[tensor]
    list of expanded tensors

""")  # noqa: E122

_help_warnxy = (
"""
!!! warning
    In mode `xy`, the first dimension of the output corresponds to the
    cardinality of the second input and the second dimension of the output
    corresponds to the cardinality of the first input.
""")  # noqa: E122

_help_ij = _help_intro + _help_prm
_help_xy = _help_intro + _help_warnxy + _help_prm


TensorList = List[torch.Tensor]

if not int(os.environ.get('PYTORCH_JIT', '1')):
    # JIT deactivated -> torch.meshgrid takes an unpacked list of tensors

    @torch.jit.script
    def meshgrid_list_ij(tensors: TensorList) -> TensorList:
        return list(torch.meshgrid(*tensors, indexing='ij'))

    @torch.jit.script
    def meshgrid_list_xy(tensors: TensorList) -> TensorList:
        return list(torch.meshgrid(*tensors, indexing='xy'))

else:
    # JIT activated -> torch.meshgrid takes a packed list of tensors

    @torch.jit.script
    def meshgrid_list_ij(tensors: TensorList) -> TensorList:
        return list(torch.meshgrid(tensors, indexing='ij'))

    @torch.jit.script
    def meshgrid_list_xy(tensors: TensorList) -> TensorList:
        return list(torch.meshgrid(tensors, indexing='xy'))


meshgrid_list_ij.__doc__ = _help_ij
meshgrid_list_xy.__doc__ = _help_xy
