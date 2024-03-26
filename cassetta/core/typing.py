"""
A set of cassetta-specific type hints.
"""
__all__ = [
    'OneOrSeveral',
    'DeviceType',
    'DataType',
    'BoundType',
    'InterpolationType',
    'ActivationType',
    'NormType',
    'DropoutType',
    'AttentionType',
    'ModelType',
    'LossType',
    'OptimType',
]
import torch
import numpy as np
from torch.nn import Module
from torch.optim import Optimizer
from bounds.types import BoundLike as BoundType
from typing import (
    Union, Optional, Sequence, Literal, Type, TypeVar
)

T = TypeVar('T')
OneOrSeveral = Union[T, Sequence[T]]
"""
Either a single value of a type, or a sequence of such values.
"""

DeviceType = Union[str, torch.device]
"""
An instantiated [`torch.device`][torch.device], or a string that allows
instantiating one, such as `"cpu"`, `"cuda"` or `"cuda:0"`.
"""

DataType = Union[str, torch.dtype, np.dtype, type]
"""
A [`torch.dtype`][torch.dtype], [`np.dtype`][numpy.dtype], or a string
that represents one such data type.

See [`to_torch_dtype`][cassetta.core.utils.to_torch_dtype] for details.
"""

ActivationType = Optional[Union[
    str, Module, Type[Module],
]]
"""
An instantiated [`nn.Module`][torch.nn.Module],
or a [`nn.Module`][torch.nn.Module] subtype,
or the name of an activation class in [`torch.nn`][torch.nn] or in
[`cassetta.layers.activations`][]

See [`make_activation`][cassetta.layers.make_activation] for details.
"""

NormType = Optional[Union[
    Literal['batch', 'instance', 'layer'], Module, Type[Module],
]]
"""
An instantiated [`nn.Module`][torch.nn.Module],
or a [`nn.Module`][torch.nn.Module] subtype,
or one of:

| String       | Class                                          |
| ------------ | ---------------------------------------------- |
| `"batch"`    | [`BatchNorm`][cassetta.layers.BatchNorm]       |
| `"instance"` | [`InstanceNorm`][cassetta.layers.InstanceNorm] |
| `"layer"`    | [`LayerNorm`][cassetta.layers.LayerNorm]       |

See [`make_norm`][cassetta.layers.make_norm] for details.
"""

DropoutType = Optional[Union[
    float, Module, Type[Module],
]]
"""
An instantiated [`nn.Module`][torch.nn.Module],
or a [`nn.Module`][torch.nn.Module] subtype,
or a dropout probability between 0 and 1.

See [`make_dropout`][cassetta.layers.make_dropout] for details.
"""

AttentionType = Optional[Union[
    Literal['sqzex', 'cbam', 'dp', 'sdp', 'mha'], Module, Type[Module],
]]
"""
An instantiated [`nn.Module`][torch.nn.Module],
or a [`nn.Module`][torch.nn.Module] subtype,
or one of:

| String    | Class                                          |
| --------- | ---------------------------------------------- |
| `"sqzex"` | [`SqzEx`][cassetta.layers.SqzEx]               |
| `"cbam"`  | [`BlockAttention`][cassetta.layers.BlockAttention]                          |
| `"dp"`    | [`DotProductAttention`][cassetta.layers.DotProductAttention]`(scaled=False)`|
| `"sdp"`   | [`DotProductAttention`][cassetta.layers.DotProductAttention]`(scaled=True)` |
| `"mha"`   | [`MultiHeadAttention`][cassetta.layers.MultiHeadAttention]                  |

See [`make_attention`][cassetta.layers.make_attention] for details.
"""  # noqa: E501

InterpolationType = Union[
    int,
    Literal[
        'nearest',
        'linear',
        'quadratic',
        'cubic',
        'fourth',
        'fifth',
        'sixth',
        'seventh',
    ]
]
"""
The degree of B-splines used for interpolation, between 0 and 7, or
one of their string aliases:

<ol start="0">
  <li><code>"nearest"</code></li>
  <li><code>"linear"</code></li>
  <li><code>"quadratic"</code></li>
  <li><code>"cubic"</code></li>
  <li><code>"fourth"</code></li>
  <li><code>"fifth"</code></li>
  <li><code>"sixth"</code></li>
  <li><code>"seventh"</code></li>
</ol>
"""

ModelType = Union[str, Module, Type[Module]]
"""
A model can be:

- the name of a cassetta model, such as `"SegNet"`;
- the fully qualified path to a model, such as
  `"cassetta.models.ElasticRegNet"`, or `"monai.networks.nets.ResNet"`;
- a [`nn.Module`][torch.nn.Module] subclass, such as
  [`SegNet`][cassetta.models.SegNet];
- an already instantiated [`nn.Module`][torch.nn.Module], such as
  [`SegNet(3, 1, 5)`][cassetta.models.SegNet].
"""

LossType = Union[str, Module, Type[Module]]
"""
A loss can be:

- the name of a cassetta loss, such as `"DiceLoss"`;
- the fully qualified path to a model, such as
  `"cassetta.losses.DiceLoss"`, or `"monai.losses.GeneralizedDiceLoss"`;
- a [`nn.Module`][torch.nn.Module] subclass, such as
  [`DiceLoss`][cassetta.losses.DiceLoss];
- an already instantiated [`nn.Module`][torch.nn.Module], such as
  [`DiceLoss()`][cassetta.losses.DiceLoss].
"""

OptimType = Union[str, Type[Optimizer]]
"""
A model can be:

- the name of a torch optimizer, such as `"Adam"`;
- the fully qualified path to a model, such as `"torch.optim.Adam"`;
- a [`Optimizer`][torch.optim.Optimizer] subclass, such as
  [`Adam`][torch.optim.Adam].
"""
