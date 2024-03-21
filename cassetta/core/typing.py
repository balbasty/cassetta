__all__ = [
    'OneOrSeveral',
    'DeviceType',
    'BoundType',
    'InterpolationType',
    'ActivationType',
    'NormType',
    'DropoutType',
]
import torch
import numpy as np
from torch.nn import Module
from bounds.types import BoundLike as BoundType
from typing import (
    Union, Optional, Sequence, Literal, Type, TypeVar
)

T = TypeVar('T')
OneOrSeveral = Union[T, Sequence[T]]
DeviceType = Union[str, torch.device]
DataType = Union[str, torch.dtype, type, np.dtype]
ActivationType = Optional[Union[
    str, Module, Type[Module],
]]
NormType = Optional[Union[
    Literal['batch', 'instance', 'layer'], Module, Type[Module],
]]
DropoutType = Optional[Union[
    float, Module, Type[Module],
]]
AttentionType = Optional[Union[
    Literal['a', 'x', 'c', 's', 'cs', 'sc', '+'], Module, Type[Module],
]]
SqzExType = Optional[Union[
    Literal['s', 'c', 'sc'], Module, Type[Module],
]]
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
