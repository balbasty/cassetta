"""
A set of cassetta-specific type hints, plus backward-compatible hints.
"""
__all__ = [
    # cassetta-specific type hints
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
    # type hints aliases (always taken from `typing`)
    'TypeVar',
    'Union',
    'Optional',
    'Literal',
    'Any',
    'Final',
    'ClassVar',
    'Generic',
    'IO',
    'TextIO',
    'BinaryIO',
    'TypedDict',
    'NoReturn',
    # backward compatible type hints (taken from `typing` if needed)
    'Type',
    'List',
    'Tuple',
    'Dict',
    'ChainMap',
    'Counter',
    'OrderedDict',
    'DefaultDict',
    'Deque',
    'Pattern',
    'Match',
    'AbstractSet',
    'ByteString',
    'Collection',
    'Container',
    'ItemsView',
    'KeysView',
    'Mapping',
    'MappingView',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'Sequence',
    'ValuesView',
    'Coroutine',
    'AsyncGenerator',
    'AsyncIterable',
    'AsyncIterator',
    'Awaitable',
    'Iterable',
    'Iterator',
    'Callable',
    'Generator',
    'Reversible',
    'ContextManager',
    'AsyncContextManager',
    'Hashable',
    'Sized',
    'LiteralString',
    'Never',
    'Self',
    'Required',
    'NotRequired',
    'ReadOnly',
    'Annotated',
]
# stdlib
import sys
from typing import (
    TypeVar,
    Union,
    Optional,
    Literal,
    Any,
    NoReturn,
    Final,
    ClassVar,
    Generic,
    IO,
    TextIO,
    BinaryIO,
    TypedDict,
)

# externals
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from bounds.types import BoundLike as BoundType

# backward compatibile imports
T = TypeVar('T')
if sys.version_info >= (3, 9):
    from collections import (
        ChainMap,
        Counter,
        OrderedDict,
        defaultdict as DefaultDict,
        deque as Deque,
    )
    from collections.abc import (
        Collection,
        Container,
        ItemsView,
        KeysView,
        Mapping,
        MappingView,
        MutableMapping,
        MutableSequence,
        MutableSet,
        Sequence,
        ValuesView,
        Coroutine,
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Iterable,
        Iterator,
        Callable,
        Generator,
        Reversible,
        Buffer as ByteString,
        Set as AbstractSet,
    )
    from re import (
        Pattern,
        Match,
    )
    from contextlib import (
        AbstractContextManager as ContextManager,
        AbstractAsyncContextManager as AsyncContextManager,

    )
    from typing import Annotated
    Type = type
    List = list
    Tuple = tuple
    Dict = dict
else:
    from typing import (
        # builtins
        Type,
        List,
        Tuple,
        Dict,
        # collections
        ChainMap,
        Counter,
        OrderedDict,
        DefaultDict,
        Deque,
        # re
        Pattern,
        Match,
        # abc
        AbstractSet,
        ByteString,
        Collection,
        Container,
        ItemsView,
        KeysView,
        Mapping,
        MappingView,
        MutableMapping,
        MutableSequence,
        MutableSet,
        Sequence,
        ValuesView,
        Coroutine,
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Iterable,
        Iterator,
        Callable,
        Generator,
        Reversible,
        # contextlib
        ContextManager,
        AsyncContextManager,
    )
    from typing_extensions import Annotated
if sys.version_info >= (3, 12):
    from collections.abc import Hashable, Sized
else:
    from typing import Hashable, Sized
if sys.version_info >= (3, 11):
    from typing import (
        LiteralString,
        Never,
        Self,
        Required,
        NotRequired,
    )
else:
    from typing_extensions import (
        LiteralString,
        Never, Self,
        Required,
        NotRequired,
    )
if sys.version_info >= (3, 13):
    from typing import ReadOnly
else:
    from typing_extensions import ReadOnly


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
