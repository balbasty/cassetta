import torch
from typing import (
    List, Union, Any, Optional, Callable, Sequence, Literal, TypeVar
)


T = TypeVar('T')
OneOrMore = Union[T, Sequence[T]]
Device = Union[str, torch.device]
Bound = Union[
    Literal['constant'],
    Literal['reflect'],
    Literal['replicate'],
    Literal['circular'],
]
