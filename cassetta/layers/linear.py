import torch
import math
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Union, Callable
from cassetta.core.typing import DeviceType


class Linear(nn.Module):
    """
    Linear layer.

    We reimplement `nn.Linear` so that the dimension that it operates
    upon is the second (by default) instead of the last. This makes it
    more compatible with vision applications.
    """

    __constants__ = ['inp_channels', 'out_channels']
    inp_channels: int
    out_channels: int
    weight: Tensor

    def __init__(
            self,
            inp_channels: int,
            out_channels: Optional[int],
            bias: bool = True,
            dim: int = 1,
            *,
            device: Optional[DeviceType] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        inp_channels : int
            Number of input channels
        out_channels : int, default=`inp_channels`
            Number of output channels
        bias : bool
            Include a bias term
        dim : int
            Dimension along which to operate

        Other Parameters
        ----------------
        device : torch.device
            Weight's device
        dtype : torch.device
            Weight's data type

        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels or inp_channels
        self.dim = dim
        self.weight = nn.Parameter(
            torch.empty((out_channels, inp_channels), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, Cinp, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, Cout, *spatial) tensor
            Output tensor
        """
        inp = inp.movedim(self.dim, -1)
        out = F.linear(input, self.weight, self.bias)
        out = out.movedim(-1, self.dim)
        return out

    def extra_repr(self) -> str:
        return (
            f'inp_channels={self.inp_channels}, '
            f'out_channels={self.out_channels}, '
            f'bias={self.bias is not None}'
        )


class LazyLinear(nn.LazyModuleMixin, Linear):
    """
    A linear layer whose weights get allocated lazily, on first call.

    This allows the number of input channels to be automatically
    determined at run time.

    We reimplement `nn.LazyLinear` so that the dimension that it operates
    upon is the second (by default) instead of the last. This makes it
    more compatible with vision applications.

    We also allow the number of output channels to be set lazily.
    """

    cls_to_become = Linear              # type: ignore[assignment]
    weight: nn.UninitializedParameter
    bias: nn.UninitializedParameter     # type: ignore[assignment]

    def __init__(
            self,
            out_channels: Optional[Union[int, Callable[[int], int]]],
            bias: bool = True,
            dim: int = 1,
            *,
            device: Optional[DeviceType] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        out_channels : int or callable, default=`inp_channels`
            Number of output channels.
            If a function, takes the number of input channels and
            returns the number of output channels.
        bias : bool
            Include a bias term
        dim : int
            Dimension along which to operate

        Other Parameters
        ----------------
        device : torch.device
            Weight's device
        dtype : torch.device
            Weight's data type

        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, False, dim=dim)
        self.weight = nn.UninitializedParameter(**factory_kwargs)
        self.out_channels = out_channels
        if bias:
            self.bias = nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.inp_channels != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.inp_channels = input.shape[self.dim]
                if callable(self.out_channels):
                    self.out_channels = self.out_channels(self.inp_channels)
                self.weight.materialize((self.out_channels, self.inp_channels))
                if self.bias is not None:
                    self.bias.materialize((self.out_channels,))
                self.reset_parameters()
