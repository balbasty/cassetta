__all__ = ['make_dropout', 'ChannelDropout']

# externals
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd as ChannelDropoutBase
from typing_extensions import Callable, Optional

# internals
from cassetta.core.typing import DropoutType


def make_dropout(dropout: DropoutType, **kwargs):
    """
    Instantiate a (channel) dropout module.

    Parameters
    ----------
    dropout : DropoutType
        An already instantiated `nn.Module`, or a `nn.Module` subclass,
        or a callable that returns an instantiated `nn.Module`, or the
        dropout probability.
    kwargs : dict
        Additional parameters to pass to the constructor or function.

    Returns
    -------
    dropout : Module
        An instantiated `nn.Module`.
    """
    if not dropout:
        return None

    if isinstance(dropout, nn.Module):
        return dropout

    if isinstance(dropout, float):
        return ChannelDropout(dropout, **kwargs)

    dropout = dropout(**kwargs)

    if not isinstance(dropout, nn.Module):
        raise ValueError('Dropout did not instantiate a Module')
    return dropout


class ChannelDropout(ChannelDropoutBase):
    """
    Randomly zeros out channels at each forward pass.
    """

    def __init__(self, *args, ndim: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ndim = ndim

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (*batch, channel, *size) tensor
            Input tensor

        out : (*batch, channel, *size) tensor
            Output tensor
        """
        ndim = self.ndim
        if ndim is None:
            ndim = inp.ndim - 2
        dropout: Callable[[Tensor], Tensor] = getattr(F, f'dropout{ndim}d')
        batch = inp.shape[:-ndim-1]
        inp = inp.reshape((-1,) + inp.shape[-ndim-1:])
        out = dropout(inp, self.p, self.training, self.inplace)
        out = out.reshape(batch + out.shape[-ndim-1:])
        return out
