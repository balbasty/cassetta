__all__ = ['make_dropout', 'ChannelDropout']
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd as ChannelDropoutBase
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

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, C, *size) tensor
            Input tensor

        out : (B, C, *size) tensor
            Output tensor
        """
        ndim = inp.ndim - 2
        dropout = getattr(F, f'dropout{ndim}d')
        return dropout(inp, self.p, self.training, self.inplace)
