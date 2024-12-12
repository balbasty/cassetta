__all__ = [
    'DownConv',
    'UpConv',
    'DownInterpol',
    'UpInterpol',
    'DownPool',
    'UpPool',
]
# externals
from torch import nn
from torch import Tensor
from typing_extensions import Optional, Tuple, Union

# internals
from cassetta.core.utils import ensure_list
from cassetta.core.typing import (
    OneOrSeveral, InterpolationType, BoundType
)

# local
from .simple import DoNothing
from .interpol import Resize


class DownConv(nn.Module):
    """
    Downsample using a strided convolution.

    !!! warning "This layer includes no activation/norm/dropout"
    """

    @property
    def inp_channels(self) -> int:
        return self.conv.inp_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    @property
    def ndim(self) -> int:
        return len(self.conv.kernel_size)

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, ddefault=`inp_channels`
            Number of output channels
        factor : [list of] int
            Downsampling factor
        """
        out_channels = out_channels or inp_channels
        factor = ensure_list(factor, ndim)
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        self.conv = Conv(
            in_channels=inp_channels,
            out_channels=out_channels,
            stride=factor,
            kernel_size=factor,
            padding=0,
        )

    def __str__(self) -> str:
        return self.conv.__str__()

    def __repr__(self) -> str:
        return self.conv.__repr__()

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (*batch, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (*batch, out_channels, *out_spatial) tensor
            Output downsampled tensor
        """
        batch = inp.shape[:-self.ndim-1]
        inp = inp.reshape((-1,) + inp.shape[-self.ndim-1:])
        out = self.conv(inp)
        out = out.reshape(batch + out.shape[-self.ndim-1:])
        return out


class UpConv(nn.Module):
    """
    Upsample using a strided convolution.

    !!! warning "This layer includes no activation/norm/dropout"
    """

    @property
    def inp_channels(self) -> int:
        return self.conv.inp_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    @property
    def ndim(self) -> int:
        return len(self.conv.kernel_size)

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, ddefault=`inp_channels`
            Number of output channels
        factor : [list of] int
            Downsampling factor
        """
        out_channels = out_channels or inp_channels
        factor = ensure_list(factor, ndim)
        super().__init__()
        Conv = getattr(nn, f'ConvTranspose{ndim}d')
        self.conv = Conv(
            in_channels=inp_channels,
            out_channels=out_channels,
            stride=factor,
            kernel_size=factor,
            padding=0,
        )

    def __str__(self) -> str:
        return self.conv.__str__()

    def __repr__(self) -> str:
        return self.conv.__repr__()

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (*batch, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (*batch, out_channels, *out_spatial) tensor
            Output downsampled tensor
        """
        batch = inp.shape[:-self.ndim-1]
        inp = inp.reshape((-1,) + inp.shape[-self.ndim-1:])
        out = self.conv(inp)
        out = out.reshape(batch + out.shape[-self.ndim-1:])
        return out


class DownPool(nn.Sequential):
    """
    Downsampling using max-pooling + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Cinp -[maxpool ÷2]-> Cinp -[conv 1x1x1]-> -> Cout
    ```
    """

    @property
    def ndim(self) -> int:
        return len(self[0].kernel_size)

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
        return_indices=False,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        factor : [list of] int
            Downsampling factor
        return_indices : bool
            Return indices on top of pooled features
        """
        super().__init__()
        MaxPool = getattr(nn, f'MaxPool{ndim}d')
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        factor = ensure_list(factor, ndim)
        layers = [MaxPool(
            kernel_size=factor,
            stride=factor,
            return_indices=return_indices,
        )]
        if out_channels != inp_channels:
            layers += [Conv(
                inp_channels,
                out_channels,
                kernel_size=1,
            )]
        else:
            layers += [DoNothing()]
        super().__init__(*layers)
        self.inp_channels = inp_channels
        self.out_channels = out_channels

    @property
    def return_indices(self):
        return self[0].return_indices

    def forward(self, inp: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        inp : (*batch, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (*batch, out_channels, *out_spatial) tensor
            Output downsampled tensor
        indices : (*batch, out_channels, *out_spatial) tensor[long]
            Argmax of the maxpooling operation.
            Only returned if `return_indices=True`
        """
        batch = inp.shape[:-self.ndim-1]
        inp = inp.reshape((-1,) + inp.shape[-self.ndim-1:])
        if self[0].return_indices:
            pool, conv = self
            inp, ind = pool(inp)
            out = conv(inp)
            out = out.reshape(batch + out.shape[-self.ndim-1:])
            ind = ind.reshape(batch + ind.shape[-self.ndim-1:])
            return out, ind
        else:
            out = super().forward(inp)
            out = out.reshape(batch + out.shape[-self.ndim-1:])
            return out


class UpPool(nn.Sequential):
    r"""
    Downsampling using max-pooling + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Indices --------------------------- .
                                        |
                                        v
    Cinp -[conv 1x1x1]-> Cout -> -[maxunpool x2]-> Cout
    ```
    """

    @property
    def ndim(self) -> int:
        return len(self[0].kernel_size)

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        factor : [list of] int
            Downsampling factor
        """
        super().__init__()
        MaxUnpool = getattr(nn, f'MaxUnpool{ndim}d')
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        factor = ensure_list(factor, ndim)
        layers = []
        if out_channels != inp_channels:
            layers += [Conv(
                inp_channels,
                out_channels,
                kernel_size=1,
            )]
        else:
            layers += [DoNothing()]
        layers += [MaxUnpool(
            kernel_size=factor,
            stride=factor,
        )]
        super().__init__(*layers)
        self.inp_channels = inp_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor, *, indices: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (*batch, inp_channels, *inp_spatial) tensor
            Input tensor
        indices : (*batch, out_channels, *inp_spatial) tensor[long]
            Indices returned by `DownPool` or `MaxPool{ndim}d`

        Returns
        -------
        out : (*batch, out_channels, *out_spatial) tensor
            Output upsampled tensor
        """
        batch = inp.shape[:-self.ndim-1]
        inp = inp.reshape((-1,) + inp.shape[-self.ndim-1:])
        indices = inp.reshape((-1,) + indices.shape[-self.ndim-1:])
        conv, unpool = self
        out = conv(inp)
        out = unpool(out, indices)
        out = out.reshape(batch + out.shape[-self.ndim-1:])
        return out


class DownInterpol(nn.Sequential):
    """
    Downsampling using spline interpolation + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Cinp -[interpol ÷2]-> Cinp -[conv 1x1x1]-> -> Cout
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
        interpolation: InterpolationType = 'linear',
        bound: BoundType = 'zero',
        prefilter: bool = True,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        factor : [list of] int
            Downsampling factor
        interpolation : [list of] InterpolationType
            Interpolation order.
        bound : [list of] BoundType
            Boundary conditions.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)
        """
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        factor = list(map(lambda x: 1/x, ensure_list(factor, ndim)))
        layers = [Resize(
            factor=factor,
            interpolation=interpolation,
            bound=bound,
            prefilter=prefilter,
        )]
        if out_channels != inp_channels:
            layers += [Conv(
                inp_channels,
                out_channels,
                kernel_size=1,
            )]
        super().__init__(*layers)
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.ndim = ndim


class UpInterpol(nn.Sequential):
    """
    Upsampling using spline interpolation + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Cinp -[conv 1x1x1]-> Cout -> -[interpol x2]-> Cout
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
        interpolation: InterpolationType = 'linear',
        bound: BoundType = 'zero',
        prefilter: bool = True,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        factor : [list of] int
            Downsampling factor
        interpolation : [list of] InterpolationType
            Interpolation order.
        bound : [list of] BoundType
            Boundary conditions.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)
        """
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        layers = []
        if out_channels != inp_channels:
            layers += [Conv(
                inp_channels,
                out_channels,
                kernel_size=1,
            )]
        layers += [Resize(
            factor=ensure_list(factor, ndim),
            interpolation=interpolation,
            bound=bound,
            prefilter=prefilter,
        )]
        super().__init__(*layers)
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.ndim = ndim
