__all__ = [
    'SeparableConv',
    'CrossHairConv',
    'ConvBlock',
    'DownConv',
    'UpConv',
    'DownInterpol',
    'UpInterpol',
    'DownPool',
    'UpPool',
    'ConvGroup',
    'DownConvGroup',
    'UpConvGroup',
]
import inspect
import torch
from torch import nn
from torch import Tensor
from bounds.types import BoundLike as BoundType
from typing import Optional, Union, Callable, Literal, Type, TypeVar, Sequence
from functools import partial
from .misc import DoNothing
from .interpol import Resize
from ..core.utils import ensure_list


T = TypeVar('T')
OneOrSeveral = Union[T, Sequence[T]]
ActivationType = Optional[Union[
    str, Callable[[Tensor], Tensor], nn.Module, Type[nn.Module],
]]
NormType = Optional[Union[
    Literal['batch', 'instance', 'layer'],
    Callable[[Tensor], Tensor], nn.Module, Type[nn.Module],
]]
DropoutType = Optional[Union[
    float, Callable[[Tensor], Tensor], nn.Module, Type[nn.Module],
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


class SeparableConv(nn.Sequential):
    """
    Separable Convolution.

    Implements a ND convolution (e.g., WxHxD) as a series of 1D
    convolutions (e.g., Wx1x1, 1xHx1, 1x1xD).

    !!! warning "The number of input and output channels will be the same"
    """

    def __init__(self, ndim, inp_channels, out_channels=None, kernel_size=3,
                 dilation=1, bias=True, padding='same', padding_mode='zeros'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=`inp_channels`
            Number of output channels
        kernel_size : [sequence of] int
        dilation : [sequence of] int
        bias : bool
        padding : int or {'same'}
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}
        """
        klass = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        mid_channels = max(inp_channels, out_channels)

        layers = []
        for dim, (K, D) in enumerate(zip(kernel_size, dilation)):
            K1 = [1] * ndim
            K1[dim] = K
            inpch = inp_channels if dim == 0 else mid_channels
            outch = out_channels if (dim == ndim-1) else mid_channels
            kwargs = dict(kernel_size=K1, dilation=D,
                          padding=padding, bias=bias and (dim == ndim-1),
                          padding_mode=padding_mode)
            layers.append(klass(inpch, outch, **kwargs))

        super().__init__(*layers)


class CrossHairConv(SeparableConv):
    """
    Separable Cross-Hair Convolution.

    Separable convolution, where the input tensor is convolved with
    a set of 1D convolutions, and all outputs are summed together
    (e.g., Wx1x1 + 1xHx1 + 1x1xD).

    !!! note "Padding must be `'same'`"

    !!! quote "Reference"
        Tetteh, Giles, et al. **"Deepvesselnet: Vessel segmentation,
        centerline prediction, and bifurcation detection in 3-d angiographic
        volumes."** _Frontiers in Neuroscience_ 14 (2020).
        [10.3389/fnins.2020.592352](https://doi.org/10.3389/fnins.2020.592352)

        ??? example "bibtex"
            ```bib
            @article{tetteh2020deepvesselnet,
                title={Deepvesselnet: Vessel segmentation, centerline prediction, and bifurcation detection in 3-d angiographic volumes},
                author={Tetteh, Giles and Efremov, Velizar and Forkert, Nils D and Schneider, Matthias and Kirschke, Jan and Weber, Bruno and Zimmer, Claus and Piraud, Marie and Menze, Bj{\"o}rn H},
                journal={Frontiers in Neuroscience},
                volume={14},
                pages={592352},
                year={2020},
                publisher={Frontiers}
            }
            ```
    """  # noqa: E501

    def forward(self, x):
        y = 0
        for layer in self:
            y += layer(x)
        return y


class _ConvBlockBase(nn.Sequential):
    """
    Base class for convolution blocks (i.e. Norm+Conv+Dropout+Activation),
    with or without strides/transpose.
    """

    def __init__(self, ndim, inp_channels, out_channels, opt_conv=None,
                 activation='ReLU', norm=None, dropout=False, order='ncda',
                 separable=False):
        super().__init__()
        self.order = self.fix_order(order)
        conv = self.make_conv(ndim, inp_channels, out_channels,
                              opt_conv or {}, separable)
        norm = self.make_norm(norm, ndim, conv, self.order)
        dropout = self.make_dropout(dropout, ndim)
        activation = self.make_activation(activation)

        # Assign submodules in order
        for o in self.order:
            if o == 'n':
                self.norm = norm
            elif o == 'c':
                self.conv = conv
            elif o == 'd':
                self.dropout = dropout
            elif o == 'a':
                self.activation = activation

    @staticmethod
    def fix_order(order):
        order = order.lower()
        if 'n' not in order:
            order = order + 'n'
        if 'c' not in order:
            order = order + 'c'
        if 'd' not in order:
            order = order + 'd'
        if 'a' not in order:
            order = order + 'a'
        return order

    @staticmethod
    def make_conv(ndim, inp_channels, out_channels, opt_conv, separable):
        transpose = opt_conv.pop('transpose', False)
        if separable:
            if transpose or 'stride' in opt_conv.get('stride', 1) != 1:
                raise ValueError('Separable convolutions cannot be '
                                 'strided or transposed')
            if isinstance(separable, str):
                if separable.lower().startswith('cross'):
                    conv_klass = CrossHairConv
                else:
                    conv_klass = SeparableConv
        else:
            conv_klass = (
                getattr(nn, f'ConvTranspose{ndim}d') if transpose else
                getattr(nn, f'Conv{ndim}d')
            )
        conv = conv_klass(inp_channels, out_channels, **opt_conv)
        return conv

    @staticmethod
    def make_activation(activation):
        #   an activation can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        #   it is useful to accept both these cases as they allow to either:
        #       * have a learnable activation specific to this module
        #       * have a learnable activation shared with other modules
        #       * have a non-learnable activation
        if not activation:
            return None
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)
        return activation

    @staticmethod
    def make_dropout(dropout, ndim):
        dropout = (
            dropout() if inspect.isclass(dropout) else
            dropout if callable(dropout) else
            getattr(nn, f'Dropout{ndim}d')(p=float(dropout)) if dropout else
            None
        )
        return dropout

    @staticmethod
    def make_norm(norm, ndim, conv, order):
        #   a normalization can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        if not norm:
            return None
        if isinstance(norm, bool) and norm:
            norm = 'batch'
        inp_channels = (
            conv.inp_channels if order.index('n') < order.index('c') else
            conv.out_channels
        )
        if isinstance(norm, str):
            if 'instance' in norm.lower():
                norm = getattr(nn, f'InstanceNorm{ndim}d')
            elif 'layer' in norm.lower():
                norm = nn.GroupNorm
            elif 'batch' in norm.lower():
                norm = getattr(nn, f'BatchNorm{ndim}d')
        norm = (
            norm(inp_channels, inp_channels) if norm is nn.GroupNorm else
            norm(inp_channels) if inspect.isclass(norm) else
            norm if callable(norm) else
            None
        )
        return norm


class ConvBlock(_ConvBlockBase):
    """
    A single convolution, in a Norm + Conv + Dropout + Activation group

    !!! warning "Padding is always `'same'`

    !!! tip "Ordering"
        The order of the Norm/Conv/Dropout/Activation layers can be chosen
        with the argument `order`. For example:

        - `order='ncda'`: Norm -> Conv -> Dropout -> Activation
        - `order='andc'`: Activation -> Norm -> Dropout -> Conv

    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: int,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = False,
        order: str = 'ncda',
        separable: Union[bool, Literal['crosshair']] = False,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input features
        out_channels : int
            Number of output features
        kernel_size : [list of] int
            Kernel size
        dilation : [list of] int
            Dilation size
        bias : bool
            Include a bias term
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'ncda')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution
        """
        super().__init__(ndim, inp_channels, out_channels,
                         activation=activation, norm=norm,
                         dropout=dropout, order=order, separable=separable,
                         opt_conv=dict(kernel_size=kernel_size,
                                       dilation=dilation,
                                       bias=bias,
                                       padding='same'))

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *spatial) tensor
            Output downsampled tensor
        """
        return super().forward(inp)


class DownConv(_ConvBlockBase):
    """
    Downsample using a strided convolution.

    !!! warning "This layer includes no activation/norm/dropout"
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Downsampling size
        """
        super().__init__(
            ndim=ndim,
            inp_channels=inp_channels,
            out_channels=out_channels,
            opt_conv=dict(stride=size, kernel_size=size, padding=0),
            activation=None,
            norm=None,
            dropout=None,
            order='c',
            separable=False,
        )

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output downsampled tensor
        """
        return super().forward(inp)


class UpConv(_ConvBlockBase):
    """
    Upsample using a strided convolution.

    !!! warning "This layer includes no activation/norm/dropout"
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Downsampling size
        """
        super().__init__(
            ndim=ndim,
            inp_channels=inp_channels,
            out_channels=out_channels,
            opt_conv=dict(stride=size, kernel_size=size, padding=0,
                          transposed=True),
            activation=None,
            norm=None,
            dropout=None,
            order='c',
            separable=False,
        )

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output upsampled tensor
        """
        return super().forward(inp)


class DownPool(nn.Sequential):
    """
    Downsampling using max-pooling + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Cinp -[maxpool ÷2]-> Cinp -[conv 1x1x1]-> -> Cout
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Kernel size
        return_indices : bool
            Return indices on top of pooled features
        """
        super().__init__()
        MaxPool = getattr(nn, f'MaxPool{ndim}d')
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        layers = [MaxPool(
            kernel_size=size,
            stride=size,
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

    @property
    def return_indices(self):
        return self[0].return_indices

    def forward(self, x):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output downsampled tensor
        indices : (B, out_channels, *out_spatial) tensor[long]
            Argmax of the maxpooling operation.
            Only returned if `return_indices=True`
        """
        if self[0].return_indices:
            pool, conv = self
            x, ind = pool(x)
            x = conv(x)
            return x, ind
        else:
            return super().forward(x)


class UpPool(nn.Sequential):
    r"""
    Downsampling using max-pooling + channel expansion

    !!! warning "This layer includes no activation/norm/dropout"

    ```
    Indices ----------------------------\
                                        |
                                        v
    Cinp -[conv 1x1x1]-> Cout -> -[maxunpool x2]-> Cout
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Kernel size
        """
        super().__init__()
        MaxUnpool = getattr(nn, f'MaxUnpool{ndim}d')
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
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
            kernel_size=size,
            stride=size,
        )]
        super().__init__(*layers)

    def forward(self, inp: Tensor, *, indices: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor
        indices : (B, out_channels, *inp_spatial) tensor[long]
            Indices returned by `DownPool` or `MaxPool{ndim}d`

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output upsampled tensor
        """
        conv, unpool = self
        out = conv(inp)
        out = unpool(out, indices)
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
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Downsampling factor size
        interpolation : [list of] InterpolationType
            Interpolation order.
        bound : [list of] BoundType
            Boundary conditions.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)
        """
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        factor = list(map(lambda x: 1/x, ensure_list(size)))
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
        size: OneOrSeveral[int] = 2,
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
        size : [list of] int
            Downsampling factor size
        interpolation : [list of] InterpolationType
            Interpolation order.
        bound : [list of] BoundType
            Boundary conditions.
        prefilter : bool
            Apply spline pre-filter (= interpolates the input)
        """
        super().__init__()
        Conv = getattr(nn, f'Conv{ndim}d')
        out_channels = out_channels or inp_channels
        layers = []
        if out_channels != inp_channels:
            layers += [Conv(
                inp_channels,
                out_channels,
                kernel_size=1,
            )]
        else:
            layers += [DoNothing()]
        layers += [Resize(
            factor=size,
            interpolation=interpolation,
            bound=bound,
            prefilter=prefilter,
        )]
        super().__init__(*layers)


class ConvGroup(nn.Sequential):
    r"""
    Multiple convolution blocks stacked together

    ```
    Non-residual variant:

    C -[cnda]-> ... -[cnda]-> C
       \__________________/
              nb_conv

    Residual variant:

    .------------.           .------------.
    |            |           |            |
    |            v           |            v
    C -[cnda]-> (+) -> C ... C -[cnda]-> (+) -> C
       \___________________________________/
                      nb_conv
    ```

    !!! tip "The recurrent variant shares weights across blocks"

    !!! warning "The number of channels is preserved throughout"

    !!! warning "Padding is always `'same'`"
    """

    def __init__(
        self,
        ndim: int,
        channels: int,
        nb_conv: int = 1,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        recurrent: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = False,
        order: str = 'ncda',
        separable: Union[bool, Literal['crosshair']] = False,
        skip: int = 0,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        channels : int
            Number of channels
        nb_conv : int
            Number of convolution blocks
        kernel_size : [list of] int
            Kernel size
        dilation : [list of] int
            Dilation size
        recurrent : bool
            Recurrent network: share weights across blocks
        residual : bool
            Use residual connections between blocks
        bias : bool
            Include a bias term
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'ncda')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution
        skip : int
            Number of additional skipped channels in the input tensor.
        """

        OneConv = partial(
            ConvBlock,
            ndim,
            output_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
            separable=separable,
        )

        layers = []
        if skip:
            nb_conv -= 1
            layers = [OneConv(channels + skip)]

        if recurrent:
            layers += [OneConv(channels)]
        else:
            layers += [OneConv(channels) for _ in range(nb_conv)]
        super().__init__(*layers)
        self.nb_conv = nb_conv
        self.residual = residual
        self.recurrent = recurrent
        self.skip = skip

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, channels, *spatial) tensor

        Returns
        -------
        out : (B, channels, *spatial) tensor
        """
        x = inp

        layers = list(self)
        if self.skip:
            first, *layers = layers
            if self.residual:
                identity = x
                x = first(x)
                x += identity[:, :x.shape[1]]
            else:
                x = first(x)

        if self.recurrent:
            layers *= self.nb_conv

        if self.residual:
            for layer in layers:
                identity = x
                x = layer(x)
                x += identity
        else:
            for layer in layers:
                x = layer(x)
        return x


class DownConvGroup(nn.Sequential):
    r"""
    A downsampling step followed by a series of convolution blocks

    ```
    Cinp -[down]-> Cout -[cnda]-> ... -[cnda]-> Cout
                         \__________________/
                                nb_conv
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
        mode: Literal['conv', 'interpol', 'pool'] = 'interpol',
        nb_conv: int = 1,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        recurrent: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = False,
        order: str = 'ncda',
        separable: Union[bool, Literal['crosshair']] = False,
        **down_options,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=`inp_channels`
            Number of output channels
        size : [list of] int
            Downsampling factor
        mode : {'conv', 'interpol', 'pool'}
            Downsampling mode
        nb_conv : int
            Number of convolution blocks
        kernel_size : [list of] int
            Kernel size
        dilation : [list of] int
            Dilation size
        recurrent : bool
            Recurrent network: share weights across blocks
        residual : bool
            Use residual connections between blocks
        bias : bool
            Include a bias term
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'ncda')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution

        Other Parameters
        ----------------
        interpolation : InterpolationType, if `mode="interpol"`
            Spline order
        bound : BoundType, if `mode="interpol"`
            Boundary conditions
        prefilter: bool, if `mode="interpol"`
            Perform proper interpolation by applying spline preflitering
        return_indices: bool, if `mode="pool"`
            Return argmax indices
        """
        mode = mode[0].lower()
        Down = (
            DownConv if mode == 'c' else
            DownPool if mode == 'p' else
            DownInterpol if mode == 'i' else
            None
        )
        down = Down(
            ndim=ndim,
            inp_channels=inp_channels,
            out_channels=out_channels,
            size=size,
            **down_options,
        )
        conv = ConvGroup(
            ndim=ndim,
            channel=out_channels,
            nb_conv=nb_conv,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
            separable=separable,
            residual=residual,
            recurrent=recurrent,
        )
        super().__init__(down, conv)

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output downsampled tensor
        indices : (B, out_channels, *out_spatial) tensor[long]
            Indices, if `return_indices=True`.
        """
        down, conv = self
        if getattr(down, 'return_indices', False):
            out, indices = down
            out = conv(out)
            return out, indices
        else:
            return super().forward(inp)


class UpConvGroup(nn.Sequential):
    r"""
    A upsampling step followed by a series of convolution blocks,
    potentially with a skip connection

    ```
    Cinp -[up]-> Cout -[cnda]-> ... -[cnda]-> Cout
                       \__________________/
                              nb_conv
    ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        size: OneOrSeveral[int] = 2,
        skip: int = 0,
        mode: Literal['conv', 'interpol', 'pool'] = 'interpol',
        nb_conv: int = 1,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        recurrent: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = False,
        order: str = 'ncda',
        separable: Union[bool, Literal['crosshair']] = False,
        **up_options,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=`inp_channels`
            Number of output channels
        size : [list of] int
            Downsampling factor
        skip : int
            Number of channels to concatenate in the skip connection.
            If 0 and skip tensors are provided, will try to add them
            instead of cat.
        mode : {'conv', 'interpol', 'pool'}
            Downsampling mode
        nb_conv : int
            Number of convolution blocks
        kernel_size : [list of] int
            Kernel size
        dilation : [list of] int
            Dilation size
        recurrent : bool
            Recurrent network: share weights across blocks
        residual : bool
            Use residual connections between blocks
        bias : bool
            Include a bias term
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'ncda')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution

        Other Parameters
        ----------------
        interpolation : InterpolationType, if `mode="interpol"`
            Spline order
        bound : BoundType, if `mode="interpol"`
            Boundary conditions
        prefilter: bool, if `mode="interpol"`
            Perform proper interpolation by applying spline preflitering
        """
        mode = mode[0].lower()
        Up = (
            UpConv if mode == 'c' else
            UpPool if mode in ('p', 'u') else
            UpInterpol if mode == 'i' else
            None
        )
        up = Up(
            ndim=ndim,
            inp_channels=inp_channels,
            out_channels=out_channels,
            size=size,
            **up_options,
        )
        conv = ConvGroup(
            ndim=ndim,
            channel=out_channels,
            nb_conv=nb_conv,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
            separable=separable,
            residual=residual,
            recurrent=recurrent,
            skip=skip,
        )
        super().__init__(up, conv)
        self.skip = skip

    def forward(self, inp, *skips, indices=None):
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_spatial) tensor
            Input tensor
        *skips : (B, skip_channels, *inp_spatial) tensor
            Skipped tensors
        indices : (B, out_tensor, *inp_spatial) tensor[long]
            Unpool indices. Only id `mode='pool'`.

        Returns
        -------
        out : (B, out_channels, *out_spatial) tensor
            Output downsampled tensor
        """
        up, conv = self
        kwargs = dict(indices=indices) if indices is not None else {}
        out = up(inp, **kwargs)
        if skips:
            if self.skip == 0:
                for skip in skips:
                    out += skip
            else:
                out = torch.cat([out, *skips], dim=1)
        return conv(out)
