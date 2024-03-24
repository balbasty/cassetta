__all__ = [
    'Conv',
    'ConvTransposed',
    'SeparableConv',
    'CrossHairConv',
]
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.lazy import LazyModuleMixin
from bounds import all_bounds, pad, to_enum, BoundType as BoundEnum
from typing import Optional, Union, Literal, Tuple, List, Callable
from cassetta.core.typing import OneOrSeveral, DeviceType, BoundType
from cassetta.core.utils import ensure_tuple


def make_conv(
    ndim: int,
    inp_channels: int,
    out_channels: Optional[int] = None,
    kernel_size: OneOrSeveral[int] = 3,
    stride: OneOrSeveral[int] = 1,
    padding: Union[Literal['same', 'valid'], OneOrSeveral[int]] = 0,
    dilation: OneOrSeveral[int] = 1,
    output_padding: OneOrSeveral[int] = 0,
    groups: int = 1,
    bias: bool = True,
    separable: Union[Literal['crosshair'], bool] = False,
    transpose=False,
    padding_mode: BoundType = 'zeros',
    device: Optional[DeviceType] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Instantiate a convolution layer.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    inp_channels : int
        Number of input channels
    out_channels : int, default=`inp_channels`
        Number of output channels
    kernel_size : [list of] int
        Kernel size
    stride : [list of] int
        Space between output elements
    padding : {'same', 'valid'} or [list of] int
        Amount of padding to apply
    dilation : [list of] int
        Space between kernel elements
    output_padding : [list of] int
        Amount of padding to apply to the ouptput of a transposed
        convolution.
    groups : int
        Number of groups
    bias : int
        Add a learnable bias term
    padding_mode : BoundType
        How to pad the tensor
    device : torch.device
        Weights' device
    dtype : torch.dtype
        Weights' data type

    Returns
    -------
    layer : Conv or ConvTransposed or SeparableConv or CrossHairConv
    """
    opt = dict(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype,
    )
    if transpose:
        klass = ConvTransposed
        if separable:
            raise ValueError('Separable convolutions cannot be transposed')
    else:
        opt.pop('output_padding')
        if separable:
            if isinstance(separable, str):
                if separable == 'crosshair':
                    klass = CrossHairConv
                else:
                    raise ValueError(f'Unknown separable value "{separable}"')
            else:
                klass = SeparableConv
            if ensure_tuple(stride, 3) != (1, 1, 1):
                # TODO: implement strided separable convs
                raise ValueError('Separable convolutions cannot be strided')
        else:
            klass = Conv
    return klass(ndim, inp_channels, out_channels, **opt)


class _Conv(nn.Module):
    """
    Convolution layer.

    We reimplement `nn.Conv{n}d` so that the number of spatial dimensions
    can be parameterized. We also implement additional padding modes.

    !!! warning "Differences with `nn.Conv{n}d`"
        -   One drawback is that input tensors **must** have a batch dimension.

        -   Another difference is that in our convention, input tensors are
            ordered [B, C, W, H, D] (instead of [B, C, D, H, W] in
            `nn.Conv{n}d`). This means that the tensor's spatial dimension
            have the same order as the parameters of `kernel_size`, `stride`,
            `dilation`, `padding`, etc.

        - Finally, this class is not compatible with TorchScript.
    """
    ndim: int
    inp_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: BoundType
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: OneOrSeveral[int] = 3,
        stride: OneOrSeveral[int] = 1,
        padding: Union[Literal['same', 'valid'], OneOrSeveral[int]] = 0,
        dilation: OneOrSeveral[int] = 1,
        output_padding: OneOrSeveral[int] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: BoundType = 'zeros',
        device: Optional[DeviceType] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int, default=`inp_channels`
            Number of output channels
        kernel_size : [list of] int
            Kernel size
        stride : [list of] int
            Space between output elements
        padding : {'same', 'valid'} or [list of] int
            Amount of padding to apply
        dilation : [list of] int
            Space between kernel elements
        output_padding : [list of] int
            Amount of padding to apply to the ouptput of a transposed
            convolution.
        groups : int
            Number of groups
        bias : int
            Add a learnable bias term
        padding_mode : BoundType
            How to pad the tensor
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.ndim = ndim
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.check_parameters()

        if self.transposed:
            self.weight = nn.Parameter(torch.empty(
                (
                    self.inp_channels,
                    self.out_channels // self.groups,
                    *self.kernel_size
                ),
                **factory_kwargs
            ))
        else:
            self.weight = nn.Parameter(torch.empty(
                (
                    self.out_channels,
                    self.inp_channels // self.groups,
                    *self.kernel_size
                ),
                **factory_kwargs
            ))
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.out_channels, **factory_kwargs
            ))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def check_parameters(self):
        self.out_channels = self.out_channels or self.inp_channels
        self.kernel_size = ensure_tuple(self.kernel_size, self.ndim)
        self.stride = ensure_tuple(self.stride, self.ndim)
        self.dilation = ensure_tuple(self.dilation, self.ndim)
        self.output_padding = ensure_tuple(self.output_padding, self.ndim)

        if self.groups <= 0:
            raise ValueError('groups must be a positive integer')
        if self.inp_channels % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if self.out_channels % self.groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        valid_padding_strings = {'same', 'valid'}
        if isinstance(self.padding, str):
            if self.padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {self.padding!r}, should be "
                    f"one of {valid_padding_strings}"
                )
            if self.padding == 'same' and any(s != 1 for s in self.stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        if self.padding_mode not in all_bounds:
            raise ValueError(
                f"padding_mode must be one of {all_bounds}, "
                f"but got padding_mode='{self.padding_mode}'"
            )

        if isinstance(self.padding, str) and self.inp_channels:
            if self.padding == 'same':
                self._padding_lr = [
                    ((d * (k - 1))//2, d * (k - 1) - (d * (k - 1))//2)
                    for d, k in zip(self.dilation, self.kernel_size)
                ]
                self._padding_lr = [q for p in self._padding_lr for q in p]
            else:
                assert self.padding == 'valid'
                self._padding_lr = [0] * (2*self.ndim)
        else:
            self.padding = ensure_tuple(self.padding, self.ndim)
            self._padding_lr = [q for p in self.padding for q in (p, p)]

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as
        # initializing with uniform(-1/sqrt(k), 1/sqrt(k)),
        # where k = weight.size(1) * prod(*kernel_size)
        # For more details see:
        # https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{inp_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv(_Conv):
    # Forward convolution

    transposed: bool = False

    def check_parameters(self):
        super().check_parameters()
        if self.output_padding != (0,) * self.ndim:
            raise ValueError(
                'Cannot use output_padding in non-transposed convolution'
            )
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only "zeros" padding mode is supported for ConvTransposed'
            )

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_size) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_size) tensor
            Convolved tensor
        """
        ndim = len(self.kernel_size)
        padding = self.padding
        conv = getattr(F, f'conv{ndim}d')
        if to_enum(self.padding_mode) != BoundEnum.zeros:
            inp = pad(inp, self._padding_lr, mode=self.padding_mode)
            padding = 0
        return conv(inp, self.weight, self.bias, self.stride,
                    padding, self.dilation, self.groups)


class ConvTransposed(_Conv):
    # Transposed convolution

    transposed: bool = True

    def check_parameters(self):
        if self.padding_mode != 'zeros':
            raise ValueError(
                'Only "zeros" padding mode is supported for ConvTransposed'
            )
        return super().check_parameters()

    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[List[int]],
    ) -> List[int]:
        if output_size is None:
            return tuple(self.output_padding)

        ndim = len(self.kernel_size)
        input_size, output_size = input.shape[-ndim:], output_size[-ndim:]
        if len(output_size) != ndim:
            raise ValueError('Output size is too short')

        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(ndim):
            dim_size = (
                1 +
                (input_size[d] - 1) * self.stride[d] -
                2 * self.padding[d] +
                self.dilation[d] * (self.kernel_size[d] - 1)
            )
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + self.stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(
                    f"requested an output size of {output_size}, but "
                    f"valid sizes range from {min_sizes} to {max_sizes} "
                    f"(for an input of {input_size})")

        res = torch.jit.annotate(List[int], [])
        for d in range(ndim):
            res.append(output_size[d] - min_sizes[d])
        return res

    def forward(self, inp: Tensor, out_size: Optional[List[int]] = None
                ) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_size) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_size) tensor
            Output tensor

        """
        ndim = len(self.kernel_size)
        output_padding = self._output_padding(inp, out_size)
        conv_transpose = getattr(F, f'conv_transpose{ndim}d')
        return conv_transpose(
            inp, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class LazyConv(LazyModuleMixin, Conv):
    """
    A convolution layer whose `ndim` and `inp_channels` are guessed
    lazily at run time.
    """
    cls_to_become = Conv

    def __init__(
        self,
        out_channels: Optional[Union[int, Callable]] = None,
        kernel_size: OneOrSeveral[int] = 3,
        stride: OneOrSeveral[int] = 1,
        padding: Union[Literal['same', 'valid'], OneOrSeveral[int]] = 0,
        dilation: OneOrSeveral[int] = 1,
        output_padding: OneOrSeveral[int] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: BoundType = 'zeros',
        device: Optional[DeviceType] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        out_channels : int or callable, default=`inp_channels`
            Number of output channels.
            If a function, takes the materialized number of input
            channels and return the materialized number of output
            channels.
        kernel_size : [list of] int
            Kernel size
        stride : [list of] int
            Space between output elements
        padding : {'same', 'valid'} or [list of] int
            Amount of padding to apply
        dilation : [list of] int
            Space between kernel elements
        groups : int
            Number of groups
        bias : int
            Add a learnable bias term
        padding_mode : BoundType
            How to pad the tensor
        """
        conv_kwargs = dict(
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, padding_mode=padding_mode
        )
        factory_kwargs = dict(dtype=dtype, device=device)
        # ndim=3 to avoid losing user parameters
        super().__init__(3, 0, 0, bias=False, **conv_kwargs)
        self.out_channels = out_channels
        self.weight = nn.UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.inp_channels != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.ndim = input.ndim - 2
                self.inp_channels = input.shape[1]
                if callable(self.out_channels):
                    self.out_channels = self.out_channels(self.inp_channels)
                elif not self.out_channels:
                    self.out_channels = self.inp_channels
                self.check_parameters()
                self.weight.materialize((
                    self.inp_channels,
                    self.out_channels // self.groups,
                    *self.kernel_size
                ))
                if self.bias is not None:
                    self.bias.materialize((self.out_channels,))
                self.reset_parameters()


class LazyConvTransposed(LazyModuleMixin, Conv):
    """
    A transposed convolution layer whose `ndim` and `inp_channels` are
    guessed lazily at run time.
    """
    cls_to_become = ConvTransposed

    def __init__(
        self,
        out_channels: Optional[Union[int, Callable]] = None,
        kernel_size: OneOrSeveral[int] = 3,
        stride: OneOrSeveral[int] = 1,
        padding: Union[Literal['same', 'valid'], OneOrSeveral[int]] = 0,
        dilation: OneOrSeveral[int] = 1,
        output_padding: OneOrSeveral[int] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: BoundType = 'zeros',
        device: Optional[DeviceType] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        out_channels : int or callable, default=`inp_channels`
            Number of output channels.
            If a function, takes the materialized number of input
            channels and return the materialized number of output
            channels.
        kernel_size : [list of] int
            Kernel size
        stride : [list of] int
            Space between output elements
        padding : {'same', 'valid'} or [list of] int
            Amount of padding to apply
        dilation : [list of] int
            Space between kernel elements
        output_padding : [list of] int
            Amount of padding to apply to the ouptput of a transposed
            convolution.
        groups : int
            Number of groups
        bias : int
            Add a learnable bias term
        padding_mode : BoundType
            How to pad the tensor
        """
        conv_kwargs = dict(
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, output_padding=output_padding, groups=groups,
            padding_mode=padding_mode
        )
        factory_kwargs = dict(dtype=dtype, device=device)
        # ndim=3 to avoid losing user parameters
        super().__init__(3, 0, 0, bias=False, **conv_kwargs)
        self.out_channels = out_channels
        self.weight = nn.UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.inp_channels != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.ndim = input.ndim - 2
                self.inp_channels = input.shape[1]
                if callable(self.out_channels):
                    self.out_channels = self.out_channels(self.inp_channels)
                elif not self.out_channels:
                    self.out_channels = self.inp_channels
                self.check_parameters()
                self.weight.materialize((
                    self.out_channels,
                    self.inp_channels // self.groups,
                    *self.kernel_size
                ))
                if self.bias is not None:
                    self.bias.materialize((self.out_channels,))
                self.reset_parameters()


class SeparableConv(nn.Sequential):
    """
    Separable Convolution.

    Implements a ND convolution (e.g., WxHxD) as a series of 1D
    convolutions (e.g., Wx1x1, 1xHx1, 1x1xD).

    !!! warning "The number of input and output channels will be the same"

    !!! warning "Padding mode is `'same'` by default"
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        bias: bool = True,
        padding: Union[int, Literal['same']] = 'same',
        padding_mode: str = 'zeros',
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
        kernel_size : [list of] int
            Kernel size
        dilation : [list of] int
            Space between kernel elements
        bias : int
            Add a learnable bias term
        padding : {'same', 'valid'} or [list of] int
            Amount of padding to apply
        padding_mode : BoundType
            How to pad the tensor
        """
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
            layers.append(Conv(ndim, inpch, outch, **kwargs))

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

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *spatial)

        Returns
        -------
        out : (B, out_channels, *spatial)
        """
        out = 0
        for layer in self:
            out += layer(inp)
        return out
