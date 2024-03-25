__all__ = [
    'ConvBlock',
    'ConvGroup',
    'DownConvGroup',
    'UpConvGroup',
]
import torch
from torch import nn
from torch import Tensor
from typing import Optional, Union, Literal
from functools import partial
from cassetta.core.typing import (
    OneOrSeveral,
    ActivationType,
    NormType,
    DropoutType,
    AttentionType,
)
from .updown import (
    DownConv, DownPool, DownInterpol, UpConv, UpPool, UpInterpol
)
from .simple import ModuleGroup
from .activations import make_activation
from .attention import make_attention
from .conv import make_conv
from .norm import make_norm
from .dropout import make_dropout


class ConvBlockBase(nn.Sequential):
    """
    Base class for unstrided convolution blocks that contain any of
    these layers:

    - Norm (n)
    - Conv (c)
    - Dropout (d)
    - Activation (a)
    - Attention (x)
    """

    @property
    def inp_channels(self) -> int:
        return self.conv.inp_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'ncdax',
        optc: Optional[dict] = None,
        opta: Optional[dict] = None,
        optn: Optional[dict] = None,
        optd: Optional[dict] = None,
        optx: Optional[dict] = None,
    ):
        super().__init__()
        self.order = self.fix_order(order)
        out_channels = out_channels or inp_channels
        norm_channels = (
            inp_channels if order.index('n') < order.index('c') else
            out_channels
        )
        attention_channels = (
            inp_channels if order.index('x') < order.index('c') else
            out_channels
        )

        conv = make_conv(ndim, inp_channels, out_channels, **(optc or {}))
        norm = make_norm(norm, norm_channels, **(optn or {}))
        dropout = make_dropout(dropout, **(optd or {}))
        activation = make_activation(activation, **(opta or {}))
        attention = make_attention(attention, channels=attention_channels,
                                   ndim=ndim, **(optx or {}))

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
            elif o == 'x':
                self.attention = attention

    @staticmethod
    def fix_order(order: str) -> str:
        order = order.lower()
        if 'n' not in order:
            order = order + 'n'
        if 'c' not in order:
            order = order + 'c'
        if 'd' not in order:
            order = order + 'd'
        if 'a' not in order:
            order = order + 'a'
        if 'x' not in order:
            order = order + 'x'
        return order


class ConvBlock(ConvBlockBase):
    """
    A single convolution, in a Norm + Conv + Dropout + Activation + Attention
    group.

    !!! tip "Diagram"
        === "`order='ncdax'`"
            ```mermaid
            flowchart LR
                subgraph ConvBlock
                    n("Norm"):::w-->
                    on["C<sub>inp</sub>"]    ---c("Conv"):::w-->
                    oc["C<sub>out</sub>"]    ---d("Dropout"):::d-->
                    od["C<sub>out</sub>"]    ---a("Activation"):::d-->
                    oa["C<sub>out</sub>"]    --- x("Attention"):::w
                end
                i["C<sub>inp</sub>"]:::i --- n
                x --> ox["C<sub>out</sub>"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "`order='cndax'`"
            ```mermaid
            flowchart LR
                subgraph ConvBlock
                    c("Conv"):::w-->
                    oc["C<sub>out</sub>"]    ---n("Norm"):::w-->
                    on["C<sub>out</sub>"]    ---d("Dropout"):::d-->
                    od["C<sub>out</sub>"]    ---a("Activation"):::d-->
                    oa["C<sub>out</sub>"]    --- x("Attention"):::w
                end
                i["C<sub>inp</sub>"]:::i --- c
                x --> ox["C<sub>out</sub>"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```

    !!! warning "Padding is always `'same'`"

    !!! tip "Ordering"
        The order of the Norm/Conv/Dropout/Activation layers can be chosen
        with the argument `order`. For example:

        - `order='ncdax'`: Norm -> Conv -> Dropout -> Activation -> Attention
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
        dropout: DropoutType = None,
        attention: AttentionType = None,
        compression: int = 16,
        order: str = 'ncdax',
        separable: Union[bool, Literal['crosshair']] = False,
        optc: Optional[dict] = None,
        optn: Optional[dict] = None,
        optd: Optional[dict] = None,
        opta: Optional[dict] = None,
        optx: Optional[dict] = None,
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
        attention : bool or {'sqzex', 'bcam', 'mha'}
            Attention layer
        compression : int
            Compression ratio of the attention layer
        order : str
            Modules order (permutation of 'ncdax')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution

        Other Parameters
        ----------------
        optc : dict
            Other convolution parameters
        optn : dict
            Other nomralization parameters
        optd : dict
            Other dropout parameters
        opta : dict
            Other activation parameters
        optx : dict
            Other attention parameters
        """
        optc = optc or {}
        optn = optn or {}
        optd = optd or {}
        opta = opta or {}
        optx = optx or {}
        optc.update(dict(
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            separable=separable,
            padding='same',
        ))
        optx.update(dict(
            compression=compression
        ))
        super().__init__(
            ndim=ndim,
            inp_channels=inp_channels,
            out_channels=out_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            optc=optc,
            optn=optn,
            optd=optd,
            opta=opta,
            optx=optx,
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
            Output downsampled tensor
        """
        return super().forward(inp)


class ConvGroup(ModuleGroup):
    r"""
    Multiple convolution blocks stacked together

    !!! tip "Diagram"
        === "`residual=False`"
            ```mermaid
            flowchart LR
                subgraph nb_blocks
                    2("ConvBlock 1"):::w  --> 3["C"] ---
                    4("ConvBlock 2"):::w  --> 5["C"] ---
                    6("..."):::n      --> 7["C"] ---
                    8("ConvBlock N"):::w
                end
                1["C [+S]"]:::i --- 2
                8 ---> 9["C"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "`residual=True`"
            ```mermaid
            flowchart LR
                subgraph nb_blocks
                    2("ConvBlock 1"):::w  --> 3["C"] ---
                    4(("+")):::d      --> 5["C"] ---
                    6("ConvBlock 2"):::w  --> 7["C"] ---
                    8(("+")):::d      --> 9["C"] ---
                    10("..."):::n     --> 11["C"] ---
                    12("ConvBlock N"):::w --> 13["C"] ---
                    14(("+")):::d
                end
                1["C [+S]"]:::i --- 2
                1 --- 4
                5 --- 8
                11 --- 14
                14 ---> 15["C"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```

    !!! note "The recurrent variant shares weights across blocks"

    !!! warning "The number of channels is preserved throughout"

    !!! warning "Padding is always `'same'`"
    """

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    def __init__(
        self,
        ndim: int,
        channels: int,
        nb_conv: int = 1,
        recurrent: bool = False,
        residual: bool = False,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        compression: int = 16,
        order: str = 'ncdax',
        separable: Union[bool, Literal['crosshair']] = False,
        skip: int = 0,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        channels : int
            Number of input and output features
        nb_conv : int
            Number of convolution blocks
        recurrent : bool
            Recurrent network: share weights across blocks
        residual : bool
            Use residual connections between blocks
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
        attention : bool or {'sqzex', 'bcam', 'mha'}
            Attention layer
        compression : int
            Compression ratio of the attention layer
        order : str
            Modules order (permutation of 'ncdax')
        separable : bool or {'cross'}
            Use a separable (or cross-hair) convolution
        skip : int
            Number of additional skipped channels in the input tensor.
        """

        OneConv = partial(
            ConvBlock,
            ndim,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            compression=compression,
            order=order,
            separable=separable,
        )

        layers = []
        if skip:
            nb_conv -= 1
            layers = [OneConv(channels + skip)]

        if recurrent:
            layers += [OneConv(channels)] * nb_conv
        else:
            layers += [OneConv(channels) for _ in range(nb_conv)]
        super().__init__(layers, residual=residual, skip=skip)


class DownGroup(nn.Sequential):
    """
    A downsampling step followed by a bunch of layers.

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            i["[C<sub>inp</sub>, W]"]:::i ---d("Down"):::w-->
            1["[C<sub>mid</sub>, W/2]"] ---b("Block"):::w-->
            o["[C<sub>out</sub>, W/2]"]:::o
            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef n fill:none,stroke:none;
        ```
    """
    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    def __init__(self, module_down: nn.Module, module_block: nn.Module):
        """
        Parameters
        ----------
        module_down : Module
            A downsamlping layer, such as
            [`DownConv`][cassetta.layers.DownConv],
            [`DownPool`][cassetta.layers.DownPool], or
            [`DownInterpol`][cassetta.layers.DownInterpol].
        module_block : Module
            A block of layers, that typically preserve the input shape,
            such as [`ConvBlock`][cassetta.layers.ConvBlock] or
            [`ConvGroup`][cassetta.layers.ConvGroup].
        """
        super().__init__(module_down, module_block)

    @property
    def return_indices(self) -> bool:
        return getattr(self[0], 'return_indices', False)

    def forward(self, inp: Tensor) -> OneOrSeveral[Tensor]:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_size) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *out_size) tensor
            Output downsampled tensor
        indices : (B, out_channels, *out_size) tensor[long]
            Indices, if `return_indices=True`.
        """
        down, block = self
        if getattr(down, 'return_indices', False):
            out, indices = down
            out = block(out)
            return out, indices
        else:
            return super().forward(inp)


class DownConvGroup(DownGroup):
    r"""
    A downsampling step followed by a series of convolution blocks

    !!! tip "Diagram"
        === "`residual=False`"
            ```mermaid
            flowchart LR
                subgraph "<code>nb_conv</code>"
                    2("ConvBlock 1"):::w  --> 3["[C<sub>out</sub>, W/2]"] ---
                    4("ConvBlock 2"):::w  --> 5["[C<sub>out</sub>, W/2]"] ---
                    6("..."):::n      --> 7["[C<sub>out</sub>, W/2]"] ---
                    8("ConvBlock N"):::w
                end
                i["[C<sub>inp</sub>, W]"]:::i ---d("Down"):::w-->
                od["[C<sub>out</sub>, W/2]"]  ---2
                8 ---> 9["[C<sub>out</sub>, W/2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "`residual=True`"
            ```mermaid
            flowchart LR
                subgraph "<code>nb_conv</code>"
                    2("ConvBlock 1"):::w  --> 3["[C<sub>out</sub>, W/2]"] ---
                    4(("+")):::d          --> 5["[C<sub>out</sub>, W/2]"] ---
                    6("ConvBlock 2"):::w  --> 7["[C<sub>out</sub>, W/2]"] ---
                    8(("+")):::d          --> 9["[C<sub>out</sub>, W/2]"] ---
                    10("..."):::n         --> 11["[C<sub>out</sub>, W/2]"] ---
                    12("ConvBlock N"):::w --> 13["[C<sub>out</sub>, W/2]"] ---
                    14(("+")):::d
                end
                i["[C<sub>inp</sub>, W]"]:::i ---d("Down"):::w-->
                od["[C<sub>out</sub>, W/2]"]  ---2
                od --- 4
                5 --- 8
                11 --- 14
                14 ---> 15["[C<sub>out</sub>, W/2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
        mode: Literal['conv', 'interpol', 'pool'] = 'interpol',
        nb_conv: int = 1,
        kernel_size: OneOrSeveral[int] = 3,
        dilation: OneOrSeveral[int] = 1,
        recurrent: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        compression: int = 16,
        order: str = 'ncdax',
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
        factor : [list of] int
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
        sqzex : bool or {'s', 'c', 'sc'}
            Squeeze & Excitation layer
        compression : int
            Compression ratio of the Squeeze & Excitation layer
        order : str
            Modules order (permutation of 'ncdax')
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
            factor=factor,
            **down_options,
        )
        conv = ConvGroup(
            ndim=ndim,
            channels=out_channels,
            nb_conv=nb_conv,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            compression=compression,
            order=order,
            separable=separable,
            residual=residual,
            recurrent=recurrent,
        )
        super().__init__(down, conv)


class UpGroup(nn.Sequential):
    """
    An upsampling step followed by a bunch of layers.

    !!! tip "Diagram"
        === "No skip connections"
            ```mermaid
            flowchart LR
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>mid</sub>, W*2]"] ---b("Block"):::w-->
                o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "Connections with `skip=True`"
            ```mermaid
            flowchart LR
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>mid</sub>, W*2]"]
                s["[C<sub>mid</sub>, W*2]"]:::i
                1 & s --- c(("c")):::d --->
                2["[C<sub>mid</sub>*2, W*2]"] ---b("Block"):::w-->
                o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "Connections with `skip=False`"
            ```mermaid
            flowchart LR
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>mid</sub>, W*2]"]
                s["[C<sub>mid</sub>, W*2]"]:::i
                1 & s --- c(("+")):::d --->
                2["[C<sub>mid</sub>, W*2]"] ---b("Block"):::w-->
                o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```

    """

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    def __init__(self, module_up: nn.Module, module_block: nn.Module,
                 skip: bool = False):
        """
        Parameters
        ----------
        module_up : Module
            An upsampling layer, such as
            [`UpConv`][cassetta.layers.UpConv],
            [`UpPool`][cassetta.layers.UpPool], or
            [`UpInterpol`][cassetta.layers.UpInterpol].
        module_block : Module
            A block of layers, that typically preserve the input shape,
            such as [`ConvBlock`][cassetta.layers.ConvBlock] or
            [`ConvGroup`][cassetta.layers.ConvGroup].
        skip : bool
            Whether to concatenate (`skip=True`) or add (`skip=False`)
            eventual skip connections.
        """
        super().__init__(module_up, module_block)
        self.skip = skip

    def forward(
        self,
        inp: Tensor,
        *skips,
        indices: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, inp_channels, *inp_size) tensor
            Input tensor
        *skips : (B, skip_channels, *inp_size) tensor
            Skipped tensors

        Other Parameters
        ----------------
        indices : (B, out_tensor, *inp_size) tensor[long]
            Unpool indices. Only if `module_up` is an
            [`UpPool`][cassetta.layers.UpPool].

        Returns
        -------
        out : (B, out_channels, *out_size) tensor
            Output downsampled tensor
        """
        up, conv = self
        kwargs = dict(indices=indices) if indices is not None else {}
        out = up(inp, **kwargs)
        if skips:
            if not self.skip:
                for skip in skips:
                    out += skip
            else:
                out = torch.cat([out, *skips], dim=1)
        return conv(out)


class UpConvGroup(UpGroup):
    r"""
    A upsampling step followed by a series of convolution blocks,
    potentially with a skip connection

    !!! tip "Diagram"
        === "No skip connections"
            ```mermaid
            flowchart LR
                subgraph "<code>nb_conv</code>"
                    2("ConvBlock 1"):::w  --> 3["[C<sub>out</sub>, W*2]"] ---
                    4("ConvBlock 2"):::w  --> 5["[C<sub>out</sub>, W*2]"] ---
                    6("..."):::n      --> 7["[C<sub>out</sub>, W*2]"] ---
                    8("ConvBlock N"):::w
                end
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>out</sub>, W*2]"] --- 2
                8 --->  o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "Connections with `skip!=0`"
            ```mermaid
            flowchart LR
                subgraph "<code>nb_conv</code>"
                    2("ConvBlock 1"):::w  --> 3["[C<sub>out</sub>, W*2]"] ---
                    4("ConvBlock 2"):::w  --> 5["[C<sub>out</sub>, W*2]"] ---
                    6("..."):::n      --> 7["[C<sub>out</sub>, W*2]"] ---
                    8("ConvBlock N"):::w
                end
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>out</sub>, W*2]"]
                s["[C<sub>out</sub>, W*2]"]:::i
                1 & s --- c(("c")):::d ---> x["[C<sub>out</sub>*2, W*2]"] --- 2
                8 ---> o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "Connections with `skip=0`"
            ```mermaid
            flowchart LR
                subgraph "<code>nb_conv</code>"
                    2("ConvBlock 1"):::w  --> 3["[C<sub>out</sub>, W*2]"] ---
                    4("ConvBlock 2"):::w  --> 5["[C<sub>out</sub>, W*2]"] ---
                    6("..."):::n      --> 7["[C<sub>out</sub>, W*2]"] ---
                    8("ConvBlock N"):::w
                end
                i["[C<sub>inp</sub>, W]"]:::i ---d("Up"):::w-->
                1["[C<sub>out</sub>, W*2]"]
                s["[C<sub>out</sub>, W*2]"]:::i
                1 & s --- c(("+")):::d ---> x["[C<sub>out</sub>, W*2]"] --- 2
                8 ---> o["[C<sub>out</sub>, W*2]"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
    """

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: Optional[int] = None,
        factor: OneOrSeveral[int] = 2,
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
        attention: AttentionType = None,
        compression: int = 16,
        order: str = 'ncdax',
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
        factor : [list of] int
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
            Modules order (permutation of 'ncdax')
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
            factor=factor,
            **up_options,
        )
        conv = ConvGroup(
            ndim=ndim,
            channels=out_channels,
            nb_conv=nb_conv,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            compression=compression,
            order=order,
            separable=separable,
            residual=residual,
            recurrent=recurrent,
            skip=skip,
        )
        super().__init__(up, conv, skip=skip)
