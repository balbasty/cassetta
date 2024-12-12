__all__ = [
    'LinearBlock',
    'LinearGroup',
]
# stdlib
from functools import partial

# exernals
from torch import nn
from torch import Tensor
from typing_extensions import Optional

# internals
from cassetta.core.typing import (
    ActivationType,
    NormType,
    DropoutType,
)

# local
from .simple import ModuleGroup
from .activations import make_activation
from .linear import make_linear
from .norm import make_norm
from .dropout import make_dropout


class LinearBlockBase(nn.Sequential):
    """
    Base class for linear blocks that contain any of these layers:

    - Norm (n)
    - Linear (l)
    - Dropout (d)
    - Activation (a)
    """

    @property
    def inp_channels(self) -> int:
        return self.linear.inp_channels

    @property
    def out_channels(self) -> int:
        return self.linear.out_channels

    def __init__(
        self,
        inp_channels: int,
        out_channels: Optional[int] = None,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        order: str = 'nlda',
        optl: Optional[dict] = None,
        opta: Optional[dict] = None,
        optn: Optional[dict] = None,
        optd: Optional[dict] = None,
    ):
        super().__init__()
        self.order = self.fix_order(order)
        out_channels = out_channels or inp_channels
        norm_channels = (
            inp_channels if order.index('n') < order.index('l') else
            out_channels
        )

        linear = make_linear(inp_channels, out_channels, **(optl or {}))
        norm = make_norm(norm, norm_channels, **(optn or {}))
        dropout = make_dropout(dropout, **(optd or {}))
        activation = make_activation(activation, **(opta or {}))

        # Assign submodules in order
        for o in self.order:
            if o == 'n':
                self.norm = norm
            elif o == 'l':
                self.linear = linear
            elif o == 'd':
                self.dropout = dropout
            elif o == 'a':
                self.activation = activation

    @staticmethod
    def fix_order(order: str) -> str:
        order = order.lower()
        if 'n' not in order:
            order = order + 'n'
        if 'l' not in order:
            order = order + 'l'
        if 'd' not in order:
            order = order + 'd'
        if 'a' not in order:
            order = order + 'a'
        return order


class LinearBlock(LinearBlockBase):
    """
    A single convolution, in a Norm + Linear + Dropout + Activation
    group.

    !!! tip "Diagram"
        === "`order='nlda'`"
            ```mermaid
            flowchart LR
                subgraph ConvBlock
                    n("Norm"):::w-->
                    on["C<sub>inp</sub>"]    ---l("Linear"):::w-->
                    ol["C<sub>out</sub>"]    ---d("Dropout"):::d-->
                    od["C<sub>out</sub>"]    ---a("Activation"):::d
                end
                i["C<sub>inp</sub>"]:::i --- n
                a --> oa["C<sub>out</sub>"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
            ```
        === "`order='lnda'`"
            ```mermaid
            flowchart LR
                subgraph ConvBlock
                    l("Conv"):::w-->
                    ol["C<sub>out</sub>"]    ---n("Norm"):::w-->
                    on["C<sub>out</sub>"]    ---d("Dropout"):::d-->
                    od["C<sub>out</sub>"]    ---a("Activation"):::d-
                end
                i["C<sub>inp</sub>"]:::i --- c
                a --> oa["C<sub>out</sub>"]:::o
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
        inp_channels: int,
        out_channels: int,
        bias: bool = True,
        dim: int = 1,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        order: str = 'nlda',
        optl: Optional[dict] = None,
        optn: Optional[dict] = None,
        optd: Optional[dict] = None,
        opta: Optional[dict] = None,
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
        bias : bool
            Include a bias term
        dim : int
            Dimension along which to operate
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'nlda')

        Other Parameters
        ----------------
        optl : dict
            Other linear parameters
        optn : dict
            Other nomralization parameters
        optd : dict
            Other dropout parameters
        opta : dict
            Other activation parameters
        """
        optl = optl or {}
        optn = optn or {}
        optd = optd or {}
        opta = opta or {}
        optl.update(dict(bias=bias, dim=dim))
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
            optl=optl,
            optn=optn,
            optd=optd,
            opta=opta,
        )

    def forward(self, inp: Tensor) -> Tensor:
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


class LinearGroup(ModuleGroup):
    r"""
    Multiple linear blocks stacked together

    !!! tip "Diagram"
        === "`residual=False`"
            ```mermaid
            flowchart LR
                subgraph nb_blocks
                    2("LinearBlock 1"):::w  --> 3["C"] ---
                    4("LinearBlock 2"):::w  --> 5["C"] ---
                    6("..."):::n            --> 7["C"] ---
                    8("LinearBlock N"):::w
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
                    2("LinearBlock 1"):::w  --> 3["C"] ---
                    4(("+")):::d            --> 5["C"] ---
                    6("LinearBlock 2"):::w  --> 7["C"] ---
                    8(("+")):::d            --> 9["C"] ---
                    10("..."):::n           --> 11["C"] ---
                    12("LinearBlock N"):::w --> 13["C"] ---
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
    """

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    def __init__(
        self,
        channels: int,
        nb_blocks: int = 1,
        recurrent: bool = False,
        residual: bool = False,
        bias: bool = True,
        dim: int = 1,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        order: str = 'nlda',
        skip: int = 0,
    ):
        """
        Parameters
        ----------
        channels : int
            Number of input and output features
        nb_blocks : int
            Number of linear blocks
        recurrent : bool
            Recurrent network: share weights across blocks
        residual : bool
            Use residual connections between blocks
        bias : bool
            Include a bias term
        dim : int
            Dimension along which to operate
        activation : ActivationType
            Activation function
        norm : NormType
            Normalization function ('batch', 'instance', 'layer')
        dropout : DropoutType
            Dropout probability
        order : str
            Modules order (permutation of 'nlda')
        skip : int
            Number of additional skipped channels in the input tensor.
        """

        OneBlock = partial(
            LinearBlock,
            out_channels=channels,
            bias=bias,
            dim=dim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
        )

        layers = []
        if skip:
            nb_blocks -= 1
            layers = [OneBlock(channels + skip)]

        if recurrent:
            layers += [OneBlock(channels)] * nb_blocks
        else:
            layers += [OneBlock(channels) for _ in range(nb_blocks)]
        super().__init__(layers, residual=residual, skip=skip)
