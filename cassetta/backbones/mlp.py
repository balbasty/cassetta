__all__ = [
    'MLP',
    'MLPWrapper',
    'SimpleMLP',
]
# stdlib
from torch import nn
from functools import partial

# externals
from typing_extensions import Iterable, List, Optional

# internals
from cassetta.core.utils import ensure_list, argalias
from cassetta.core.typing import (
    ActivationType, NormType, DropoutType, OneOrSeveral
)
from cassetta.layers.linear import Linear
from cassetta.layers.linearblocks import LinearBlock


class MLP(nn.Sequential):
    """A multilayer perceptron backbone.

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            1["`[F0]`"]    ---2("LinearBlock"):::w-->
            3["`[F1]`"]    ---4("LinearBlock"):::w-->
            5["`[F2]`"]
            classDef w fill:papayawhip,stroke:peachpuff;
        ```

    !!! warning "The last block is not a pure linear block"
        It is therefore likely to contain an activation.
    """  # noqa: E501

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    inp_features = inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    out_features = out_channels

    @property
    def all_out_channels(self) -> List[int]:
        return [layer.out_channels for layer in self]

    all_out_features = all_out_channels

    @argalias({"channels": "features"})
    def __init__(
        self,
        channels: Iterable[int] = (128,) * 3,
        bias: bool = True,
        dim: int = 1,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        order: str = 'nlda',
    ) -> None:
        """
        Parameters
        ----------
        channels : list[int]
            Channels at each level. There will be `len(channels) - 1`
            linear blocks in the network.
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
        """
        OneBlock = partial(
            LinearBlock,
            bias=bias,
            dim=dim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
        )
        layers = []
        channels = list(channels)
        for i in range(len(channels)-1):
            layers.append(OneBlock(channels[i], channels[i+1]))
        super().__init__(*layers)


class MLPWrapper(nn.Sequential):
    """
    An MLP backbone sandwiched between two channel adapter blocks.

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            1["`[C0]`"]    ---2("Linear"):::w-->
            3["`[F0]`"]    ---4("MLP"):::w-->
            5["`[F1]`"]    ---6("Linear"):::w-->
            7["`[C1]`"]
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
    """

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    inp_features = inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    out_features = out_channels

    @property
    def all_out_channels(self) -> List[int]:
        return [layer.out_channels for layer in self]

    all_out_features = all_out_channels

    @argalias({"inp_channels": "inp_features", "out_channels": "out_features"})
    def __init__(
        self,
        inp_channels: int,
        out_channels: Optional[int] = None,
        backbone: Optional[nn.Module] = None,
        dim: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        inp_channels : int
            Number of input channels
        out_channels : out
            Number of output channels
        backbone : Module, optional
            The MLP backbone
        dim : int
            Dimension along which to operate
        """
        backbone = backbone or MLP(dim=dim)
        first = Linear(inp_channels, backbone.inp_channels, dim=dim)
        last = Linear(backbone.out_channels, out_channels, dim=dim)
        super().__init__(first, backbone, last)


class SimpleMLP(nn.Sequential):
    """A simple multi-layer perceptron.

    !!! tip "Diagram"
        === "One hidden layer"
        ```mermaid
        flowchart LR
            1["`[C0]`"]    ---2("LinearBlock"):::w-->
            3["`[F]`"]     ---4("Linear"):::w-->
            5["`[C1]`"]
            classDef w fill:papayawhip,stroke:peachpuff;
        === "Multiple hidden layer"
        ```mermaid
        flowchart LR
            1["`[C0]`"]    ---2("LinearBlock"):::w-->
            3["`[F0]`"]    ---4("LinearBlock"):::w-->
            5["`[F1]`"]    ---6("LinearBlock"):::w-->
            7["`[F2]`"]    ---8("Linear"):::w-->
            9["`[C1]`"]
            classDef w fill:papayawhip,stroke:peachpuff;
        ```

    !!! warning "In this class, the final block is a pure linear block"
        It therefore does not contain an activation.
    """

    @property
    def inp_channels(self) -> int:
        return self[0].inp_channels

    inp_features = inp_channels

    @property
    def out_channels(self) -> int:
        return self[-1].out_channels

    out_features = out_channels

    @property
    def all_out_channels(self) -> List[int]:
        return [layer.out_channels for layer in self]

    all_out_features = all_out_channels

    @argalias({
        "inp_channels": "inp_features",
        "out_channels": "out_features",
        "nb_features": "nb_channels",
    })
    def __init__(
        self,
        inp_channels: int,
        out_channels: Optional[int] = None,
        nb_features: OneOrSeveral[int] = 128,
        bias: bool = True,
        dim: int = 1,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        order: str = 'nlda',
    ) -> None:
        """
        Parameters
        ----------
        inp_channels : int
            Number of input channels
        out_channels : out
            Number of output channels
        nb_features : int | list[int]
            Number of hidden channels at each level.
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
        """

        OneBlock = partial(
            LinearBlock,
            bias=bias,
            dim=dim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            order=order,
        )
        LastBlock = partial(
            Linear,
            bias=bias,
            dim=dim,
        )

        layers = []
        nb_features = ensure_list(nb_features)
        nb_features.insert(0, inp_channels)
        for i in range(len(nb_features)-1):
            layers.append(OneBlock(nb_features[i], nb_features[i+1]))
        layers.append(LastBlock(nb_features[-1], out_channels))
        super().__init__(*layers)
