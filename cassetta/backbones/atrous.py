__all__ = [
    'MeshNet',
    'ATrousNet',
]
from functools import partial
from cassetta.core.typing import (
    OneOrSeveral, ActivationType, NormType, DropoutType, AttentionType)
from ..layers.convblocks import ConvGroup, ModuleGroup
from ..layers.simple import ModuleSum


class MeshNet(ModuleGroup):
    """
    A stack of dilated convolutions

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            II0["`[F, W]`"]         ---CI0("`ConvGroup(dilation=1)`"):::w-->
            IO0["`[F, W]`"]         ---CI1("`ConvGroup(dilation=2)`"):::w-->
            IO1["`[F, W]`"]         ---CI2("`ConvGroup(dilation=4)`"):::w-->
            OO2["`[F, W]`"]         ---CO1("`ConvGroup(dilation=8)`"):::w-->
            OO1["`[F, W]`"]         ---CO0("`ConvGroup(dilation=16)`"):::w-->
            OO0["`[F, W]`"]:::o
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef o fill:mistyrose,stroke:lightpink;
        ```

    !!! note "Difference with Fedorov et al."
        - Default parameters are from Fedorov et al.
        - However, Fedorov et al. end with a final convolution block
          with `dilation=1`, which our default network discards.
        - To recover their behavior, explictely set the dilation list:
          `dilation=[1, 2, 4, 8, 16, 1]`.

    !!! quote "References"
        1.  Yu & Koltun, **"Multi-Scale Context Aggregation by Dilated
            Convolutions."** _ICLR_ (2016).
            [arxiv:1511.07122](https://arxiv.org/abs/1511.07122)

        2.  Fedorov, Johnson, Damaraju, Ozerin, Calhoun & Plis, **"End-to-end
            learning of brain tissue segmentation from imperfect labeling."**
            _IJCNN_ (2017).
            [arxiv:1612.00940](https://arxiv.org/abs/1612.00940)
    """

    def __init__(
        self,
        ndim: int,
        nb_features: int = 21,
        nb_layers: int = 6,
        nb_conv_per_layer: int = 2,
        dilation: OneOrSeveral[int] = 1,
        mul_dilation: int = 2,
        kernel_size: OneOrSeveral[int] = 3,
        residual: bool = False,
        activation: ActivationType = 'ReLU',
        norm: NormType = 'batch',
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'caxnd',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        nb_features : int
            Number of features at the finest level.
            If a list, number of features at each level of the encoder.
        nb_layers : int
            Number of levels in the network.
        nb_conv_per_layers : int
            Number of convolutional blocks in each layer.
        dilation : [list of] int
            Dilation factor in the first layer.
            If a list, number of features in each layer.
        mul_dilation : int
            Multiply the dilation by this number
            each time we go down one level.
        kernel_size : [list of] int
            Kernel size
        residual : bool
            Use residual connections between convolutional blocks and
            between layers.
        activation : ActivationLike
            Type of activation
        norm : NormType
            Normalization
        dropout : DropoutType
            Channel dropout probability
        attention : AttentionType
            Attention
        order : str
            Modules order (permutation of 'ncdax')
        """
        if isinstance(dilation, int):
            dilation = [
                int(dilation * mul_dilation**layer)
                for layer in range(nb_layers)
            ]
        else:
            dilation = list(dilation)
            dilation += [
                int(dilation[-1:] * mul_dilation**layer)
                for layer in range(nb_layers - len(dilation))
            ]
            dilation = dilation[:nb_layers]

        make_layer = partial(
            ConvGroup,
            ndim,
            channels=nb_features,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_layer,
        )
        layers = [make_layer(dilation=d) for d in dilation]
        super().__init__(layers, residual=residual)


class ATrousNet(ModuleGroup):
    """
    Parallel dilated convolutions

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            1["`[F, W]`"] ---C11("`ConvGroup(dilation=1)`"):::w-->
            2["`[F, W]`"] ---C21("`ConvGroup(dilation=1)`"):::w--> 3["`[F, W]`"]
            2             ---C22("`ConvGroup(dilation=2)`"):::w--> 4["`[F, W]`"]
            3 & 4         ---Z2(("+")):::d-->
            5["`[F, W]`"] ---C31("`ConvGroup(dilation=1)`"):::w--> 6["`[F, W]`"]
            5             ---C32("`ConvGroup(dilation=2)`"):::w--> 7["`[F, W]`"]
            5             ---C34("`ConvGroup(dilation=4)`"):::w--> 8["`[F, W]`"]
            6 & 7 & 8     ---Z3(("+")):::d-->
            9["`[F, W]`"] ---C41("`ConvGroup(dilation=1)`"):::w-->10["`[F, W]`"]
            9             ---C42("`ConvGroup(dilation=2)`"):::w-->11["`[F, W]`"]
            10 & 11       ---Z4(("+")):::d-->
            12["`[F, W]`"]---C51("`ConvGroup(dilation=1)`"):::w-->13["`[F, W]`"]:::o
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef o fill:mistyrose,stroke:lightpink;
        ```

    !!! quote "Reference"
        Chen, Papandreou, Kokkinos, Murphy & Yuille,
        **"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs."**
        _TPAMI_ (2017). [arxiv:1606.00915](https://arxiv.org/abs/1606.00915)
    """  # noqa: E501

    def __init__(
        self,
        ndim: int,
        nb_features: int = 21,
        nb_levels: int = 5,
        nb_conv_per_level: int = 2,
        dilation: OneOrSeveral[int] = 1,
        mul_dilation: int = 2,
        kernel_size: OneOrSeveral[int] = 3,
        residual: bool = False,
        activation: ActivationType = 'ReLU',
        norm: NormType = 'batch',
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'caxnd',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        nb_features : int
            Number of features at the finest level.
            If a list, number of features at each level of the encoder.
        nb_levels : int
            Number of levels in the network.
        nb_conv_per_level : int
            Number of convolutional blocks in each layer.
        dilation : [list of] int
            Dilation factor in the first layer.
            If a list, number of features in each layer.
        mul_dilation : int
            Multiply the dilation by this number
            each time we go down one level.
        kernel_size : [list of] int
            Kernel size
        residual : bool
            Use residual connections between convolutional blocks and
            between layers.
        activation : ActivationLike
            Type of activation
        norm : NormType
            Normalization
        dropout : DropoutType
            Channel dropout probability
        attention : AttentionType
            Attention
        order : str
            Modules order (permutation of 'ncdax')
        """
        if isinstance(dilation, int):
            enc_dilation = [
                int(dilation * mul_dilation**layer)
                for layer in range(nb_levels)
            ]
        else:
            dilation = list(dilation)
            enc_dilation += [
                int(enc_dilation[-1:] * mul_dilation**layer)
                for layer in range(nb_levels - len(dilation))
            ]
            enc_dilation = enc_dilation[:nb_levels]

        make_layer = partial(
            ConvGroup,
            ndim,
            channels=nb_features,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
        )
        layers = []
        for level in range(nb_levels):
            sublayers = []
            for n in range(level+1):
                sublayers += [make_layer(dilation=enc_dilation[n])]
            layers += [ModuleSum(sublayers)]
        for level in range(nb_levels-1, 0, -1):
            sublayers = []
            for n in range(level):
                sublayers += [make_layer(dilation=enc_dilation[n])]
            layers += [ModuleSum(sublayers)]
        super().__init__(layers, residual=residual)
