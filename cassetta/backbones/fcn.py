__all__ = [
    'ConvEncoder',
    'ConvDecoder',
]
from torch import nn
from functools import partial
from typing import Union
from cassetta.core.typing import (
    OneOrSeveral, ActivationType, NormType, DropoutType, AttentionType)
from ..layers import ConvGroup, UpConvGroup, DownConvGroup


class ConvEncoder(nn.Sequential):
    """A fully convolutional encoder

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            1["`[F0, W]`"]    ---2("ConvGroup"):::w-->
            3["`[F0, W]`"]    ---4("Down"):::w-->
            5["`[F1, W//2]`"] ---6("ConvGroup"):::w-->
            7["`[F1, W//2]`"] --- 8("Down"):::w-->
            9["`[F2, W//4]`"] ---10("ConvGroup"):::w-->
            11["`[F2, W//4]`"]
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
    """  # noqa: E501

    def __init__(
        self,
        ndim: int,
        nb_features: OneOrSeveral[int] = 16,
        mul_features: int = 2,
        nb_levels: int = 3,
        nb_conv_per_level: int = 2,
        kernel_size: OneOrSeveral[int] = 3,
        residual: bool = False,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'cndax',
        pool_factor: OneOrSeveral[int] = 2,
        pool_mode: str = 'interpolate',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        nb_features : [list of] int
            Number of features at the finest level.
            If a list, number of features at each level of the encoder.
        mul_features : int
            Multiply the number of features by this number
            each time we go down one level.
        nb_levels : int
            Number of levels in the encoder
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        kernel_size : [list of] int
            Kernel size
        residual : bool
            Use residual connections between convolutional blocks
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
        pool_factor : [list of] int
            Downsampling factor (per dimension).
        pool_mode : {'interpolate', 'conv', 'pool'}
            Method used to go down one level.
        """
        make_inp = partial(
            ConvGroup,
            ndim,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
        )
        make_down = partial(
            DownConvGroup,
            ndim,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
            factor=pool_factor,
            mode=pool_mode,
        )

        # number of features per level
        if isinstance(nb_features, int):
            enc_features = [
                int(nb_features * mul_features**level)
                for level in range(nb_levels)
            ]
        else:
            enc_features = list(nb_features)
            enc_features += [
                int(enc_features[-1:] * mul_features**level)
                for level in range(nb_levels - len(enc_features))
            ]
            enc_features = enc_features[:nb_levels]

        # build encoder
        encoder = [make_inp(enc_features[0])]
        for i in range(1, nb_levels):
            encoder += [make_down(enc_features[i-1], enc_features[i])]
        super().__init__(*encoder)

    def forward(self, inp, *, return_all=False):
        """
        Parameters
        ----------
        inp : (B, nb_features[0], *inp_size) tensor
            Input tensor
        return_all : bool
            Return all intermediate output tensors (at each level)

        Returns
        -------
        out : [tuple of] (B, nb_features[-1], *out_size) tensor
            Output tensor(s).
            If `return_all`, return all intermediate tensors, from
            finest to coarsest. Else, return the final tensor only.
        """
        if return_all:
            out = inp
            all = []
            for layer in self:
                out = layer(out)
                all.append(out)
            return tuple(all)
        else:
            return super().forward(inp)


class ConvDecoder(nn.Sequential):
    """A fully convolutional decoder

    !!! tip "Diagram: pure decoder"
        === "No skip connections"
            ```mermaid
            flowchart LR
                1["`[F0, W]`"]     ---2("Up"):::w-->
                3["`[F1, W*2]`"]   ---4("ConvGroup"):::w-->
                5["`[F1, W*2]`"]   ---6("Up"):::w-->
                7["`[F2, W*4]`"]   ---8("ConvGroup"):::w-->
                9["`[F2, W*4]`"]
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        === "Concatenated skip connections (`skip!=0`)"
            ```mermaid
            flowchart LR
                S1["`[S1, W*2]`"]
                S2["`[S2, W*2]`"]
                1["`[F0, W]`"]        ---2("Up"):::w-->
                3["`[F1, W*2]`"]      ---4(("c")):::d-->
                5["`[F1+S1, W]`"]     ---6("ConvGroup"):::w-->
                7["`[F1, W*2]`"]      ---8("Up"):::w-->
                9["`[F2, W*4]`"]      ---10(("c")):::d-->
                11["`[F2+S2, W*4]`"]  ---12("ConvGroup"):::w-->
                13["`[F2, W*4]`"]
                S1 --- 4
                S2 --- 10
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
            ```
        === "Summed skip connections (`skip=0`)"
            ```mermaid
            flowchart LR
                S1["`[F1, W*2]`"]
                S2["`[F2, W*2]`"]
                1["`[F0, W]`"]        ---2("Up"):::w-->
                3["`[F1, W*2]`"]      ---4(("+")):::d-->
                5["`[F1, W]`"]        ---6("ConvGroup"):::w-->
                7["`[F1, W*2]`"]      ---8("Up"):::w-->
                9["`[F2, W*4]`"]      ---10(("+")):::d-->
                11["`[F2, W*4]`"]     ---12("ConvGroup"):::w-->
                13["`[F2, W*4]`"]
                S1 --- 4
                S2 --- 10
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
            ```
    """

    def __init__(
        self,
        ndim: int,
        nb_features: OneOrSeveral[int] = 16,
        div_features: int = 2,
        nb_levels: int = 3,
        nb_conv_per_level: int = 2,
        skip: Union[bool, OneOrSeveral[int]] = False,
        kernel_size: OneOrSeveral[int] = 3,
        residual: bool = False,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'cndax',
        unpool_factor: OneOrSeveral[int] = 2,
        unpool_mode: str = 'interpolate',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        nb_features : [list of] int
            Number of features at the finest level.
            If a list, number of features at each level of the encoder.
        div_features : int
            Divide the number of features by this number
            each time we go up one level.
        nb_levels : int
            Number of levels in the encoder
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        skip : int or bool
            Number of channels to concatenate in the skip connection.
            If 0 (or False) and skip tensors are provided, will try to
            add them instead of cat. If True, the number of skipped
            channels and the number of features are identical.
        kernel_size : [list of] int
            Kernel size
        residual : bool
            Use residual connections between convolutional blocks
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
        unpool_factor : [list of] int
            Upsampling factor (per dimension).
        unpool_mode : {'interpolate', 'conv'}
            Method used to go up one level.
        """
        make_up = partial(
            UpConvGroup,
            ndim,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
            factor=unpool_factor,
            mode=unpool_mode,
        )

        # number of features per level
        if isinstance(nb_features, int):
            dec_features = [
                max(1, int(nb_features // div_features**level))
                for level in range(nb_levels)
            ]
        else:
            dec_features = list(nb_features)
            dec_features += [
                max(1, int(dec_features[-1:] // div_features**level))
                for level in range(nb_levels - len(dec_features))
            ]
            dec_features = dec_features[:nb_levels]

        # number of skipped channels per level
        if skip is True:
            skip = dec_features[1:]
        elif not skip:
            skip = [0] * (nb_levels-1)
        elif isinstance(skip, int):
            skip = [skip] * (nb_levels - 1)
        else:
            skip = list(skip) + [0] * max(0, nb_levels - 1 - len(skip))

        # build decoder
        decoder = []
        for i in range(nb_levels-1):
            decoder += [make_up(
                dec_features[i], dec_features[i+1], skip=skip[i]
            )]
        super().__init__(*decoder)

    def forward(self, *inp, return_all=False):
        """
        Parameters
        ----------
        *inp : (B, nb_features[n], *inp_size[n]) tensor
            Input tensor(s), eventually including skip connections.
            Ordered from coarsest to finest.
        return_all : bool
            Return all intermediate output tensors (at each level).

        Returns
        -------
        out : [tuple of] (B, nb_features[-1], *out_size) tensor
            Output tensor(s).
            If `return_all`, return all intermediate tensors, from
            coarsest to finest. Else, return the final tensor only.
        """
        inp, *skips = inp
        skips = list(skips)
        all = []

        out = inp
        for layer in self:
            args = [skips.pop(0)] if skips else []
            out = layer(out, *args)
            if return_all:
                all.append(out)
        return tuple(all) if return_all else out
