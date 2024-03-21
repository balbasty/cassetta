from torch import nn
from functools import partial
from typing import Optional
from cassetta.core.typing import (
    OneOrSeveral, ActivationType, NormType, DropoutType, AttentionType)
from ..layers import ConvGroup, UpConvGroup, DownConvGroup


class UNet(nn.Module):
    """A UNet

    ```
    C -[conv xN]-> F ----------------------(cat)----------------------> 2*F -[conv xN]-> Cout
                   |                                                     ^
                   v                                                     |
                  F*m -[conv xN]-> F*m  ---(cat)---> 2*F*m -[conv xN]-> F*m
                                    |                  ^
                                    v                  |
                                  F*m*m -[conv xN]-> F*m*m
    ```
    """  # noqa: E501

    def __init__(
        self,
        ndim: int,
        nb_features: OneOrSeveral[int] = 16,
        mul_features: int = 2,
        nb_levels: int = 3,
        nb_levels_decoder: Optional[int] = None,
        nb_conv_per_level: int = 2,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'cndax',
        pool_mode: str = 'interpolate',
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output chanels
        nb_features : [list of] int
            Number of features at the finest level.
            If a list, number of features at each level of the encoder.
        mul_features : int
            Multiply the number of features by this number
            each time we go down one level.
        nb_levels : int
            Number of levels in the encoder
        nb_levels_decoder : int, default=`nb_levels`
            Number of levels in the decoder
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        activation : ActivationLike
            Type of activation
        norm : NormType
            Normalization
        dropout : DropoutType
            Channel dropout probability
        attention : AttentionType
            Attention
        pool_mode : {'interpolate', 'conv', 'pool'}
            Method used to go down/up one level.
            If `interpolate`, use `torch.nn.functional.interpolate`.
            If `conv`, use strided convolutions on the way down, and
            transposed convolutions on the way up.
        """
        make_inp = partial(
            ConvGroup,
            ndim,
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
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
            mode=pool_mode,
        )
        make_up = partial(
            UpConvGroup,
            ndim,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            nb_conv=nb_conv_per_level,
            mode=pool_mode,
        )

        # number of features per level
        nb_levels_decoder = nb_levels_decoder or nb_levels
        if isinstance(nb_features, int):
            enc_features = [
                nb_features * mul_features**level for level in range(nb_levels)
            ]
        else:
            enc_features = list(nb_features)
            enc_features += [
                enc_features[-1:] * mul_features**level
                for level in range(nb_levels - len(enc_features))
            ]
            enc_features = enc_features[:nb_levels]
        dec_features = list(reversed(enc_features))
        dec_features += [
            dec_features[-1:] * mul_features**(-level)
            for level in range(nb_levels_decoder - len(dec_features))
        ]
        dec_features = dec_features[:nb_levels_decoder]

        downpath = [make_inp(enc_features[0], enc_features[0])]
        for i in range(1, nb_levels):
            downpath += [make_down(enc_features[i-1], enc_features[i])]
        uppath = []
        for i in range(nb_levels_decoder-1):
            uppath += [make_up(dec_features[i], dec_features[i+1])]

        super().__init__()
        self.downpath = nn.Sequential(*downpath)
        self.uppath = nn.Sequential(*uppath)

    def forward(self, inp, return_pyramid=False):
        """
        Parameters
        ----------
        inp : (B, in_channels, X, Y)
            Input tensor

        Returns
        -------
        out : (B, out_channels, X, Y)
            Output tensor
        """
        x, skips, pyramid = inp, [], []
        for layer in self.downpath:
            x = layer(x)
            skips.append(x)
        skips.pop(-1)  # no need to keep the corsest features
        if return_pyramid:
            pyramid += [x]
        for layer in self.uppath:
            x = layer(x, skips.pop(-1))
            if return_pyramid:
                pyramid += [x]
        return tuple(pyramid) if return_pyramid else x
