__all__ = ['UNet']
# externals
from torch import nn

# internals
from cassetta.core.typing import (
    OneOrSeveral, ActivationType, NormType, DropoutType, AttentionType,
    Optional, Union, Literal
)
from cassetta.backbones.fcn import ConvEncoder, ConvDecoder


class UNet(nn.Module):
    """A UNet

    !!! tip "Diagram"
        === "`skip=True`"
            ```mermaid
            flowchart LR
                II0["`[F0, W]`"]            ---CI0("`ConvGroup`"):::w-->
                IO0["`[F0, W]`"]            ---D1("`Down`"):::w-->
                II1["`[F1, W//2]`"]         ---CI1("`ConvGroup`"):::w-->
                IO1["`[F1, W//2]`"]         ---D2("`Down`"):::w-->
                II2["`[F2, W//4]`"]         ---CI2("`ConvGroup`"):::w-->
                OO2["`[F2, W//4]`"]:::o     ---U2("`Up`"):::w-->
                OI1["`[F1, W//2]`"]         ---Z1(("c")):::d-->
                OZ1["`[F1*2, W//2]`"]       ---CO1("`ConvGroup`"):::w-->
                OO1["`[F1, W//2]`"]:::o     ---U1("`Up`"):::w-->
                OI0["`[F0, W]`"]            ---Z0(("c")):::d-->
                OZ0["`[F0*2, W]`"]          ---CO0("`ConvGroup`"):::w-->
                OO0["`[F0, W]`"]:::o
                IO0 --- Z0
                IO1 --- Z1
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef o fill:mistyrose,stroke:lightpink;
            ```
        === "`skip='+'`"
            ```mermaid
            flowchart LR
                II0["`[F0, W]`"]            ---CI0("`ConvGroup`"):::w-->
                IO0["`[F0, W]`"]            ---D1("`Down`"):::w-->
                II1["`[F1, W//2]`"]         ---CI1("`ConvGroup`"):::w-->
                IO1["`[F1, W//2]`"]         ---D2("`Down`"):::w-->
                II2["`[F2, W//4]`"]         ---CI2("`ConvGroup`"):::w-->
                OO2["`[F2, W//4]`"]:::o     ---U2("`Up`"):::w-->
                OI1["`[F1, W//2]`"]         ---Z1(("+")):::d-->
                OZ1["`[F1, W//2]`"]         ---CO1("`ConvGroup`"):::w-->
                OO1["`[F1, W//2]`"]:::o     ---U1("`Up`"):::w-->
                OI0["`[F0, W]`"]            ---Z0(("+")):::d-->
                OZ0["`[F0, W]`"]            ---CO0("`ConvGroup`"):::w-->
                OO0["`[F0, W]`"]:::o
                IO0 --- Z0
                IO1 --- Z1
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef o fill:mistyrose,stroke:lightpink;
            ```
        === "`skip=False`"
            ```mermaid
            flowchart LR
                II0["`[F0, W]`"]            ---CI0("`ConvGroup`"):::w-->
                IO0["`[F0, W]`"]            ---D1("`Down`"):::w-->
                II1["`[F1, W//2]`"]         ---CI1("`ConvGroup`"):::w-->
                IO1["`[F1, W//2]`"]         ---D2("`Down`"):::w-->
                II2["`[F2, W//4]`"]         ---CI2("`ConvGroup`"):::w-->
                OO2["`[F2, W//4]`"]:::o     ---U2("`Up`"):::w-->
                OI1["`[F1, W//2]`"]         ---CO1("`ConvGroup`"):::w-->
                OO1["`[F1, W//2]`"]:::o     ---U1("`Up`"):::w-->
                OI0["`[F0, W]`"]            ---CO0("`ConvGroup`"):::w-->
                OO0["`[F0, W]`"]:::o
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef o fill:mistyrose,stroke:lightpink;
            ```

    !!! note "Difference with Ronneberger et al."
        - Default parameters are from Ronneberger et al.
        - However, instead of performing a 3x3 channel-expanding
          convolution + ReLU in the encoder, we first perform a
          1x1 channel-expanding convolution without ReLU, followed by
          a 3x3 channel-preserving convolution + ReLU.
        - Both implementations have the same reprentation power,
          although ours adds unneeded free parameters.
        - The benefit of our approach is it brings a bit more flexibility.
          We can easily replace max-pooling with other types of downsampling
          operators (e.g., linear downsampling or strided convolution)
          using `pool_mode="interpolate"` or `pool_mode="conv"`.

    !!! quote "Reference"
        Ronneberger, Fischer & Brox, **"U-Net: Convolutional Networks
        for Biomedical Image Segmentation."**
        _MICCAI_ (2015). [arxiv:1505.04597](https://arxiv.org/abs/1505.04597)
    """  # noqa: E501

    @property
    def inp_channels(self):
        return self.encoder.inp_channels

    @property
    def out_channels(self):
        return self.decoder.out_channels

    @property
    def all_out_channels(self):
        return self.decoder.all_out_channels

    def __init__(
        self,
        ndim: int,
        nb_features: OneOrSeveral[int] = 64,
        mul_features: int = 2,
        nb_levels: int = 5,
        nb_levels_decoder: Optional[int] = None,
        nb_conv_per_level: int = 2,
        kernel_size: OneOrSeveral[int] = 3,
        residual: bool = False,
        activation: ActivationType = 'ReLU',
        norm: NormType = None,
        dropout: DropoutType = None,
        attention: AttentionType = None,
        order: str = 'cndax',
        pool_factor: OneOrSeveral[int] = 2,
        pool_mode: str = 'pool',
        unpool_mode: Optional[str] = 'conv',
        skip: Union[bool, Literal["+"]] = True,
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
        nb_levels_decoder : int, default=`nb_levels`
            Number of levels in the decoder
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
            Down/Upsampling factor (per dimension).
        pool_mode : {'interpolate', 'conv', 'pool'}
            Method used to go down one level.

            - If `"interpolate"`, use linear interpolation.
            - If `"conv"`, use strided convolutions.
            - If `"pool"`, use max pooling.
        unpool_mode : {'interpolate', 'conv'}, default=`pool_mode`
            Method used to go up one level.

            - If `"interpolate"`, use linear interpolation.
            - If `"conv"`, use transposed convolutions.
            - `"pool"` (i.e., unpooling) is not supported right now.
        skip : bool or {'+'}
            Type of skip connections:

            - `False`: no skip connections
            - `True`: concatenate skip connections
            - `'+'`: add skip connections
        """
        # number of features per level
        nb_levels_decoder = nb_levels_decoder or nb_levels
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
        dec_features = list(reversed(enc_features))
        dec_features += [
            max(1, int(dec_features[-1:] * mul_features**(-level)))
            for level in range(nb_levels_decoder - len(dec_features))
        ]
        dec_features = dec_features[:nb_levels_decoder]

        unpool_mode = unpool_mode or pool_mode
        if unpool_mode == 'pool':
            unpool_mode = 'interpol'

        # build encoder/decoder
        super().__init__()
        self.encoder = ConvEncoder(
            ndim,
            nb_features=enc_features,
            mul_features=1,
            nb_levels=nb_levels,
            nb_conv_per_level=nb_conv_per_level,
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            pool_factor=pool_factor,
            pool_mode=pool_mode,
        )
        self.decoder = ConvDecoder(
            ndim,
            nb_features=dec_features,
            nb_levels=nb_levels_decoder,
            nb_conv_per_level=nb_conv_per_level,
            skip=(skip and skip != '+'),
            kernel_size=kernel_size,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            attention=attention,
            order=order,
            unpool_factor=pool_factor,
            unpool_mode=unpool_mode,
        )
        self.skip = skip

    def forward(self, inp, *, return_all=False):
        """
        Parameters
        ----------
        inp : (B, nb_features[0], *inp_size)
            Input tensor
        return_all : bool
            Return all intermediate output tensors (at each level).

        Returns
        -------
        out : [tuple of] (B, nb_features[n], *out_size[n]) tensor
            Output tensor(s).

            - If `return_all`, return all intermediate tensors, from
              coarsest to finest.
            - Else, return the final tensor only.
        """
        out = self.encoder(inp, return_all=bool(self.skip))
        if self.skip:
            out = list(reversed(out))
        else:
            out = [out]
        return self.decoder(*out, return_all=return_all)
