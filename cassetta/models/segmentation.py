__all__ = ['SegNet']
from torch import nn
from typing import Optional, Union
from cassetta.core.typing import ActivationType, OneOrSeveral
from cassetta.layers import ConvBlock, make_activation
from cassetta import backbones


class SegNet(nn.Sequential):
    r"""
    A generic segmentation network that works with any backbone

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            i["C"]:::i ---fx("Conv"):::w-->
            fi["F"]    ---b("Backbone"):::w -->
            fo["F"]    ---fk("Conv 1x1x1"):::w-->
            l["K"]     ---s(("σ")):::d-->
            o["K"]:::o
            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef n fill:none,stroke:none;
        ```
    """

    @property
    def inp_channels(self):
        return self[0].inp_channels

    @property
    def out_channels(self):
        return self[-1].out_channels

    def __init__(
        self,
        ndim: int,
        inp_channels: int,
        out_channels: int,
        kernel_size: OneOrSeveral[int] = 3,
        activation: ActivationType = 'Softmax',
        backbone: Union[str, nn.Module] = 'UNet',
        opt_backbone: Optional[dict] = None,
    ):
        """

        Parameters
        ----------
        ndim : {2, 3}
            Number of spatial dimensions
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output classes
        kernel_size : int, default=3
            Kernel size **of the initial feature extraction** layer
        activation : str
            **Final** activation function
        backbone : str or Module
            Generic backbone module. Can be already instantiated.

            Examples:
            [`UNet`][cassetta.backbones.UNet] (default),
            [`ATrousNet`][cassetta.backbones.ATrousNet],
            [`MeshNet`][cassetta.backbones.MeshNet].
        opt_backbone : dict
            Parameters of the backbone (if backbone is not pre-instantiated)
        """
        if isinstance(backbone, str):
            backbone_kls = getattr(backbones, backbone)
            backbone = backbone_kls(ndim, **(opt_backbone or {}))
        activation = make_activation(activation)
        feat = ConvBlock(
            ndim,
            inp_channels=inp_channels,
            out_channels=backbone.inp_channels,
            kernel_size=kernel_size,
            activation=None,
        )
        pred = ConvBlock(
            ndim,
            inp_channels=backbone.out_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        super().__init__(feat, backbone, pred)

    def predict_logits(self, inp):
        """
        Run the forward pass and return the logits (pre-softmax)

        Parameters
        ----------
        inp : (B, inp_channels, *size) tensor
            Input tensor

        Returns
        -------
        out : (B, out_channels, *size) tensor
            Logits
        """
        if not hasattr(self[-1], 'activation'):
            return self(inp)
        feat, backbone, pred = self
        return pred.conv(backbone(feat(inp)))
