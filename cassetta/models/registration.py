__all__ = ['ElasticRegNet']
# externals
from torch import nn

# internals
from cassetta.core.typing import ActivationType, OneOrSeveral, Optional, Union
from cassetta.io.loadable import LoadableMixin
from cassetta.layers import ConvBlock, make_activation, FlowPull, FlowExp, Cat
from cassetta import backbones


class ElasticRegNet(LoadableMixin, nn.Sequential):
    r"""
    A generic pairwise nonlinear registration network that works with
    any backbone

    !!! bug "Not tested yet -- do not use"

    !!! tip "Diagram"
        === "Prediction"
            ```mermaid
            flowchart LR
                subgraph "Prediction"
                    cat(("c")):::d-->
                    mf["C*2"]  ---fx("Conv"):::w-->
                    fi["F"]    ---b("Backbone"):::w -->
                    fo["F"]    ---fk("Conv 1x1x1"):::w
                end
                mov["C"]:::i & fix["C"]:::i ---cat
                fk --> o["D"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef n fill:none,stroke:none;
            ```
        === "`symmetric=True`"
            ```mermaid
            flowchart LR
                mf["(mov, fix)"]:::i ---p1("Prediction"):::w--> v1["D"]
                fm["(fix, mov)"]:::i ---p2("Prediction"):::w--> v2["D"]
                v1 & v2 ---minus(("-")):::d--> o["D"]:::o
                p1 -.-|"shared weights"| p2
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef n fill:none,stroke:none;
            ```
        === "`predict_moved(mov, fix)`"
            ```mermaid
            flowchart LR
                mov["<b>mov</b>\nC"]:::i & fix["<b>fix</b>\nC"]:::i ---
                pred("Prediction"):::w-->
                vel["<b>velocity</b>\nD"] ---exp("Exp"):::d -->
                flow["<b>flow</b>\nD"]
                mov & flow ---w("Pull"):::d--> out["<b>moved</b>\nC"]:::o
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
        symmetric: bool = False,
        nb_steps: int = 0,
        inp_channels: int = 1,
        kernel_size: OneOrSeveral[int] = 3,
        activation: ActivationType = None,
        backbone: Union[str, nn.Module] = 'UNet',
        opt_backbone: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        ndim : {2, 3}
            Number of spatial dimensions
        symmetric : bool
            Make the network symmetric by averaging `model(mov, fix)` and
            `model(fix, mov)`.
        nb_steps : int
            Number of scaling and squaring steps.
        inp_channels : int
            Number of input channels, per image.
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
            Parameters of the backbone (if backbone is not pre-instantiated).

            Note that, unless user-defined, we set `activation="LeakyReLU"`.
        """
        opt_backbone.setdefault('activation', 'LeakyReLU')
        if isinstance(backbone, str):
            backbone_kls = getattr(backbones, backbone)
            backbone = backbone_kls(ndim, **(opt_backbone or {}))
        activation = make_activation(activation)
        feat = ConvBlock(
            ndim,
            inp_channels=inp_channels*2,
            out_channels=backbone.inp_channels,
            kernel_size=kernel_size,
            activation=None,
        )
        pred = ConvBlock(
            ndim,
            inp_channels=backbone.out_channels,
            out_channels=ndim,
            kernel_size=1,
            activation=activation,
        )
        super().__init__()
        self.net = nn.Sequential(feat, backbone, pred)
        self.exp = FlowExp(nb_steps)
        self.warp = FlowPull()
        self.symmetric = symmetric

    def forward(self, mov, fix):
        """
        Predict the encoding of the displacement field

        Parameters
        ----------
        mov : (B, inp_channels, *size) tensor
            Moving image
        fix : (B, inp_channels, *size) tensor
            Fixed image

        Returns
        -------
        vel : (B, ndim, *size) tensor
            - If `nb_steps>0`: stationary velocity field
            - Else: voxel displacement field
        """
        vel = self.net(Cat()(mov, fix))
        if self.symmetric:
            vel -= self.net(Cat()(fix, mov))
            vel *= 0.5
        return vel

    def predict_flow(self, mov, fix):
        """
        Predict the forward displacement field, used to warp `mov` to `fix`

        Parameters
        ----------
        mov : (B, inp_channels, *size) tensor
            Moving image
        fix : (B, inp_channels, *size) tensor
            Fixed image

        Returns
        -------
        flow : (B, ndim, *size) tensor
            Voxel displacement field
        """
        return self.exp(self(mov, fix))

    def predict_flows(self, mov, fix):
        """
        Predict the forward displacement field (used to warp `mov` to `fix`)
        and thr backward displacement field (used to warp `fix` to `mov`)

        Parameters
        ----------
        mov : (B, inp_channels, *size) tensor
            Moving image
        fix : (B, inp_channels, *size) tensor
            Fixed image

        Returns
        -------
        flow_forward : (B, ndim, *size) tensor
            Forward voxel displacement field
        flow_backward : (B, ndim, *size) tensor
            Backward voxel displacement field
        """
        vel = self(mov, fix)
        return self.exp(vel), self.exp(-vel)

    def predict_moved(self, mov, fix):
        """
        Predict the warped moving image

        Parameters
        ----------
        mov : (B, inp_channels, *size) tensor
            Moving image
        fix : (B, inp_channels, *size) tensor
            Fixed image

        Returns
        -------
        moved : (B, inp_channels, *size) tensor
            Moved image
        """
        flow = self.predict_flow(mov, fix)
        return self.warp(mov, flow)

    def predict_both_moved(self, mov, fix):
        """
        Predict the warped moving image, and the warped fixed image.

        Parameters
        ----------
        mov : (B, inp_channels, *size) tensor
            Moving image
        fix : (B, inp_channels, *size) tensor
            Fixed image

        Returns
        -------
        warped_mov : (B, inp_channels, *size) tensor
            Moving image warped to fixed space
        warped_fix : (B, inp_channels, *size) tensor
            Fixed image warped to moving space
        """
        fwd, bwd = self.predict_flows(mov, fix)
        return self.warp(mov, fwd), self.warp(fix, bwd)
