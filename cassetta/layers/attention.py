__all__ = [
    'make_attention',
    'ChannelSqzEx',
    'SpatialSqzEx',
    'SqzEx',
    'ChannelBlockAttention',
    'SpatialBlockAttention',
    'BlockAttention',
    'DotProductAttention',
    'MultiHeadAttention',
]
from torch import nn
from torch import Tensor
from typing import Literal, Optional
from cassetta.core.typing import (
    ActivationType, OneOrSeveral, AttentionType, DeviceType, DataType)
from cassetta.core.utils import to_torch_dtype
from .activations import make_activation
from .simple import Cat, MoveDim, GlobalPool
from .linear import Linear
from .conv import Conv


def make_attention(
    attention: AttentionType,
    channels: int,
    ndim: int = None,
    **kwargs
):
    """
    Instantiate an attention layer

    Parameters
    ----------
    attention : AttentionType
        An already instantiated `nn.Module`, or a `nn.Module` subclass,
        or a callable that retgurns an instantiated `nn.Module`, or the
        name of an attention type:

        - `"sqzex"` : Squeeze & Excite
        - `"cbam"` : Convolutional Block Attention Module
        - `"dp"` : Dot-Product Attention
        - `"sdp"` : Scaled Dot-Product Attention
        - `"mha"` : Multi-Head Attention
    channels : int
        Number of channels
    ndim : int
        Number of spatial dimensions

    Returns
    -------
    attention : Module
        An attention layer
    """
    if not attention:
        return None
    if isinstance(attention, nn.Module):
        return attention
    if isinstance(attention, str):
        attention = attention.lower()
        if attention == 'sqzex':
            attention = SqzEx
        elif attention == 'cbam':
            attention = BlockAttention
        elif attention == 'dp':
            attention = DotProductAttention
            kwargs.setdefault('scaled', False)
        elif attention == 'sdp':
            attention = DotProductAttention
            kwargs.setdefault('scaled', True)
        elif attention == 'mha':
            attention = MultiHeadAttention
        else:
            raise ValueError(f'Unknown attention "{attention}"')
    kwargs['ndim'] = ndim
    kwargs['channels'] = channels
    attention = attention(**kwargs)
    if not isinstance(attention, nn.Module):
        raise ValueError('Attention did not instantiate a Module')
    return attention


class ChannelSqzEx(nn.Sequential):
    r"""
    Spatial Squeeze & Channel Excitation layer

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Spatial Squeeze
            1["`[C, W]`"]    ---2("`MeanPool`"):::d-->  3["`[C, 1]`"]
            end
            subgraph MLP
                                4("`Linear`"):::w   -->
            5["`[C//r, 1]`"] ---6("`ReLU`"):::d     -->
            7["`[C//r, 1]`"] ---8("`Linear`"):::w
            end
            subgraph Channel Excitation
            9["`[C, 1]`"]    ---10("`Sigmoid`"):::d -->
            11["`[C, 1]`"]   ---12(("*")):::d       -->
            13["`[C, W]`"]
            end
            3 --- 4
            8 --> 9
            1 --> 12
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Hu, J, et al. **"Squeeze-and-Excitation Networks."**
            _CVPR_ (2018), _TPAMI_ (2019).
            [arxiv:1709.01507](https://arxiv.org/abs/1709.01507)

        2.  Roy, AG, et al. **"Concurrent Spatial and Channel
            Squeeze & Excitation in Fully Convolutional Networks"**
            _MICCAI_ (2018).
            [arxiv:1803.02579](https://arxiv.org/abs/1803.02579)

    """

    def __init__(
        self,
        channels: int,
        compression: int = 16,
        activation: ActivationType = 'ReLU',
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels
        compression : int
            Compression ratio for the number of channels in the squeeze
        activation : ActivationType
            Activation function
        """
        opt = dict(bias=False, device=device, dtype=to_torch_dtype(dtype))
        super().__init__(
            GlobalPool(reduction='mean', keepdim=True),
            Linear(channels, max(1, channels//compression), **opt),
            make_activation(activation),
            Linear(max(1, channels//compression), channels, **opt),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        return inp * super().forward(inp)


class SpatialSqzEx(nn.Sequential):
    """
    Channel Squeeze & Spatial Excitation layer

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Channel Squeeze
            1["`[C, W]`"]    ---2("`Linear`"):::w-->  3["`[1, W]`"]
            end
            subgraph Spatial Excitation
            4("`Sigmoid`"):::d--> 5["`[1, W]`"] ---6(("*")):::d-->7["`[C, W]`"]
            end
            3 --- 4
            1 --> 6
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Roy, AG, et al. **"Concurrent Spatial and Channel
            Squeeze & Excitation in Fully Convolutional Networks"**
            _MICCAI_ (2018).
            [arxiv:1803.02579](https://arxiv.org/abs/1803.02579)

    """

    def __init__(
        self,
        channels: int,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels
        """
        opt = dict(bias=False, device=device, dtype=to_torch_dtype(dtype))
        super().__init__(
            Linear(channels, 1, **opt),
            nn.Sigmoid(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        return inp * super().forward(inp)


class SqzEx(nn.Sequential):
    r"""
    Concurrent Spatial and Channel Squeeze & Spatial Excitation layer

    !!! tip "Diagram"
        === "`mode='+'`"
            ```mermaid
            flowchart LR
                6(("+")):::d  --> 7["`[C, W]`"]
                1["`[C, W]`"] ---2("Channel Squeeze & Excite"):::w-->
                3["`[C, W]`"]
                1             ---4("Spatial Squeeze & Excite"):::w-->
                5["`[C, W]`"]
                3 --- 6
                5 --> 6
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        === "`mode='cs'`"
            ```mermaid
            flowchart LR
                1["`[C, W]`"] ---2("Channel Squeeze & Excite"):::w-->
                3["`[C, W]`"] ---4("Spatial Squeeze & Excite"):::w-->
                5["`[C, W]`"]
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        === "`mode='sc'`"
            ```mermaid
            flowchart LR
                1["`[C, W]`"] ---2("Spatial Squeeze & Excite"):::w-->
                3["`[C, W]`"] ---4("Channel Squeeze & Excite"):::w-->
                5["`[C, W]`"]
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Roy, AG, et al. **"Concurrent Spatial and Channel
            Squeeze & Excitation in Fully Convolutional Networks"**
            _MICCAI_ (2018).
            [arxiv:1803.02579](https://arxiv.org/abs/1803.02579)

    """

    def __init__(
        self,
        channels: int,
        mode: Literal['+', 'c', 's', 'cs', 'sc'] = '+',
        compression: int = 16,
        activation: ActivationType = 'ReLU',
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        channels : int
            Number of input and output channels
        mode : {'+', 'c', 's', 'sc', 'cs'}
            Squeeze and excitation mode:

            - `'c'` : channel only
            - `'s'` : spatial only
            - `'cs'` : channel, then spatial
            - `'sc'` : spatial, then channel
            - `'+'` : concurrent spatial and channel
        compression : int
            Compression ratio for the number of channels in the squeeze
        activation : ActivationType
            Activation function
        """
        mode = mode.lower()
        opt = dict(device=device, dtype=to_torch_dtype(dtype))
        if 's' in mode or '+' in mode:
            s = SpatialSqzEx(channels, **opt)
        if 'c' in mode or '+' in mode:
            c = ChannelSqzEx(channels, compression, activation, **opt)
        if mode == 's':
            layers = [s]
        elif mode == 'c':
            layers = [c]
        elif mode == 'cs':
            layers = [c, s]
        elif mode == 'sc':
            layers = [s, c]
        elif mode == '+':
            layers = [s, c]
        else:
            raise ValueError(f'Unknown mode "{mode}"')
        super().__init__(*layers)
        self.mode = mode

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        if self.mode == '+':
            return sum([layer(inp) for layer in self])
        else:
            return super().forward(inp)


class ChannelBlockAttention(nn.Sequential):
    """
    Channel Attention for Convolutional Block Attention Module

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Spatial Squeeze
            1["`[C, W]`"]    ---2("`MeanPool`"):::d-->  3["`[C, 1]`"]
            1                ---4("`MaxPool`"):::d-->   5["`[C, 1]`"]
            end
            subgraph MLP - shared weights
                                6("`Linear`"):::w   -->
            7["`[C//r, 1]`"] ---8("`ReLU`"):::d     -->
            9["`[C//r, 1]`"] ---10("`Linear`"):::w
            end
            subgraph MLP - shared weights
                                 11("`Linear`"):::w   -->
            12["`[C//r, 1]`"] ---13("`ReLU`"):::d     -->
            14["`[C//r, 1]`"] ---15("`Linear`"):::w
            end
            subgraph Channel Attention
            20("`Sigmoid`"):::d -->
            21["`[C, 1]`"] --- 22(("*")):::d  --> 23["`[C, W]`"]
            end
            16["`[C, 1]`"] & 17["`[C, 1]`"] ---18(("+")):::d--> 19["`[C, 1]`"]
            3 --- 6
            5 --- 11
            10 --> 16
            15 --> 17
            19 --- 20
            1 --> 22
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Woo, S, et al.
            **"CBAM: Convolutional Block Attention Module."**
            _ECCV_ (2018).
            [arxiv:1807.06521](https://arxiv.org/abs/1807.06521)

    """

    def __init__(
        self,
        channels: int,
        compression: int = 16,
        activation: ActivationType = 'ReLU',
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        channels : int
            Number of input and output channels
        compression : int
            Compression ratio for the number of channels in the squeeze
        activation : ActivationType
            Activation function
        """
        opt = dict(device=device, dtype=to_torch_dtype(dtype))
        super().__init__(
            GlobalPool(keepdim=False, reduction='mean'),
            GlobalPool(keepdim=False, reduction='max'),
            Linear(channels, max(1, channels//compression), bias=False, **opt),
            make_activation(activation),
            Linear(max(1, channels//compression), channels, bias=False, **opt),
            nn.Sigmoid(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        ndim = inp.ndim - 2
        meanpool, maxpool, *mlp, sigmoid = self
        mlp = nn.Sequential(*mlp)
        out = sigmoid(mlp(meanpool(inp)) + mlp(maxpool(inp)))
        out = out.reshape(out.shape + (1,) * ndim)
        out = inp * out
        return out


class SpatialBlockAttention(nn.Sequential):
    """
    Spatial Attention for Convolutional Block Attention Module

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Channel Squeeze
            1["`[C, W]`"]    ---2("`ChannelMean`"):::d-->  3["`[1, W]`"]
            1                ---4("`ChannelMax`"):::d-->   5["`[1, W]`"]
            end
            3 & 5 ---6(("c")):::d--> 7["`[2, W]`"]
            7 ---8("`Conv 7`"):::w--> 9["`[1, W]`"]
            subgraph Spatial Excitation
            10("`Sigmoid`"):::d--> 11["`[1, W]`"] ---12(("*")):::d-->
            13["`[C, W]`"]
            end
            9 --> 10
            1 --> 12
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
        ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Woo, S, et al.
            **"CBAM: Convolutional Block Attention Module."**
            _ECCV_ (2018).
            [arxiv:1807.06521](https://arxiv.org/abs/1807.06521)

    """

    def __init__(
        self,
        ndim: int,
        kernel_size: OneOrSeveral[int] = 7,
        device: Optional[DeviceType] = None,
        dtype: Optional[DataType] = None,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dim
        kernel_size : [list of] int
            Kernel size of the convolution layer
        """
        opt = dict(bias=False, device=device, dtype=to_torch_dtype(dtype))
        super().__init__(
            GlobalPool(keepdim=True, dim=1, reduction='mean'),
            GlobalPool(keepdim=True, dim=1, reduction='max'),
            Conv(ndim, 2, 1, kernel_size=kernel_size, **opt),
            nn.Sigmoid(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        meanpool, maxpool, conv, sigmoid = self
        out = sigmoid(conv(Cat()(meanpool(inp), maxpool(inp))))
        out = inp * out
        return out


class BlockAttention(nn.Sequential):
    """
    Channel + Spatial Attention layer

    !!! tip "Diagram"
        === "`mode='+'`"
            ```mermaid
            flowchart LR
                6(("+")):::d  --> 7["`[C, W]`"]
                1["`[C, W]`"] ---2("Channel Attention"):::w--> 3["`[C, W]`"]
                1             ---4("Spatial Attention"):::w--> 5["`[C, W]`"]
                3 --- 6
                5 --> 6
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        === "`mode='cs'`"
            ```mermaid
            flowchart LR
                1["`[C, W]`"] ---2("Channel Attention"):::w-->
                3["`[C, W]`"] ---4("Spatial Attention"):::w-->
                5["`[C, W]`"]
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        === "mode='sc'`"
            ```mermaid
            flowchart LR
                1["`[C, W]`"] ---2("Spatial Attention"):::w-->
                3["`[C, W]`"] ---4("Channel Attention"):::w-->
                5["`[C, W]`"]
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
            ```
        Note that the batch dimension is not represented, but must
        be present.

    !!! quote "Reference"

        1.  Woo, Sanghyun, et al.
            **"CBAM: Convolutional Block Attention Module."**
            _ECCV_ (2018). https://arxiv.org/abs/1807.06521v2

    """

    def __init__(
        self,
        ndim: int,
        channels: int,
        mode: Literal['c', 's', 'cs', 'sc', '+'] = 'cs',
        compression: int = 16,
        activation: ActivationType = 'ReLU',
        kernel_size: OneOrSeveral[int] = 7,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dim
        channels : int
            Number of input and output channels
        mode : {'cs', 'sc', 'c', 's', '+'}
            Attention mode:

            - `'c'` : channel only
            - `'s'` : spatial only
            - `'cs'` : channel, then spatial
            - `'sc'` : spatial, then channel
            - `'+'` : concurrent spatial and channel
        compression : int
            Compression ratio for the number of channels in the squeeze
        activation : ActivationType
            Activation function
        kernel_size : [list of] int
            Kernel size of the convolution layer
        """
        mode = mode.lower()
        if 's' in mode or '+' in mode:
            s = SpatialBlockAttention(ndim, kernel_size)
        if 'c' in mode or '+' in mode:
            c = ChannelBlockAttention(channels, compression, activation)
        if mode == 's':
            layers = [s]
        elif mode == 'c':
            layers = [c]
        elif mode == 'cs':
            layers = [c, s]
        elif mode == 'sc':
            layers = [s, c]
        elif mode == '+':
            layers = [s, c]
        else:
            raise ValueError(f'Unknown mode "{mode}"')
        super().__init__(*layers)
        self.mode = mode

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        if self.mode == '+':
            return sum([layer(inp) for layer in self])
        else:
            return super().forward(inp)


class DotProductAttention(nn.Module):
    """
    !!! quote "References"

        1.  Vaswani, Ashish, et al. **"Attention Is All You Need."
            _NeurIPS_ (2017). https://arxiv.org/abs/1706.03762v7
    """

    def __init__(
        self,
        key_channels: int,
        val_channels: int,
        scaled=True,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        key_channels: int
            Number of keys
        val_channels: int
            Number of values
        scaled : bool
            Scale the dot product
        """
        super().__init__()
        self.key_channels = key_channels
        self.val_channels = val_channels
        self.scaled = scaled

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, K+K*V+V, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, V, *spatial) tensor
        """
        nk, nv = self.key_channels, self.val_channels
        q, k, v = inp.split([nk, nk*nv, nv], dim=1)
        q, k, v = q.movedim(1, -1), k.movedim(1, -1), v.movedim(1, -1)
        q, k = q.unsqueeze(-2), k.reshape(k.shape[:-1] + (nk, nv))
        qk = q.matmul(k).squeeze(-2)
        if self.scaled:
            qk /= self.key_channels ** 0.5
        qk = nn.Softmax(dim=-1)
        return (qk * v).movedim(-1, 1)


class MultiHeadAttention(nn.Module):
    """
    !!! quote "References"

        1.  Vaswani, Ashish, et al. **"Attention Is All You Need."
            _NeurIPS_ (2017). https://arxiv.org/abs/1706.03762v7
    """

    def __init__(
        self,
        inp_channels: int,
        key_channels: int,
        val_channels: int,
        nb_heads: int,
        scaled=True,
        **unused_kwargs,
    ):
        """
        Parameters
        ----------
        inp_channels: int
            Number of input channels
        key_channels: int
            Number of keys
        val_channels: int
            Number of values
        nb_heads : int
            Number of heads
        scaled : bool
            Scale the dot product
        """
        super().__init__()
        qkv_channels = (
            key_channels * val_channels + key_channels + val_channels
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                MoveDim(1, -1),
                nn.Linear(inp_channels, qkv_channels, bias=False),
                MoveDim(-1, 1),
                DotProductAttention(key_channels, val_channels, scaled=scaled)
            )
            for _ in range(nb_heads)
        ])
        self.combine = nn.Sequential(
            MoveDim(1, -1),
            nn.Linear(val_channels*nb_heads, inp_channels),
            MoveDim(-1, 1),
        )

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, C, *spatial) tensor
        """
        out = Cat()([head(inp) for head in self.heads])
        out = self.combine(out)
        return out
