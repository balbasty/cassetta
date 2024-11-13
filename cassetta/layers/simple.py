__all__ = [
    'Cat',
    'Add',
    'Split',
    'DoNothing',
    'Hadamard',
    'ModuleSum',
    'ModuleGroup',
    'GlobalPool',
]
# externals
import torch
from torch import nn
from torch import Tensor

# internals
from cassetta.core.typing import OneOrSeveral, List, Literal, Union


class Cat(nn.Module):
    """
    Concatenate tensors

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Inputs
                i1["C<sub>1</sub>"]:::i
                i2["C<sub>2</sub>"]:::i
                i3["..."]:::n
                i4["C<sub>N</sub>"]:::i
            end
            i1 & i2 & i3 & i4 ---z(("c")):::d--->
            o["C<sub>1</sub> + C<sub>2</sub> + ... + C<sub>N</sub>"]:::o

            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef n fill:none,stroke:none;
        ```
    """

    def __init__(self, dim=1):
        """
        Parameters
        ----------
        dim : int
            Dimension to concatenate. Default is 1, the channel dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        """
        Parameters
        ----------
        *inputs : tensor
            A series of tensors

        Returns
        -------
        output : tensor
            A single concatenated tensor
        """
        return torch.cat(inputs, self.dim)


class Add(nn.Module):
    """Add tensors

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Inputs
                i1["C"]:::i
                i2["C"]:::i
                i3["..."]:::n
                i4["C"]:::i
            end
            i1 & i2 & i3 & i4 ---z(("+")):::d--->
            o["C"]:::o

            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef n fill:none,stroke:none;
        ```
    """

    def forward(self, *inputs):
        """
        Parameters
        ----------
        *inputs : tensor
            A series of tensors

        Returns
        -------
        output : tensor
            A single summed tensor
        """
        return sum(inputs)


class Split(nn.Module):
    """Split tensor

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph Outputs
                o1["C"]:::o
                o2["C"]:::o
                o3["..."]:::n
                o4["C"]:::o
            end
            i["NxC"]:::i ---z(("s")):::d---> o1 & o2 & o3 & o4

            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef n fill:none,stroke:none;
        ```
    """

    def __init__(self, nb_chunks=2, dim=1):
        """
        Parameters
        ----------
        nb_chunks : int
            Number of output tensors
        dim : int
            Dimension to chunk. Default is 1, the channel dimension.
        """
        super().__init__()
        self.dim = dim
        self.nb_chunks = nb_chunks

    def forward(self, input):
        """
        Parameters
        ----------
        input : tensor
            The tensor to chunk

        Returns
        -------
        output : list[tensor]
            Tencor chunks
        """
        return torch.tensor_split(input, self.nb_chunks, dim=self.dim)


class DoNothing(nn.Module):
    """A layer that does nothing

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            i["C"]:::i ---> o["C"]:::o
            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
        ```
    """

    def forward(self, x, *args, **kwargs):
        return x


class MoveDim(nn.Module):
    """Move dimension in a tensor"""

    def __init__(self, src, dst):
        super().__init__()
        self.src = src
        self.dst = dst

    def forward(self, inp):
        return inp.movedim(self.src, self.dst)


class Hadamard(nn.Module):
    """
    Reparameterize tensors using the Hadamard transform:
    (x, y) -> (x + y, x - y)

    !!! tip "Diagram"
        === "Two tensors"
            ```mermaid
            flowchart LR
                x["C"]:::i
                y["C"]:::i
                x & y ---plus(("+")):::d---> oplus["C"]:::o
                x & y ---minus(("-")):::d---> ominus["C"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef n fill:none,stroke:none;
            ```
        === "One tensor"
            ```mermaid
            flowchart LR
                inp["2xC"]:::i ---split(("s")):::d---> x["C"] & y["C"]
                x & y ---plus(("+")):::d---> oplus["C"]
                x & y ---minus(("-")):::d---> ominus["C"]
                oplus & ominus ---cat(("c")):::d---> o["2xC"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef n fill:none,stroke:none;
            ```
    """
    def forward(self, x, y=None):
        """

        !!! note
            This layer can be applied to a single tensor, or to two tensors.

            * If two tensors are provided, their Hadamard transform is
              computed, and two tensors are returned.
            * If a single tensor is provided, it is split into two chunks,
              their Hadamard transform is computed, and the resulting chunks
              are concatenated and returned.

        Parameters
        ----------
        x, y : (B, C, *shape) tensor
            One or two tensors

        Returns
        -------
        hx, hy : (B, C, *shape) tensor
            One or two transformedtensors
        """
        if y is None:
            x, y = Split()(x)
            return Cat()(x + y, x - y)
        else:
            return x + y, x - y


class ModuleSum(nn.ModuleList):
    """
    Apply modules in parallel and sum their outputs.

    !!! tip "Diagram"
        ```mermaid
        flowchart LR
            subgraph nb_blocks
                2("Block 1"):::w  --> 3["C"]
                4("Block 2"):::w  --> 5["C"]
                6("..."):::n
                8("Block N"):::w  --> 9["C"]
            end
            1["C"]:::i --- 2 & 4 & 6 & 8
            3 & 5 & 6 & 9  --- 10(("+")):::d
            10 --> 11["C"]:::o
            classDef i fill:honeydew,stroke:lightgreen;
            classDef o fill:mistyrose,stroke:lightpink;
            classDef w fill:papayawhip,stroke:peachpuff;
            classDef d fill:lightcyan,stroke:lightblue;
            classDef n fill:none,stroke:none;
        ```

    !!! warning "The output of all modules must have the same shape"
    """

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, channels, *size) tensor
            Input tensor

        Returns
        -------
        out : (B, channels, *size) tensor
            Output tensor
        """
        out = 0
        for layer in self:
            out += layer(inp)
        return out


class ModuleGroup(nn.Sequential):
    r"""
    Multiple layers stacked together, eventually with residual connections.

    !!! tip "Diagram"
        === "`residual=False`"
            ```mermaid
            flowchart LR
                subgraph nb_blocks
                    2("Block 1"):::w  --> 3["C"] ---
                    4("Block 2"):::w  --> 5["C"] ---
                    6("..."):::n      --> 7["C"] ---
                    8("Block N"):::w
                end
                1["C"]:::i --- 2
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
                    2("Block 1"):::w  --> 3["C"] ---
                    4(("+")):::d      --> 5["C"] ---
                    6("Block 2"):::w  --> 7["C"] ---
                    8(("+")):::d      --> 9["C"] ---
                    10("..."):::n     --> 11["C"] ---
                    12("Block N"):::w --> 13["C"] ---
                    14(("+")):::d
                end
                1["C"]:::i --- 2
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
        === "`residual=True, skip!=0`"
            ```mermaid
            flowchart LR
                subgraph nb_blocks
                    2("Block 1"):::w  --> 3["C"] ---
                    4(("+")):::d      --> 5["C"] ---
                    6("Block 2"):::w  --> 7["C"] ---
                    8(("+")):::d      --> 9["C"] ---
                    10("..."):::n     --> 11["C"] ---
                    12("Block N"):::w --> 13["C"] ---
                    14(("+")):::d
                end
                1["C+S"]:::i --- 2
                1 --- split(("s")):::d --> c["C"] & s["S"]
                c --- 4
                s --- void[" "]:::n
                5 --- 8
                11 --- 14
                14 ---> 15["C"]:::o
                classDef i fill:honeydew,stroke:lightgreen;
                classDef o fill:mistyrose,stroke:lightpink;
                classDef w fill:papayawhip,stroke:peachpuff;
                classDef d fill:lightcyan,stroke:lightblue;
                classDef n fill:none,stroke:none;
                linkStyle 17 stroke:none;
            ```

    !!! note "The recurrent variant shares weights across blocks"

    !!! warning "The number of channels should be preserved throughout"

    !!! warning "The spatial size should be preserved throughout"
    """
    def __init__(
        self,
        blocks: List[nn.Module],
        residual: bool = False,
        skip: int = 0,
    ):
        """
        Parameters
        ----------
        blocks : list[Module]
            Number of blocks
        residual : bool
            Use residual connections between blocks
        skip : int
            Number of additional skipped channels in the input tensor.
        """
        super().__init__(*blocks)
        self.residual = residual
        self.skip = skip

    def forward(self, inp: Tensor) -> Tensor:
        """
        Parameters
        ----------
        inp : (B, channels [+skip], *size) tensor
            Input tensor

        Returns
        -------
        out : (B, channels, *size) tensor
            Output tensor
        """
        x = inp

        layers = list(self)
        if self.skip:
            first, *layers = layers
            if self.residual:
                identity = x
                x = first(x)
                x += identity[:, :x.shape[1]]
            else:
                x = first(x)

        if self.residual:
            for layer in layers:
                identity = x
                x = layer(x)
                x += identity
        else:
            for layer in layers:
                x = layer(x)
        return x


class GlobalPool(nn.Module):
    """
    Global pooling across spatial dimensions

    !!! tip "Diagram"
        === "`dim='spatial', keepdim=True`"
            ```mermaid
            flowchart LR
                1["`[B, C, W, H]`"] ---2("`GlobalPool`"):::d-->
                3["`[B, C, 1, 1]`"]
                classDef d fill:lightcyan,stroke:lightblue;
            ```
        === "`dim='spatial', keepdim=True`"
            ```mermaid
            flowchart LR
                1["`[B, C, W, H]`"] ---2("`GlobalPool`"):::d-->
                3["`[B, C]`"]
                classDef d fill:lightcyan,stroke:lightblue;
            ```
        === "`dim=1, keepdim=True`"
            ```mermaid
            flowchart LR
                1["`[B, C, W, H]`"] ---2("`GlobalPool`"):::d-->
                3["`[B, 1, W, H]`"]
                classDef d fill:lightcyan,stroke:lightblue;
            ```
    """

    def __init__(
        self,
        reduction: Literal['mean', 'max'] = 'mean',
        keepdim: bool = True,
        dim: Union[OneOrSeveral[int], Literal['spatial']] = 'spatial',
    ):
        """
        Parameters
        ----------
        reduction : {'mean', 'max'}
            Reduction type
        keepdim : bool
            Keep spatial dimensions
        dim : [list of] int or {'spatial'}
            Dimension(s) to pool
        """
        super().__init__()
        self.reduction = reduction.lower()
        self.keepdim = keepdim
        self.dim = dim

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, C, *spatial) tensor
            Input tensor

        Returns
        -------
        out : (B, C, [*ones]) tensor
            Output tensor
        """
        if isinstance(self.dim, str):
            if self.dim[0].lower() != 's':
                raise ValueError('Unknown dimension:', self.dim)
            dims = list(range(2, inp.ndim))
        else:
            dims = self.dim
        if self.reduction == 'max':
            return inp.max(dim=dims, keepdim=self.keepdim).values
        elif self.reduction == 'mean':
            return inp.mean(dim=dims, keepdim=self.keepdim)
        else:
            raise ValueError(f'Unknown reduction "{self.reduction}"')
