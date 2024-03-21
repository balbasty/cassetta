__all__ = [
    'Cat',
    'Add',
    'Split',
    'DoNothing',
    'Hadamard',
]
import torch
from torch import nn


class Cat(nn.Module):
    """Concatenate tensors"""

    def __init__(self, dim=1):
        """
        Parameters
        ----------
        dim : int
            Dimension to concatenate. Default is 1, the channel dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, *args):
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
        return torch.cat(args, self.dim)


class Add(nn.Module):
    """Add tensors"""

    def forward(self, *args):
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
        return sum(args)


class Split(nn.Module):
    """Split tensor"""

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
    """A layer that does nothing"""

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
    Reparameterize tensors using the Hadamard transform.

    (x, y) -> (x + y, x - y)
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
