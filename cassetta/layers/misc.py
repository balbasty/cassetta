import torch
from torch import nn


class Cat(nn.Module):
    """Concatenate tensors along  the channel dimension"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.cat(args, self.dim)


class Add(nn.Module):
    """Add tensors"""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return sum(args)


class Split(nn.Module):
    """Split tensor along the channel dimension"""

    def __init__(self, nb_chunks=2, dim=1):
        super().__init__()
        self.dim = dim
        self.nb_chunks = nb_chunks

    def forward(self, x):
        return torch.tensor_split(x, self.nb_chunks, dim=self.dim)


class DoNothing(nn.Module):
    """
    Dummy layer
    """
    def forward(self, x, *args, **kwargs):
        return x
