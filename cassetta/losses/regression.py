__all__ = [
    'LoadableMSE'
]

from torch.nn import MSELoss
from cassetta.io.loadable import LoadableMixin


class LoadableMSE(LoadableMixin, MSELoss):
    """
    A loadable variant of PyTorch's MSE loss.
    [`torch.nn`][`torch.nn.MSELoss`]
    """
    ...
