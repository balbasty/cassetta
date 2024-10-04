__all__ = [
    'LoadableMSE'
]

from torch.nn import MSELoss
from cassetta.io.modules import LoadableMixin


class LoadableMSE(LoadableMixin, MSELoss):
    """
    A loadable variant of PyTorch's MSE loss.
    [`torch.nn`][`torch.nn.MSELoss`]
    """
    @LoadableMixin.save_args
    def __init__(self):
        """
        Initialize the LoadableMSE loss function.
        """
        super().__init__()
