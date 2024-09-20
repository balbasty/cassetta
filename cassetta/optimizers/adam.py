__all__ = [
    'LoadableAdam'
]

from torch.optim import Adam
from cassetta.io.modules import LoadableMixin


class LoadableAdam(LoadableMixin, Adam):
    """
    A loadable variant of PyTorch's Adam optimizer. [torch.optim.Adam]
    """
    @LoadableMixin.save_args
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        """
        Initialize the LoadableAdam optimizer.
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
