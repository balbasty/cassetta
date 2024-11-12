__all__ = [
    'LoadableAdam',
]

from torch import optim
from cassetta.io.modules import LoadableOptimizer


class LoadableAdam(LoadableOptimizer, optim.Adam):
    """
    A loadable variant of [`optim.Adam`][torch.optim.Adam]

    This optimizer saves everything except model parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
