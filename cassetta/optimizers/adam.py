__all__ = [
    'LoadableAdam',
]

from torch import optim
from cassetta.io.optim import LoadableOptimizer


class LoadableAdam(LoadableOptimizer, optim.Adam):
    """
    A loadable variant of [`optim.Adam`][torch.optim.Adam]

    This optimizer saves everything except model parameters.
    """
    ...
