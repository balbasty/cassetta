__all__ = [
    'Loss',
]
from torch import nn


class Loss(nn.Module):
    """Base class for losses"""

    def __init__(self, reduction='mean'):
        """
        Parameters
        ----------
        reduction : {'mean', 'sum'} or callable
            Reduction to apply across batch elements
        """
        super().__init__()
        self.reduction = reduction

    def reduce(self, x):
        if not self.reduction:
            return x
        if isinstance(self.reduction, str):
            if self.reduction.lower() == 'mean':
                return x.mean()
            if self.reduction.lower() == 'sum':
                return x.sum()
            raise ValueError(f'Unknown reduction "{self.reduction}"')
        if callable(self.reduction):
            return self.reduction(x)
        raise ValueError(f'Don\'t know what to do with reduction: '
                         f'{self.reduction}')
