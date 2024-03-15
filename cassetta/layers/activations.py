__all__ = [
    'SymExp',
    'SymLog',
]
from torch import nn


class SymExp(nn.Module):
    """
    Symmetric Exponential Activation

    ```python
    SymExp(x) = sign(x) * (exp(|x|) - 1)
    ```
    """

    def forward(self, x):
        sign = x.sign()
        x = x.abs().exp().sub_(1).mul_(sign)
        return x


class SymLog(nn.Module):
    """
    Symmetric Logarithmic Activation

    ```python
    SymLog(x) = sign(x) * log(1 + |x|)
    ```
    """

    def forward(self, x):
        sign = x.sign()
        x = x.abs().add_(1).log().mul_(sign)
        return x
