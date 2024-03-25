__all__ = [
    'make_activation',
    'SymExp',
    'SymLog',
]
from torch import nn


def make_activation(activation, **kwargs):
    """
    Instantiate an activation module.

    To be accepted in a `nn.Sequential` module or in a `nn.ModuleList`,
    an activation **must** be a `nn.Module`. This function takes other
    forms of "activation parameters" that are typically passed to the
    constructor of larger models, and generate the corresponding
    instantiated Module.

    An activation-like can be a `nn.Module` subclass, which is
    then instantiated, or a callable function that returns an
    instantiated Module.

    It is useful to accept both these cases as they allow to either:

    * have a learnable activation specific to this module
    * have a learnable activation shared with other modules
    * have a non-learnable activation

    Parameters
    ----------
    activation : ActivationType
        An already instantiated `nn.Module`, or a `nn.Module` subclass,
        or a callable that retgurns an instantiated `nn.Module`, or the
        name of an activation type from `nn`. For example:
        `"ReLU"` `"LeakyReLU"`, `"ELU"`, `"GELU"`, `"Tanh"`, etc.
    kwargs : dict
        Additional parameters to pass to the constructor or function.

    Returns
    -------
    activation : Module
        An instantiated `nn.Module`.
    """
    if not activation:
        return None

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        if hasattr(nn, activation):
            activation = getattr(nn, activation)
        elif activation in locals():
            activation = locals()[activation]
        else:
            inp_act = activation
            activation = _find_act(activation, [nn.__dict__, locals()])
            if not activation:
                raise ValueError(f'Unknown activation "{inp_act}"')

    if isinstance(activation, type):
        if not issubclass(activation, nn.Module):
            raise TypeError('Activation should be a Module subclass')
        if activation is nn.Softmax:
            kwargs.setdefault('dim', 1)
        activation = activation(**kwargs)

    elif callable(activation):
        activation = activation(**kwargs)

    if not isinstance(activation, nn.Module):
        raise ValueError('Activation did not instantiate a Module')
    return activation


def _find_act(activation, module_dict):

    if isinstance(module_dict, (list, tuple)):
        module_dicts = module_dict
        for module_dict in module_dicts:
            maybe_act = _find_act(activation, module_dict)
            if maybe_act:
                return maybe_act
        return None

    activation = activation.lower()
    for key, value in module_dict.items():
        if key.lower() == activation:
            return value
    return None


class SymExp(nn.Module):
    """
    Symmetric Exponential Activation

    ```python
    SymExp(x) = sign(x) * (exp(abs(x)) - 1)
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
    SymLog(x) = sign(x) * log(1 + abs(x))
    ```
    """

    def forward(self, x):
        sign = x.sign()
        x = x.abs().add_(1).log().mul_(sign)
        return x
