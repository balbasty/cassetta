"""
## Overview

Optimizers are modules responsible for updating the model parameters to
minimize the loss function during training (specifically backprop).
Optimizers implement various optimization algorithms that dictate how the
weights of the model are adjusted based on the gradients computed during
training.

Modules
-------
base
    Base class for all optimizers and provides common functionalities.
adam
    Adaptive Moment Estimation optimizer.

Example
-------
```python
from optimizers import Adam

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# During training loop
for data, targets in dataloader:
    optimizer.zero_grad()           # Reset gradients
    outputs = model(data)           # Forward pass
    loss = loss_fn(outputs, targets) # Compute loss
    loss.backward()                 # Backward pass
    optimizer.step()                # Update parameters
"""
__all__ = ['make_optimizer']
from cassetta.io.utils import import_fullname
from cassetta.core.utils import import_submodules
from cassetta.core.typing import OptimType
from torch.optim import Optimizer

import_submodules([
    'adam',
    'base'
], __name__, __all__, True)


def make_optimizer(optim: OptimType, *args, **kwargs):
    """
    Instantiate a optimizer.

    An optimizer can be:

    - the fully qualified path to an optimizer, such as
    `"cassetta.optimizers.adam"`, or `"monai.optimizers.Novograd"`;
    - a [`optimizer`][torch.optim.optimizer] subclass, such as
    [`Adam`][cassetta.optimizers.Adam];
    - an already instantiated [`optimizer`][torch.nn.Module], such as
    [`Adam()`][cassetta.optimizers.Adam].

    Parameters
    ----------
    optim : OptimType
        Instantiated or non-instantiated optimizer
    *args : tuple
        Positional arguments pass to the optimizer constructor
    **kwargs : dict
        Keyword arguments pass to the optimizer constructor

    Returns
    -------
    optim : torch.optim.optimizer
        Instantiated optimizer
    """
    reentrant = kwargs.pop('__reentrant', False)
    if isinstance(optim, str):
        if not reentrant:
            kwargs['__reentrant'] = True
            for prefix in ('', 'cassetta.', 'cassetta.optimizers.',
                           'torch.optimizer.'):
                try:
                    return make_optimizer(prefix + optim, *args, **kwargs)
                except Exception:
                    pass
        optim = import_fullname(optim)
    if not isinstance(optim, Optimizer):
        optim = optim(*args, **kwargs)
    if not isinstance(optim, Optimizer):
        raise ValueError('Instantiated object is not an optimizer')
    return optim
