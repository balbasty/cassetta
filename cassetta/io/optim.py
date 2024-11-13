__all__ = ["LoadableOptimizer"]
# externals
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

# internals
from cassetta.io.loadable import LoadableMixin


class LoadableOptimizer(LoadableMixin, Optimizer):
    """
    A loadable variant of [`optim.Optimizer`][torch.optim.Optimizer]

    This is a loadable mixin for optimizers **without** saving model params.
    """

    def serialize(self) -> dict:
        # Serialize as normal
        serialized_state = super().serialize()
        # Gather state dict as standard from pytorch optimizer
        serialized_state["state"] = self.state_dict()
        # Gather args and kwargs (to be manipulated)
        args = serialized_state.get("args", tuple())
        kwargs = serialized_state.get("kwargs", dict())
        # Remove params from args if present
        if args and isinstance(args[0], (Parameter, Tensor)):
            args = [[]] + list(args[1:])
        # Replace args and kwargs
        serialized_state["args"] = args
        serialized_state["kwargs"] = kwargs

        return serialized_state
