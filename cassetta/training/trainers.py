import torch
from torch import nn
from inspect import signature
from torch import optim as torch_optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
from cassetta.io.utils import import_fullname, import_qualname
from cassetta import models, losses
from cassetta.io.modules import (
    LoadableModule,
    LoadableModuleDict,
    StateMixin,
    LoadableMixin,
)


@dataclass
class TrainerState(StateMixin):
    """
    Stores the state of the trainer, including metrics, losses, epoch, and
    step.
    """

    all_losses: Dict[str, Any] = None
    all_metrics: Dict[str, Any] = None
    current_losses: Dict[str, Any] = None
    current_metrics: Dict[str, Any] = None
    current_epoch: int = 0
    current_step: int = 0


class Trainer(LoadableModule):
    """
    Base class for training models with serialization and state loading.
    Handles registering models, saving entire state to a file, and loading it.
    """

    @LoadableModule.save_args
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = LoadableModuleDict()
        self.optimizers = {}
        self.trainer_state = TrainerState()

    def serialize(self) -> dict:
        state = super().serialize()

        # Serialize all models
        models_state = {
            name: model.serialize() for name, model in self.models.items()
            }

        # Serialize all optimizers
        optimizers_state = {}
        for name, optimizer in self.optimizers.items():
            optimizer_class = optimizer.__class__
            optimizer_params = self._get_optimizer_params(optimizer)
            optimizer_state_dict = optimizer.state_dict()
            optimizers_state[name] = {
                "qualname": optimizer_class.__name__,
                "module": optimizer_class.__module__,
                "kwargs": optimizer_params,
                "state": optimizer_state_dict,
            }

        # Update state with models and optimizers
        state["models"] = models_state
        state["optimizers"] = optimizers_state
        state["trainer_state"] = self.trainer_state  # .serialize()

        return state

    @classmethod
    def load(cls, state):
        if not isinstance(state, dict):
            state = torch.load(state)

        # Create an uninitialized instance of Trainer
        obj = cls.__new__(cls)

        # Initialize the instance without calling __init__
        super(Trainer, obj).__init__(
            *state.get("_args", ()), **state.get("_kwargs", {})
        )

        # Load models first
        obj.models = nn.ModuleDict()
        models_state = state.get("models", {})
        for name, model_state in models_state.items():
            model = LoadableMixin._nested_unserialize(model_state)
            obj.models[name] = model

        # Now we can load the state dict
        obj.load_state_dict(state["state"])

        # Load optimizers
        obj.optimizers = {}
        optimizers_state = state.get("optimizers", {})
        for name, optimizer_info in optimizers_state.items():
            optimizer_module_name = optimizer_info["module"]
            optimizer_class_name = optimizer_info["qualname"]
            optimizer_params = optimizer_info["kwargs"]
            optimizer_state_dict = optimizer_info["state"]

            # Dynamically import the optimizer class
            optimizer_module = __import__(
                optimizer_module_name, fromlist=[optimizer_class_name]
            )
            optimizer_class = getattr(optimizer_module, optimizer_class_name)

            # Initialize the optimizer with the model's parameters
            optimizer = optimizer_class(
                obj.models[name].parameters(), **optimizer_params
            )
            optimizer.load_state_dict(optimizer_state_dict)
            obj.optimizers[name] = optimizer

        obj.trainer_state = state.get("trainer_state", {})

        return obj

    def _get_optimizer_params(self, optimizer):
        params = {}
        sig = signature(optimizer.__class__.__init__)
        for name, param in sig.parameters.items():
            if name in ["self", "params"]:
                continue
            if hasattr(optimizer, name):
                params[name] = getattr(optimizer, name)
            elif name in optimizer.defaults:
                params[name] = optimizer.defaults[name]
        return params

    def register_model(self, name, model: nn.Module):
        """
        Register a model to the Trainer.
        """
        self.models[name] = model

    def register_optimizer(self, name, optim: torch.optim.Optimizer = None):
        """
        Register an optimizer to the trainer.
        """
        self.optimizers[name] = optim


class BasicTrainer(Trainer):

    # @Trainer.save_args
    def __init__(
        self,
        model: Union[str, nn.Module],
        loss: Union[str, nn.Module],
        optim: Union[str, Optimizer],
        trainset: Union[Dataset, DataLoader],
        evalset: Optional[Union[Dataset, DataLoader]] = None,
        nb_epochs: int = None,
        nb_steps: int = None,
        early_stopping: float = 0,
        lr=1e-4,
        *,
        opt_model: dict = None,
        opt_loss: dict = None,
    ):
        super().__init__()

        # instantiate and register model
        self.models["model"] = models.make_model(model, **(opt_model or {}))

        # instantiate and register loss
        self.models["loss"] = losses.make_loss(loss, **(opt_loss or {}))

        # instantiate and register optim
        if isinstance(optim, str):
            if "." not in optim:
                optim = import_qualname(torch_optim, optim)
            else:
                optim = import_fullname(optim)
            self.optimizers["model"] = (optim, lr)
