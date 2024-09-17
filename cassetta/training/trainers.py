import os
import torch
from torch import nn
from torch import optim as torch_optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass, asdict
from cassetta.io.utils import import_fullname, import_qualname
from cassetta import models, losses
from cassetta.io.modules import LoadableModuleDict, StateMixin


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


class Trainer(nn.Module):
    """
    Base class for training models with serialization and state loading.
    Handles registering models, saving entire state to a file, and loading it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = LoadableModuleDict()
        self.optimizers = {}
        self.trainer_state = TrainerState()

    def register_model(
        self, name, model: nn.Module, optimizer: torch.optim.Optimizer = None
    ):
        """
        Register a model to the Trainer, along with its optimizer.
        """
        self.models[name] = model
        if optimizer:
            self.optimizers[name] = optimizer

    def save(self, file_path):
        """
        Save the Trainer state, including all models, optimizers, and trainer
        state.
        """
        checkpoint = {
            "models": {
                name: model.state_dict() for name, model in self.models.items()
                },
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            "trainer_state": self.trainer_state.__dict__,
            "model_classes": {
                name: model.__class__ for name, model in self.models.items()
            },
            "model_init_args": {
                name: model._init_args for name, model in self.models.items()
            },  # Save model initialization arguments
        }
        torch.save(checkpoint, file_path)
        print(f"Trainer state saved to {file_path}")

    def load(self, file_path):
        """
        Load the Trainer state from a file, restoring models, optimizers,
        and trainer state.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved trainer found at {file_path}")

        checkpoint = torch.load(file_path)

        # Re-create models from saved classes and their init args
        for name, model_class in checkpoint["model_classes"].items():
            # Load model initialization arguments
            init_args = checkpoint["model_init_args"][name]

            if name not in self.models:
                # Dynamically create the model using the saved init args
                model = model_class(**init_args)
                model.load_state_dict(checkpoint["models"][name])
                self.models[name] = model
            else:
                self.models[name].load_state_dict(checkpoint["models"][name])

            if name not in self.optimizers:
                # Adjust optimizer as needed
                optimizer = torch.optim.Adam(self.models[name].parameters())
                optimizer.load_state_dict(checkpoint["optimizers"][name])
                self.optimizers[name] = optimizer
            else:
                self.optimizers[name].load_state_dict(
                    checkpoint["optimizers"][name]
                    )

        # Restore trainer state
        self.trainer_state.__dict__.update(checkpoint["trainer_state"])
        print(f"Trainer state loaded from {file_path}")
        return self


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
