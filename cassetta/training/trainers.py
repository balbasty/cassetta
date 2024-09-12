import torch
from torch import nn
from torch import optim as torch_optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional
from dataclasses import dataclass, asdict
from cassetta.io.modules import LoadableModule
from cassetta.io.utils import import_fullname, import_qualname
from cassetta import models, losses


@dataclass
class TrainerState:
    all_losses: dict = {}
    all_metrics: dict = {}
    current_losses: dict = {}
    current_metrics: dict = {}
    current_epoch: int = 0
    current_step: int = 0

    def serialize(self) -> dict:
        return asdict(self)

    def save(self, path):
        torch.save(self.serialize(), path)

    @classmethod
    def load(cls, state):
        if not isinstance(state, dict):
            state = torch.load(state)
        return cls(**state)


class Trainer(LoadableModule):

    @LoadableModule.save_args
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleDict()
        self.optimizers = {}
        self.trainer_state = TrainerState()

    def serialize(self) -> dict:
        state = super().serialize()
        state.update({
            'optimizers': [optim.state_dict() for optim in self.optimizers],
            'trainer_state': self.trainer_state.serialize(),
        })
        return state

    def load(cls, state):
        if not isinstance(state, dict):
            state = torch.load(state)
        obj = super().load(state)
        for name, optim in obj.optimizers.items():
            optim.load_state_dict(state['optimizers'][name])
        obj.trainer_state = TrainerState.load(state['trainer_state'])
        return obj

    def register_model(self, name, model: nn.Module):
        self.models[name] = model

    def register_optimizer(self, name, optim: Optimizer):
        self.optimizers[name] = optim

    def train_step(self, minibatch):
        raise NotImplementedError

    def eval_step(self, minibatch):
        raise NotImplementedError

    def log_loss(self, name, value):
        raise NotImplementedError

    def log_metric(self, name, value):
        raise NotImplementedError

    def train_epoch(self):
        for minibatch in self.iter_trainset():
            self.train_step(minibatch)
            self.update_progress_train()

    def eval_epoch(self):
        for minibatch in self.iter_evalset():
            self.eval_step(minibatch)
            self.update_progress_eval()


class BasicTrainer(Trainer):

    @Trainer.save_args
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
        self.models['model'] = models.make_model(model, **(opt_model or {}))

        # instantiate and register loss
        self.models['loss'] = losses.make_loss(loss, **(opt_loss or {}))

        # instantiate and register optim
        if isinstance(optim, str):
            if '.' not in optim:
                optim = import_qualname(torch_optim, optim)
            else:
                optim = import_fullname(optim)
            self.optimizers['model'] = (optim, lr)
