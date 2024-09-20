import torch
from torch import nn
from inspect import signature
from torch import optim as torch_optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass
from cassetta.core.utils import refresh_experiment_dir
from cassetta.io.utils import import_fullname, import_qualname
from cassetta import models, losses
from cassetta.io.modules import (
    LoadableModule,
    LoadableModuleDict,
    StateMixin,
    LoadableMixin,
)


@dataclass
class TrainerConfig(StateMixin):
    """
    State configuration of the training process.

    Attributes
    ----------
    experiment_dir : str
        The filesystem path where model checkpoints and logs will be saved.
    nb_epochs : int, optional
        The number of epochs to train the model. Default is 100.
    batch_size : int, optional
        The number of samples per training batch. Default is 1.
    lr : float, optional
        The learning rate for the optimizer. Default is 1e-4.
    logging : bool, optional
        Optionally enable logging during training. Default is True.
    early_stopping : float
    refresh_experiment_dir : bool
        Delete contents of `experiment_dir` when run starts.
    """
    experiment_dir: str
    nb_epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-4
    logging: bool = True
    early_stopping: float = 0
    refresh_experiment_dir: bool = False


@dataclass
class TrainerState(StateMixin):
    """
    Tracks the current state of the training process.

    Attributes
    ----------
    current_epoch : int, optional
        The current epoch number being processed. Default is 0.
    current_step : int, optional
        The current step number within the current epoch. Default is 0.
    epoch_train_loss : float, optional
        The cumulative training loss for the current epoch. Default is 0.0.
    epoch_eval_loss : float, optional
        The cumulative evaluation loss for the current epoch. Default is 0.0.
    best_eval_loss : None, optional
        The best evaluation loss for the entire training process.
        Default is None
    """
    current_epoch: int = 0
    current_step: int = 0
    epoch_train_loss: float = 0.0
    epoch_eval_loss: float = 0.0
    best_eval_loss: float = 100  # Arbitrary


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
        optimizers_state = {
            name: opt.serialize() for name, opt in self.optimizers.items()
        }

        # Update state with models and optimizers
        state["models"] = models_state
        state["optimizers"] = optimizers_state
        state["trainer_state"] = self.trainer_state

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
        for name, optimizer_state in optimizers_state.items():
            optimizer = LoadableMixin._nested_unserialize(optimizer_state)
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


class BasicSupervisedTrainer(Trainer):

    # @Trainer.save_args
    def __init__(
        self,
        loss: Union[str, nn.Module],
        trainset: Union[Dataset, DataLoader],
        evalset: Optional[Union[Dataset, DataLoader]] = None,
        trainer_config: TrainerConfig = None,
        *,
        opt_model: dict = None,
        opt_loss: dict = None,
    ):
        super().__init__()
        self.loss = loss
        self.trainset = trainset
        self.evalset = evalset
        self.trainer_config = trainer_config
        self.trainer_state = TrainerState()
        trainer_config.save_state_dict(
            f'{trainer_config.experiment_dir}/trainer_config.yaml'
            )
        if self.trainer_config.logging:
            self.writer = SummaryWriter(self.trainer_config.experiment_dir)

    def train_step(self, minibatch):
        # Set model to train mode
        self.models["model"].train()
        # Unpack minibatch
        x, y = minibatch
        # Zero optimizer gradients
        self.optimizers["model"].zero_grad()
        # Forward pass
        outputs = self.models["model"](x)
        # Calculate loss
        _loss = self.loss(y, outputs)
        # Update trainer state
        self.trainer_state.epoch_train_loss += _loss.item()
        # Optionally log.
        # TODO: incorporate logger
        if self.trainer_config.logging:
            self.log_metric('train', _loss.item())
        # Backward pass
        _loss.backward()
        # Step optimizer
        self.optimizers["model"].step()
        # Increment current step
        self.trainer_state.current_step += 1

    def eval_step(self, minibatch):
        # Do not keep track of gradients
        with torch.no_grad():
            # Set model to eval mode
            self.models["model"].eval()
            # Unpack minibatch
            x, y = minibatch
            # Forward pass
            outputs = self.models["model"](x)
            # Calculate loss
            _loss = self.loss(y, outputs)
            self.trainer_state.epoch_eval_loss += _loss.item()

    def log_loss(self, name, value):
        raise NotImplementedError

    def log_metric(self, phase, scalar_value):
        self.writer.add_scalar(
            tag=f'{phase}_loss',
            scalar_value=scalar_value,
            global_step=self.trainer_state.current_step
            )

    def train_epoch(self):
        # For sanity check dataset, must load minibatches like this or else
        # it will go to infinity.
        self.trainer_state.epoch_train_loss = 0
        for n in range(len(self.trainset)):
            self.train_step(self.trainset[n])
        # Average train epoch loss
        self.trainer_state.epoch_train_loss /= len(self.trainset)
        if self.trainer_config.logging:
            self.log_metric('train_epoch', self.trainer_state.epoch_train_loss)
            self.log_parameter_hist()

    def eval_epoch(self):
        # Reset eval loss
        self.trainer_state.epoch_eval_loss = 0
        # Iterate through eval set
        for n in range(len(self.evalset)):
            self.eval_step(self.evalset[n])
        # Average eval epoch loss
        self.trainer_state.epoch_eval_loss /= len(self.trainset)
        # Optionally log to tensorboard
        if self.trainer_config.logging:
            self.log_metric('eval_epoch', self.trainer_state.epoch_eval_loss)
        # If this is best checkpoint...
        if self.trainer_state.epoch_eval_loss < (
            self.trainer_state.best_eval_loss
        ):
            self.trainer_state.best_eval_loss = (
                self.trainer_state.epoch_eval_loss
                )
            # TODO: self.save_checkpoint()

    def train(self):
        if self.trainer_config.logging:
            self.log_model_graph()
        if self.trainer_config.refresh_experiment_dir:
            refresh_experiment_dir(self.trainer_config.experiment_dir)
        for i in range(self.trainer_config.nb_epochs):
            self.train_epoch()
            if self.evalset:
                self.eval_epoch()
            # Increment current epoch
            self.trainer_state.current_epoch += 1

    def register_model(self, name, model):
        super().register_model(name, model)
        if self.trainer_config.logging:
            self.log_model_graph()

    def log_model_graph(self) -> SummaryWriter:
        """
        Logs the model graph to TensorBoard.

        Returns
        -------
        SummaryWriter
            TensorBoard SummaryWriter object.
        """
        # Initialize writer
        sample_inputs, _ = self.trainset[0]
        model = self.models['model']
        self.writer.add_graph(model, sample_inputs)

    def log_parameter_hist(self) -> None:
        """
        Log histograms of model parameters and gradients for TensorBoard.
        """
        for name, param in self.models["model"].named_parameters():
            self.writer.add_histogram(
                name,
                param,
                self.trainer_state.current_epoch
                )
            if param.grad is not None:
                self.writer.add_histogram(
                    tag=f'{name}.grad',
                    values=param.grad,
                    global_step=self.trainer_state.current_epoch
                    )
