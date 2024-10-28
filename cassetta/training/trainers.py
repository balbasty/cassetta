import torch
import importlib
from torch import nn
from inspect import signature
from torch import optim as torch_optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional
from dataclasses import dataclass
from cassetta.io.utils import import_fullname, import_qualname
from cassetta import models, losses
from cassetta.core.utils import (
    refresh_experiment_dir,
    delete_files_with_pattern,
    )
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
    logging_verbosity : int, optional
        Optionally log training progress with a specified verbosity {0, 1, 2}.
        Default is 1.\n
        Options are:
            - 0: No logging to tensorboard.
            - 1: Basic logging to tensorboard (eval and train loss).
            - 2: Detailed logging to tensorboard, eval loss, training loss,
                parameter histograms, and model graph.
    early_stopping : float
    refresh_experiment_dir : bool
        Delete contents of `experiment_dir` when run starts.
    """
    experiment_dir: str
    nb_epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-4
    logging_verbosity: int = 1
    early_stopping: float = 0
    refresh_experiment_dir: bool = False
    train_to_val: float = 0.8
    num_workers: int = 0


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
        Default is inf.
    """
    current_epoch: int = 0
    current_step: int = 0
    epoch_train_loss: float = 0.0
    epoch_eval_loss: float = 0.0
    best_eval_loss: float = float('inf')


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

        optimizers_state = {
            name: optimizer.serialize() for name,
            optimizer in self.optimizers.items()
        }

        state["models"] = models_state
        state["optimizers"] = optimizers_state
        state["trainer_state"] = self.trainer_state

        return state

    @classmethod
    def load(cls, state):
        # TODO: Set option to load trainer in fine-tuning mode. This will:
        # 1. Set trainer state eval loss back to negative infinity
        if not isinstance(state, dict):
            state = torch.load(state)

        # Create an uninitialized instance of Trainer
        obj = cls.__new__(cls)

        # Initialize the instance without calling __init__
        super(Trainer, obj).__init__(
            *state.get("_args", ()), **state.get("_kwargs", {})
        )

        # obj = super().load(state)

        # Load models first
        obj.models = nn.ModuleDict()
        models_state = state.get("models", {})
        for name, model_state in models_state.items():
            model = LoadableMixin._nested_unserialize(model_state)
            # TODO: Make something to put model on device
            obj.models[name] = model.cuda()

        # Now we can load the state dict
        obj.load_state_dict(state["state"])

        obj.optimizers = {}
        optimizers_state = state.get("optimizers", {})
        for name, optimizer_state in optimizers_state.items():
            # Deserialize optimizer class using 'module' and 'qualname'
            module_path = optimizer_state['module']
            class_name = optimizer_state['qualname']
            optimizer_class = getattr(
                importlib.import_module(module_path), class_name
                )

            # Get optimizer args and kwargs
            args = optimizer_state.get('args', ())
            kwargs = optimizer_state.get('kwargs', {})

            # Initialize optimizer with model's parameters and saved
            # args and kwargs
            optimizer = optimizer_class(
                obj.models[name].parameters(), *args, **kwargs
                )

            # Load optimizer's state_dict
            optimizer.load_state_dict(optimizer_state["state_dict"])

            # Register optimizer
            obj.optimizers[name] = optimizer

        obj.trainer_state = state.get('trainer_state')
        # Resetting the best eval loss so fine tuning isn't expected to perform
        # as well.
        obj.trainer_state.best_eval_loss = float('inf')

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
        dataset: Union[Dataset, DataLoader],
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

    @Trainer.save_args
    def __init__(
        self,
        loss: Union[str, nn.Module],
        dataset: Union[Dataset, DataLoader],
        evalset: Optional[Union[Dataset, DataLoader]] = None,
        trainer_config: TrainerConfig = None,
        *,
        opt_model: dict = None,
        opt_loss: dict = None,
    ):
        super().__init__()
        self.loss = loss
        self.dataset = dataset
        self.trainer_config = trainer_config
        self.get_loaders(self.dataset)
        if self.trainer_config.logging_verbosity >= 1:
            self.writer = SummaryWriter(self.trainer_config.experiment_dir)

    def get_loaders(self, dataset):
        seed = torch.Generator().manual_seed(42)
        train_set_size = round(len(dataset) * self.trainer_config.train_to_val)
        val_set_size = len(dataset) - train_set_size

        train_set, eval_set = random_split(
            dataset,
            [
                train_set_size,
                val_set_size
            ],
            seed
        )
        self.train_loader = DataLoader(
                    dataset=train_set,
                    batch_size=self.trainer_config.batch_size,
                    shuffle=True,
                    num_workers=self.trainer_config.num_workers
                    )
        self.eval_loader = DataLoader(
            dataset=eval_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.trainer_config.num_workers
        )

    def train_step(self, minibatch):
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
        if self.trainer_config.logging_verbosity >= 1:
            self.log_metric('train', _loss.item(), 'step')
        # Backward pass
        _loss.backward()
        # Step optimizer
        self.optimizers["model"].step()
        # Increment current step
        self.trainer_state.current_step += 1

    def eval_step(self, minibatch):
        # Do not keep track of gradients
        with torch.no_grad():
            # Unpack minibatch
            x, y = minibatch
            # Forward pass
            outputs = self.models["model"](x)
            # Calculate los`s
            _loss = self.loss(y, outputs)
            self.trainer_state.epoch_eval_loss += _loss.item()

    def log_loss(self, name, value):
        raise NotImplementedError

    def log_metric(self, phase, scalar_value, timestep):
        if timestep == 'epoch':
            timestep = self.trainer_state.current_epoch
        elif timestep == 'step':
            timestep = self.trainer_state.current_step
        else:
            raise f'Invalid timestep {timestep}. Must be "step" or "epoch".'

        self.writer.add_scalar(
            tag=f'{phase}_loss',
            scalar_value=scalar_value,
            global_step=timestep
            )

    def train_epoch(self):
        # Set model to train mode
        self.models["model"].train()
        # For sanity check dataset, must load minibatches like this or else
        # it will go to infinity.
        self.trainer_state.epoch_train_loss = 0
        for minibatch in self.train_loader:
            self.train_step(minibatch)
        # Average train epoch loss
        self.trainer_state.epoch_train_loss /= len(self.train_loader)
        if self.trainer_config.logging_verbosity >= 1:
            self.log_metric(
                'train_epoch',
                self.trainer_state.epoch_train_loss,
                'epoch')
        if self.trainer_config.logging_verbosity >= 2:
            self.log_parameter_hist()

    def eval_epoch(self):
        # Set model to eval mode
        self.models["model"].eval()
        # Reset eval loss
        self.trainer_state.epoch_eval_loss = 0
        # Iterate through eval set
        for minibatch in self.eval_loader:
            self.eval_step(minibatch)
        # Average eval epoch loss
        self.trainer_state.epoch_eval_loss /= len(self.eval_loader)
        # Optionally log to tensorboard
        if self.trainer_config.logging_verbosity >= 1:
            self.log_metric(
                'eval_epoch',
                self.trainer_state.epoch_eval_loss,
                'epoch')
        # If this is best checkpoint...
        if self.trainer_state.epoch_eval_loss < (
            self.trainer_state.best_eval_loss
                ):

            self.trainer_state.best_eval_loss = (
                self.trainer_state.epoch_eval_loss
                )

            self.save_checkpoint('best')
        self.save_checkpoint('last')

    def train(self):
        if self.trainer_config.logging_verbosity >= 2:
            self.log_model_graph()
        if self.trainer_config.refresh_experiment_dir:
            refresh_experiment_dir(self.trainer_config.experiment_dir)
        for i in range(self.trainer_config.nb_epochs):
            self.train_epoch()
            if self.eval_loader:
                self.eval_epoch()
            # Increment current epoch
            self.trainer_state.current_epoch += 1

    def register_model(self, name, model):
        super().register_model(name, model)
        if self.trainer_config.logging_verbosity >= 2:
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
        sample_inputs, _ = next(iter(self.train_loader))
        model = self.models['model']
        # print(sample_inputs.shape)
        self.writer.add_graph(model, sample_inputs)

    def log_parameter_hist(self) -> None:
        """
        Log histograms of model parameters and gradients for TensorBoard.
        """
        for name, param in self.models["model"].named_parameters():
            self.writer.add_histogram(
                name,
                param.to(torch.uint32),
                self.trainer_state.current_epoch
                )
            if param.grad is not None:
                self.writer.add_histogram(
                    tag=f'{name}.grad',
                    values=param.grad,
                    global_step=self.trainer_state.current_epoch
                )

    def save_checkpoint(self, type: str = 'last'):
        """
        Save a checkpoint.

        type : str
            Type of checkpoint to save {`last`, `best`}
        """
        checkpoint_dir = f'{self.trainer_config.experiment_dir}/checkpoints'
        if type == 'last':
            delete_files_with_pattern(checkpoint_dir, '*last*')
            self.save(
                f'{checkpoint_dir}/last-{self.trainer_state.current_epoch}.pt'
                )
        if type == 'best':
            delete_files_with_pattern(checkpoint_dir, '*best*')
            self.save(
                f'{checkpoint_dir}/best-{self.trainer_state.current_epoch}.pt'
                )

    def serialize(self) -> dict:
        """
        Override the serialize method to exclude the 'dataset' attribute.
        """
        state = super().serialize()

        # Exclude 'dataset' by setting it to an empty dictionary
        state['kwargs']['dataset'] = {}
        return state

    # @classmethod
    #def load(experiment_dir: str, type: str = 'last', verbose: bool = False):
        #"""
        #experiment_dir : str
        #    Directory of experiment
        #type : str
        #    Checkpoint to load {'best', 'last'}
        #verbose : bool
        #    Loading is verbose
        #"""
        #experiment_dir += '/checkpoints'
        #if type not in ['best', 'last']:
        #    raise ValueError("Type must be either 'best' or 'last'.")

        #if type == 'last':
        #    hits = find_files_with_pattern(experiment_dir, '*last*')
        #elif type == 'best':
        #    hits = find_files_with_pattern(experiment_dir, '*best*')
        #checkpoint = hits[0]
        #if verbose:
        #    print(f'Loading checkpoint: {checkpoint}')

        #return super().load(checkpoint)
