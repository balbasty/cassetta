__all__ = ['Logger']
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class Logger(object):
    """
    A logger for training insights using tensorboard.

    Parameters
    ----------
    model_dir : str
        Directory where model is stored and logger will write.
    model : nn.Sequential
        The model architecture for logging.
    train_loader : DataLoader
        The loader for training data to sample pass through network.
    """
    def __init__(
        self,
        model_dir: str,
        model: nn.Sequential,
        train_loader: DataLoader
    ):
        """
        A logger for training insights using tensorboard.

        Parameters
        ----------
        model_dir : str
            Directory where model is stored and logger will write.
        model : nn.Sequential
            The model architecture for logging.
        train_loader : DataLoader
            The loader for training data to sample pass through network.
        """
        self.model_dir = model_dir
        self.model = model
        self.train_loader = train_loader
        self.writer = SummaryWriter(model_dir)

    def log_model_graph(self):
        """
        Log the graph of the model by passing a tensor through with tracking.
        """
        sample_inputs, _ = next(iter(self.train_loader))
        sample_inputs = sample_inputs[0]
        self.writer.add_graph(
            self.model,
            sample_inputs.to(next(self.model.parameters()).device)
        )

    def log_parameter_histograms(self, epoch: int):
        """
        Log the frequency histograms for each group of learnable parameters
        (grouped by name) of the model.

        Parameters
        ----------
        epoch : int
            Current epoch associated with state of parameters.
        """
        # Iterate through all named parameter groups
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, epoch)
            # TODO: Should we just log all?
            if param.grad is not None:
                self.writer.add_histogram(f'{name}.grad', param.grad, epoch)

    def log_metrics(self, phase: str, epoch: int, metrics: dict):
        """
        Log the model performance metrics based on the current training phase
        and the step in training.

        Parameters
        ----------
        phase : str
            Phase of training. {'train', 'test'}
        epoch : int
            Current epoch associated with metrics to be logged.
        metrics : dict
            Dictionary of metrics to log. Ex. {'loss': running_loss}
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}_{key}', value, epoch)
