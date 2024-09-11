__all__ = ['Trainer']
import torch
from torch import nn
from typing import Optional
from torch.utils.data import DataLoader
from cassetta.training.loggers import Logger


class SimpleSupervisedTrainer:
    """
    A generic training loop that works with any model.

    Parameters
    ----------
    model : nn.Sequential
        The model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim
        Optimizer for updating model weights.
    device : str
        Device to run the training on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        model_dir: str,
        model: nn.Sequential,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim,
        device: Optional[str] = 'cuda'
    ):
        """
        A generic training loop that works with any model.

        Parameters
        ----------
        model : nn.Sequential
            The model to be trained.
        train_loader : DataLoader
            DataLoader for the training dataset.
        val_loader : DataLoader
            DataLoader for the validation dataset.
        criterion : nn.Module
            Loss function.
        optimizer : torch.optim
            Optimizer for updating model weights.
        device : str
            Device to run the training on. ('cpu' or 'cuda').
        """
        self.model_dir = model_dir
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        # Initialize logger
        self.logger = Logger(
            model_dir=self.model_dir,
            model=self.model,
            train_loader=self.train_loader
        )
        # Initializing states of tracking attributes
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = 1

    def train(self, num_epochs: int = 100):
        """
        Train the model over num_epochs epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model.
        """
        # Refreshing output directories (making them if they don't exist)
        # TODO: Add something to refresh model saving directory.
        # Log model's graph
        self.logger.log_model_graph()
        # Iterate across epochs
        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}")
            if self.current_epoch % 5 == 0:
                # Log parameter and gradient frequency distributions
                self.logger.log_parameter_histograms(self.current_epoch)
            # Increment epoch
            self.current_epoch += 1

    def _train_one_epoch(self):
        """
        Perform a single epoch of training.

        Parameters
        ----------
        epoch : int
            Curent epoch.
        num_epochs : int
            Total number of epochs.

        Returns
        -------
        epoch_train_loss : float
            Average loss over epoch.
        """
        self.model.train()  # Set the model to training mode
        epoch_train_loss = 0.0  # Init running training loss
        for i, (x0, zt) in enumerate(self.train_loader):
            x0, zt = x0.to(self.device)[0], zt.to(self.device)[0]
            self.optimizer.zero_grad()  # Zero parameter gradients

            # Forward pass
            outputs = self.model(zt)  # Forward pass through model
            loss = self.criterion(x0, outputs)  # Calculate loss

            # Backward pass
            loss.backward()  # Backprop
            self.optimizer.step()  # Step the optimizer

            epoch_train_loss += loss.item()  # Add loss to running train loss
            # Log instantaneous training metrics every 10 steps
            if i % 10 == 0:
                # Log instantaneous training metrics
                # TODO: add lr scheduler logging
                self.logger.log_metrics(
                    phase='intermediate_train',
                    # calling epoch the current step so we can track loss
                    # within training loop
                    epoch=self.current_step,
                    metrics={'loss': loss.item()})

            self.current_step += 1  # Increment step

        # Calculate and log average metrics over epoch
        epoch_train_loss /= len(self.train_loader)
        self.logger.log_metrics(
                    phase='train',
                    epoch=self.current_epoch,
                    metrics={'loss': epoch_train_loss},
                    )

        return epoch_train_loss

    def _validate(self):
        """
        Evaluate model on validation set.

        Returns
        -------
        val_loss : float
            Average loss for validation loop.
        """
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Init running val loss

        # Make predictions without tracking gradients (faster!)
        with torch.no_grad():
            for x0, labels in self.val_loader:
                x0, labels = x0.to(self.device)[0], labels.to(self.device)[0]

                outputs = self.model(labels)  # Forward pass through model
                loss = self.criterion(x0, outputs)  # Calculate loss
                val_loss += loss.item()  # Add loss to running val loss

        val_loss /= len(self.val_loader)
        # Logging validation loss to tensorboard
        self.logger.log_metrics(
            phase='test',
            epoch=self.current_epoch,
            metrics={'loss': val_loss}
        )
        # TODO: Add checkpoint/model saving logic for compatibility with
        # LoadableMixin compatibility.
        return val_loss
