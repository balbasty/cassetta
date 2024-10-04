__all__ = [
    "DummySupervisedDataset"
]

import torch
from torch.utils.data import Dataset


class DummySupervisedDataset(Dataset):
    """
    A synthetic dataset for supervised learning with paired inputs and targets.

    This dataset generates random tensors to simulate x-y paired data, which
    can be used for debugging/training supervised learning models and
    pipelines for regression and classification.
    """

    def __init__(
        self,
        n_samples: int = 100,
        x_shape: tuple = (1, 64, 64, 64),
        y_shape: tuple = (1, 64, 64, 64),
        n_classes: int = None,
        device: str = 'cuda',
    ):
        """
        Initializes the DummySupervisedDataset.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples (input-target pairs) in the
            dataset (default is 100).
        x_shape : tuple, optional
            Shape of the input tensor (default is (1, 64, 64, 64)).
        y_shape : tuple, optional
            Shape of the target tensor (default is (1, 64, 64, 64)).
        n_classes : int, optional
            Number of classes for classification tasks. If `None`, targets are
            continuous tensors for regression tasks (default is None).
        device : str, optional
            Device to allocate data {'cuda', 'cpu'}
            (default is 'cuda').
        """
        self.n_samples = n_samples
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.n_classes = n_classes
        self.device = device

    def __len__(self):
        """
        Returns length of patches dataset.

        Returns
        -------
        int
            Length of random dataset (x-y pairs).
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get patch and transformed patch.

        Returns
        -------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Target tensor.
        """
        # Generate random input tensor
        x = torch.randn(self.x_shape, device=self.device).unsqueeze(0)
        # Generate random target tensor based on task (classification vs reg)
        if isinstance(self.n_classes, int):
            # Classification: integer target class labels
            y = torch.randint(
                high=self.n_classes,
                size=self.y_shape,
                device=self.device
                ).unsqueeze(0)
        else:
            # Regression: continuious targets
            y = torch.randn(self.y_shape, device=self.device).unsqueeze(0)
        return x.to(torch.float32), y.to(torch.float32)
