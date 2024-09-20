__all__ = [
    "SupervisedSanitycheckDataset"
]

import torch
from torch.utils.data import Dataset


class SupervisedSanitycheckDataset(Dataset):
    """
    Dataset for x-y paired data.
    """

    def __init__(self,
                 n_samples: str = 100,
                 x_shape: tuple = (1, 64, 64, 64),
                 y_shape: tuple = (1, 64, 64, 64),
                 n_classes: int = None,
                 device: str = 'cuda',
                 ):
        """
        Dataset for random input an.

        patch_directory : str
            Directory containing patches of data in .pt file format
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
        length : int
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
        # Sample x data and add batch dimension
        x = torch.randn(self.x_shape, device=self.device).unsqueeze(0)
        # Sample y data and add batch dimension        
        if isinstance(self.n_classes, int):
            y = torch.randint(
                high=self.n_classes,
                size=self.y_shape,
                device=self.device
                ).unsqueeze(0)
        else:
            y = torch.randn(self.y_shape, device=self.device).unsqueeze(0)
        return x.to(torch.float32), y.to(torch.float32)
