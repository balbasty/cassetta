"""
This test module verifies the basic functionalities of `BasicSupervisedTrainer`

Example
-------
>>> pytest tests/trainer.py
"""
import os
import shutil
import pytest
import torch
import torch.nn as nn
import tempfile
from torch.utils.data import Dataset
from cassetta.io.modules import make_loadable
from cassetta.models.segmentation import SegNet
from cassetta.backbones.unet import UNet
from cassetta.optimizers.adam import LoadableAdam
from cassetta.training.trainers import (TrainerConfig, BasicSupervisedTrainer)


class DummyDataset(Dataset):
    def __init__(self, size=20):
        self.size = size
        self.data = torch.randn(size, 1, 32, 32, 32)
        self.targets = torch.randn(size, 1, 32, 32, 32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def dummy_dataset():
    return DummyDataset(size=20)


@pytest.fixture
def trainer_config(temp_dir):
    return TrainerConfig(
        experiment_dir=temp_dir,
        nb_epochs=2,
        batch_size=1,
        lr=0.01,
        logging_verbosity=0,
        refresh_experiment_dir=False
    )


@pytest.fixture
def trainer(trainer_config, dummy_dataset):
    loss_fn = make_loadable(nn.MSELoss)
    loss_fn = loss_fn()
    return BasicSupervisedTrainer(
        loss=loss_fn,
        dataset=dummy_dataset,
        trainer_config=trainer_config
    )


# Test Function
def test_save_and_load_trainer(temp_dir, trainer, dummy_dataset):
    # Train for one epoch
    opt_backbone = {
        "nb_features": 8,
        "nb_levels": 2
    }
    model = SegNet(3, 1, 1, backbone='UNet', opt_backbone=opt_backbone)
    optimizer = LoadableAdam(model.parameters())

    trainer.register_model('model', model)
    trainer.register_optimizer('model', optimizer)
    trainer.train_epoch()

    # Capture original trainer state
    original_epoch = trainer.trainer_state.current_epoch
    original_step = trainer.trainer_state.current_step
    original_train_loss = trainer.trainer_state.epoch_train_loss
    original_model_state = {
        k: v.clone() for k, v in trainer.models["model"].state_dict().items()
    }

    # Save trainer
    checkpoint_path = os.path.join(temp_dir, 'trainer_checkpoint.pt')
    trainer.save(checkpoint_path)
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # Load the trainer
    loaded_trainer = BasicSupervisedTrainer.load(checkpoint_path).cpu()

    # Verify trainer state
    assert (
        loaded_trainer.trainer_state.current_epoch == original_epoch
    ), "Current epoch does not match after loading."
    assert (
        loaded_trainer.trainer_state.current_step == original_step
    ), "Current step does not match after loading."
    assert (
        pytest.approx(loaded_trainer.trainer_state.epoch_train_loss, 1e-5)
    ) == original_train_loss, "Epoch train loss does not match after loading."

    # Verify model parameters
    loaded_model_state = loaded_trainer.models["model"].state_dict()
    for key in original_model_state:
        assert torch.allclose(
            original_model_state[key],
            loaded_model_state[key],
            atol=1e-6
        ), f"Model parameter '{key}' does not match after loading."

    # TODO: Verify the rest of optimizer state.
    original_lr = trainer.optimizers["model"].param_groups[0]['lr']
    loaded_lr = loaded_trainer.optimizers["model"].param_groups[0]['lr']
    assert (
        original_lr == loaded_lr
    ), "Optimizer learning rate does not match after loading."


if __name__ == '__main__':
    pytest.main()
