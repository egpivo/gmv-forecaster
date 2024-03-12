import logging

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from forecaster.data.data_proprocessor import DataPreprocessor
from forecaster.data.training_data_generator import TrainingDataset
from forecaster.training.model.xdfm import ExtremeDeepFactorizationMachineModel
from forecaster.training.utils import EarlyStopper, test_model, train_model


class Trainer:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.1,
        device: str = "cpu",
        save_dir: str = "checkpoint/",
        user_data_path: str = "data/users.csv",
        transaction_data_path: str = "data/transactions.csv",
        store_data_path: str = "data/stores.csv",
        epoch: int = 5,
        dropout: float = 0.2,
        num_workers: int = 8,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.user_data_path = user_data_path
        self.transaction_data_path = transaction_data_path
        self.store_data_path = store_data_path
        self.epoch = epoch
        self.dropout = dropout
        self.num_workers = num_workers
        self.model_name = "your_model_name"  # Replace with an appropriate model name

        # Set up data loaders and model
        self.setup_data()
        self.setup_model()

    def setup_data(self):
        full_data_pdf = DataPreprocessor(
            self.user_data_path, self.transaction_data_path, self.store_data_path
        ).process()
        self.dataset = TrainingDataset(full_data_pdf)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.dataset.train_indices),
            num_workers=self.num_workers,
        )
        self.valid_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.dataset.valid_indices),
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.dataset.test_indices),
            shuffle=False,
            num_workers=self.num_workers,
        )

    def setup_model(self):
        self.model = ExtremeDeepFactorizationMachineModel(
            self.dataset.field_dims,
            embed_dim=16,
            cross_layer_sizes=(16, 16),
            split_half=False,
            mlp_dims=(16, 16),
            dropout=self.dropout,
        )

    def train(self):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        early_stopper = EarlyStopper(
            num_trials=2, save_path=f"{self.save_dir}/{self.model_name}.pt"
        )

        for epoch_i in range(self.epoch):
            train_model(
                self.model, optimizer, self.train_loader, criterion, self.device
            )
            auc = test_model(self.model, self.valid_loader, self.device)
            self.logger.info(f"Epoch: {epoch_i}, Validation AUC: {auc}")

            if not early_stopper.is_continuable(self.model, auc):
                self.logger.info(f"Validation: Best AUC: {early_stopper.best_accuracy}")
                break

        auc = test_model(self.model, self.test_loader, self.device)
        self.logger.info(f"Test AUC: {auc}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
