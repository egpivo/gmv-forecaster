import pandas as pd
import torch

from forecaster.data.training_data_generator import TrainingDataGenerator
from forecaster.logger.logging import setup_logger
from forecaster.training.model.xdfm import ExtremeDeepFactorizationMachineModel
from forecaster.training.utils import EarlyStopper, test_model, train_model


class Trainer:
    _model_name = "xdfm"

    def __init__(
        self,
        processed_data: pd.DataFrame,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.1,
        device: str = "cpu",
        save_dir: str = "checkpoint/",
        epoch: int = 5,
        dropout: float = 0.2,
        num_workers: int = 8,
        model_name: str = None,
    ) -> None:
        self.processed_data = processed_data
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.epoch = epoch
        self.dropout = dropout
        self.num_workers = num_workers
        self.model_name = model_name or self._model_name

        # Set up
        self.setup_data()
        self.setup_model()
        self.logger = setup_logger()

    def setup_data(self):
        generator = TrainingDataGenerator(self.processed_data)
        self.train_loader = generator.train_loader
        self.valid_loader = generator.valid_loader
        self.test_loader = generator.test_loader

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
            auc_pr, recall = test_model(self.model, self.valid_loader, self.device)
            self.logger.info(
                f"Epoch: {epoch_i}, Validation AUCPR: {auc_pr} Validation Recall: {recall}"
            )

            if not early_stopper.is_continuable(self.model, auc_pr):
                self.logger.info(
                    f"Validation: Best Top-k AUC-PR: {early_stopper.best_accuracy}"
                )
                break

        auc = test_model(self.model, self.test_loader, self.device)
        self.logger.info(f"Test AUC: {auc}")
