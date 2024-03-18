import logging
import os
from typing import Optional, Union

import pandas as pd
import torch

from forecaster.data.training_data_generator import TrainingDataGenerator
from forecaster.evaluation.utils import validate_model
from forecaster.training.early_stopper import EarlyStopper
from forecaster.training.model.xdfm import ExtremeDeepFactorizationMachineModel
from forecaster.training.utils import train_model


class Trainer:
    def __init__(
        self,
        processed_data: pd.DataFrame,
        field_dims: pd.Series,
        test_month: pd.Timestamp,
        embed_dim: int = 16,
        cross_layer_sizes: tuple[int, int] = (16, 16),
        mlp_dims: tuple[int, int] = (16, 16),
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        weight_decay: float = 0.1,
        device: Union[str, torch.device] = "cpu",
        save_dir: str = "checkpoint",
        epoch: int = 5,
        dropout: float = 0.2,
        num_workers: int = 8,
        model_name: str = "xdfm",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the Trainer.

        Parameters
        ----------
        processed_data : pd.DataFrame
            Processed data for training.
        field_dims : pd.Series
            Field dimensions for the model.
        test_month : pd.Timestamp
            Test month for evaluation.
        embed_dim : int, optional
            Embedding dimension, by default 16.
        cross_layer_sizes : tuple[int, int], optional
            Sizes of cross layers, by default (16, 16).
        mlp_dims : tuple[int, int], optional
            Dimensions of MLP layers, by default (16, 16).
        learning_rate : float, optional
            Learning rate for optimization, by default 1e-3.
        batch_size : int, optional
            Batch size for training, by default 128.
        weight_decay : float, optional
            Weight decay for optimization, by default 0.1.
        device : Union[str, torch.device], optional
            Device for training, by default 'cpu'.
        save_dir : str, optional
            Directory to save model, by default 'checkpoint'.
        epoch : int, optional
            Number of epochs, by default 5.
        dropout : float, optional
            Dropout rate, by default 0.2.
        num_workers : int, optional
            Number of workers for data loading, by default 8.
        model_name : str, optional
            Name of the model, by default 'xdfm'.
        logger : Optional[logging.Logger], optional
            Logger object, by default None.
        """
        self.field_dims = field_dims
        self.generator = TrainingDataGenerator(
            processed_data,
            test_month=test_month,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.embed_dim = embed_dim
        self.cross_layer_sizes = cross_layer_sizes
        self.mlp_dims = mlp_dims
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model_path = os.path.join(save_dir, f"{model_name}.pt")
        self.epoch = epoch
        self.dropout = dropout
        self.num_workers = num_workers
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

        # Set up
        self.setup_data_loaders()
        self.model = self.setup_model()

    def setup_data_loaders(self) -> None:
        """
        Set up data loaders.
        """
        self.train_loader = self.generator.train_loader
        self.valid_loader = self.generator.valid_loader
        self.test_loader = self.generator.test_loader

    def setup_model(self) -> ExtremeDeepFactorizationMachineModel:
        """
        Set up the model.

        Returns
        -------
        ExtremeDeepFactorizationMachineModel
            Initialized model.
        """
        if os.path.exists(self.model_path):
            self.logger.info(f"Load the existing model from {self.model_path}")
            return torch.load(self.model_path)
        else:
            return ExtremeDeepFactorizationMachineModel(
                field_dims=self.field_dims,
                embed_dim=self.embed_dim,
                cross_layer_sizes=self.cross_layer_sizes,
                mlp_dims=self.mlp_dims,
                dropout=self.dropout,
            )

    def train(self) -> None:
        """
        Train the model.
        """
        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        early_stopper = EarlyStopper(num_trials=self.epoch, save_path=self.model_path)

        for epoch_i in range(self.epoch):
            train_model(self.model, optimizer, self.train_loader, loss, self.device)
            if len(self.valid_loader) > 0:
                auroc = validate_model(self.model, self.valid_loader, self.device)
                self.logger.info(f"Epoch: {epoch_i}, Validation AUC: {auroc}")

                if not early_stopper.dose_continue_training(self.model, auroc):
                    self.logger.info(
                        f"Validation: Best AUC: {early_stopper.best_metric}"
                    )
                    break
