import logging
from typing import Optional

import pandas as pd
import torch

from forecaster.data.data_proprocessor import DataPreprocessor
from forecaster.inference.utils import test_model
from forecaster.training.trainer import Trainer


class RollingWindowTrainer:
    def __init__(
        self,
        start_month: str,
        end_month: str,
        user_data_path: str,
        transaction_data_path: str,
        store_data_path: str,
        embed_dim: int,
        learning_rate: float,
        batch_size: int,
        weight_decay: float,
        save_dir: str,
        epoch: int,
        dropout: float,
        num_workers: int,
        model_name: str,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ):
        self.start_month = start_month
        self.end_month = end_month
        self.user_data_path = user_data_path
        self.transaction_data_path = transaction_data_path
        self.store_data_path = store_data_path
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.epoch = epoch
        self.dropout = dropout
        self.num_workers = num_workers
        self.model_name = model_name
        self.device = device
        self.logger = logger

    def train(self):
        for month in range(int(self.start_month), int(self.end_month) + 1):
            test_month = pd.to_datetime(str(month), format="%Y%m")
            train_lower_bound = test_month - pd.DateOffset(years=1, month=1)

            processor = DataPreprocessor(
                user_data_path=self.user_data_path,
                transaction_data_path=self.transaction_data_path,
                store_data_path=self.store_data_path,
                start_month=train_lower_bound,
                end_month=test_month,
            )

            # Train the model
            trainer = Trainer(
                processed_data=processor.process(),
                test_month=test_month,
                embed_dim=self.embed_dim,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                save_dir=self.save_dir,
                epoch=self.epoch,
                dropout=self.dropout,
                num_workers=self.num_workers,
                model_name=self.model_name,
                device=self.device,
                logger=self.logger,
            )
            trainer.train()

            # Evaluate the model
            test_recall_at_k = test_model(
                trainer.model, trainer.test_loader, self.device
            )
            self.logger.info(f"Test Recall for month {month}: {test_recall_at_k}")
