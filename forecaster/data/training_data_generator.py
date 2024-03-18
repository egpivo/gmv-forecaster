from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from forecaster.data.data_preprocessor import CONTEXT_FIELDS, STORE_FIELDS, USER_FIELDS


class ModelDataset(Dataset):
    def __init__(
        self, full_data_pdf: pd.DataFrame, train_indices: torch.tensor
    ) -> None:
        train_data_pdf = full_data_pdf.loc[train_indices]
        selected_features = full_data_pdf[
            [*USER_FIELDS, *STORE_FIELDS, *CONTEXT_FIELDS]
        ].copy()
        self.labels = torch.tensor(full_data_pdf["label"].values, dtype=torch.long)

        # Set unseen stores' label to 0
        unseen_stores = set(train_data_pdf["store_id_label"])
        selected_features.loc[
            ~selected_features["store_id_label"].isin(unseen_stores),
            "store_id_label",
        ] = 0

        # Set unseen users' label to 0
        unseen_users = set(train_data_pdf["user_id_label"])
        selected_features.loc[
            ~selected_features["user_id_label"].isin(unseen_users),
            "user_id_label",
        ] = 0

        self.features = torch.tensor(selected_features.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.features[idx], self.labels[idx]


class TrainingDataGenerator:
    """Produce train/valid/test data loaders"""

    def __init__(
        self,
        full_data_pdf: pd.DataFrame,
        test_month: Union[pd.Timestamp, str],
        batch_size: int = 128,
        num_workers: int = 32,
    ) -> None:
        self.full_data_pdf = full_data_pdf
        self.test_month = (
            pd.to_datetime(test_month, format="%Y%m")
            if isinstance(test_month, str)
            else test_month
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Split indices for train, validation, and test sets
        (
            self.train_indices,
            self.valid_indices,
            self.test_indices,
        ) = self._split_indices()
        self.dataset = ModelDataset(self.full_data_pdf, self.train_indices)

    def _split_indices(self) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        test_upper_bound = self.test_month
        test_lower_bound = test_upper_bound - pd.DateOffset(months=1)
        validation_threshold = test_lower_bound - pd.DateOffset(months=1)
        train_lower_bound = validation_threshold - pd.DateOffset(years=1)

        # Split indices based on time
        train_indices = torch.tensor(
            self.full_data_pdf[
                (self.full_data_pdf["event_occurrence"] >= train_lower_bound)
                & (self.full_data_pdf["event_occurrence"] < validation_threshold)
            ].index,
            dtype=torch.int64,
        )
        valid_indices = torch.tensor(
            self.full_data_pdf[
                (self.full_data_pdf["event_occurrence"] >= validation_threshold)
                & (self.full_data_pdf["event_occurrence"] < test_lower_bound)
            ].index,
            dtype=torch.int64,
        )
        test_indices = torch.tensor(
            self.full_data_pdf[
                (self.full_data_pdf["event_occurrence"] >= test_lower_bound)
                & (self.full_data_pdf["event_occurrence"] < test_upper_bound)
            ].index,
            dtype=torch.int64,
        )

        return train_indices, valid_indices, test_indices

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.train_indices),
            num_workers=self.num_workers,
        )

    @property
    def valid_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.valid_indices),
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            shuffle=False,
            num_workers=self.num_workers,
        )
