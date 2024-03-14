import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class ModelDataset(Dataset):
    """
    Examples
    --------
    >>> import torch
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> from forecaster.data.training_data_generator import ModelDataset
    >>> processor = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv")
    >>> full_data_pdf = processor.process()
    >>> field_dims = processor.field_dims
    >>> dataset = ModelDataset(full_data_pdf, field_dims)
    >>> next(iter(dataset))
    (tensor([6.1000e+02, 1.0000e+00, 5.1000e+01, 8.8546e+04, 3.0000e+00, 1.0500e+02,
        8.0000e+00, 3.5647e+01, 1.4004e+02, 0.0000e+00, 1.0000e+00, 2.0000e+00],
       dtype=torch.float64), tensor(1))
    """

    _removed_id = (
        "user_id",
        "store_id",
        "gender",
        "nam",
        "laa",
        "category",
        "amount",
        "event_occurrence",
    )

    def __init__(self, full_data_pdf: pd.DataFrame, field_dims: list[int]) -> None:
        selected_features = full_data_pdf.drop([*self._removed_id, "label"], axis=1)
        self.features = torch.tensor(selected_features.values)
        self.field_dims = field_dims
        self.labels = torch.tensor(full_data_pdf["label"].values)

    def __len__(self):
        # The total length is the sum of all samples
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.features[idx], self.labels[idx].to(torch.long)


class TrainingDataGenerator:
    """Produce train/valid/test data loaders
    Examples
    --------
    >>> import torch
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> from forecaster.data.training_data_generator import TrainingDataGenerator
    >>> processor = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv")
    >>> full_data_pdf = processor.process()
    >>> field_dims = processor.field_dims
    >>> generator = TrainingDataGenerator(full_data_pdf, field_dims, batch_size=1))
    >>> next(iter(generator.train_loader))
    [tensor([[1.3640e+03, 1.0000e+00, 4.4000e+01, 1.9645e+04, 1.8000e+01, 7.7900e+02,
             7.0000e+00, 3.5289e+01, 1.3913e+02, 1.0000e+00, 1.0000e+00, 2.0000e+00]],
           dtype=torch.float64), tensor([0])]
    """

    def __init__(
        self,
        full_data_pdf: pd.DataFrame,
        field_dims: list[int],
        split_month: tuple[int, int] = (1, 2),
        batch_size: int = 128,
        num_workers: int = 32,
    ) -> None:
        self.full_data_pdf = full_data_pdf
        # Split indices for train, validation, and test sets
        (
            self.train_indices,
            self.valid_indices,
            self.test_indices,
        ) = self._split_indices(split_month)
        self.dataset = ModelDataset(self.full_data_pdf, field_dims)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _split_indices(
        self, split_month: tuple[int, int]
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Introduce a time-based split strategy"""
        # Sort the DataFrame based on event_occurrence
        self.full_data_pdf = self.full_data_pdf.sort_values(by="event_occurrence")

        # Set the threshold dates for validation and test sets
        validation_threshold = self.full_data_pdf[
            "event_occurrence"
        ].max() - pd.DateOffset(months=split_month[1])
        test_threshold = self.full_data_pdf["event_occurrence"].max() - pd.DateOffset(
            months=split_month[0]
        )

        # Split indices based on time
        train_indices = torch.tensor(
            self.full_data_pdf[
                self.full_data_pdf["event_occurrence"] < validation_threshold
            ].index,
            dtype=torch.int64,
        )
        valid_indices = torch.tensor(
            self.full_data_pdf[
                (self.full_data_pdf["event_occurrence"] >= validation_threshold)
                & (self.full_data_pdf["event_occurrence"] < test_threshold)
            ].index,
            dtype=torch.int64,
        )
        test_indices = torch.tensor(
            self.full_data_pdf[
                self.full_data_pdf["event_occurrence"] >= test_threshold
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
