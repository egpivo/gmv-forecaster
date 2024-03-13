import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """
    Examples
    --------
    >>> import torch
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> from forecaster.data.training_data_generator import TrainingDataset
    >>> full_data_pdf = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv").process()
    >>> dataset = TrainingDataset(full_data_pdf)
    >>> train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(dataset.train_indices))
    >>> next(iter(train_loader))
    [tensor([[6.0500e+03, 1.0000e+00, 8.1000e+01, 6.9688e+04, 4.4000e+01, 9.3300e+02,
              1.0000e+00, 3.8076e+01, 1.4020e+02, 0.0000e+00, 2.0000e+00, 5.0000e+00,
              0.0000e+00]], dtype=torch.float64),
     tensor([0])]
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

    def __init__(
        self, full_data_pdf: pd.DataFrame, split_month: tuple[int, int] = (1, 2)
    ):
        # Drop unnecessary columns and convert to tensors
        self.full_data_pdf = full_data_pdf
        # Split indices for train, validation, and test sets
        (
            self.train_indices,
            self.valid_indices,
            self.test_indices,
        ) = self._split_indices(split_month)
        selected_features = self.full_data_pdf.drop(
            [*self._removed_id, "label"], axis=1
        )
        self.features = torch.tensor(selected_features.values)
        self.field_dims = selected_features.nunique() + 1
        self.labels = torch.tensor(self.full_data_pdf["label"].values)

    def _split_indices(
        self, split_month: tuple[int, int]
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
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
        train_indices = self.full_data_pdf[
            self.full_data_pdf["event_occurrence"] < validation_threshold
        ].index
        valid_indices = self.full_data_pdf[
            (self.full_data_pdf["event_occurrence"] >= validation_threshold)
            & (self.full_data_pdf["event_occurrence"] < test_threshold)
        ].index
        test_indices = self.full_data_pdf[
            self.full_data_pdf["event_occurrence"] >= test_threshold
        ].index

        return train_indices, valid_indices, test_indices

    def __len__(self):
        # The total length is the sum of all samples
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.features[idx], self.labels[idx]
