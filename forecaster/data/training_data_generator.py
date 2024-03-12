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
    )

    def __init__(self, full_data_pdf, split_ratio=(0.8, 0.1)):
        # Drop unnecessary columns and convert to tensors
        selected_features = full_data_pdf.drop([*self._removed_id, "label"], axis=1)
        self.features = torch.tensor(selected_features.values)
        self.field_dims = selected_features.nunique() + 1
        self.labels = torch.tensor(full_data_pdf["label"].values)
        self.split_ratio = split_ratio

        # Split indices for train, validation, and test sets
        (
            self.train_indices,
            self.valid_indices,
            self.test_indices,
        ) = self._split_indices()

    def _split_indices(self):
        # Calculate lengths of splits
        train_size = int(self.split_ratio[0] * len(self.features))
        valid_size = int(self.split_ratio[1] * len(self.features))

        # Generate indices and split
        indices = torch.randperm(len(self.features)).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size : train_size + valid_size]
        test_indices = indices[train_size + valid_size :]
        return train_indices, valid_indices, test_indices

    def __len__(self):
        # The total length is the sum of all samples
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return self.features[idx], self.labels[idx]
