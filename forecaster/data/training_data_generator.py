import torch
from forecaster.data.utils import generate_negative_samples
from torch.utils.data import Dataset, random_split


class TrainingDataset(Dataset):
    """
    Examples
    --------
    >>> import torch
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> from forecaster.data.training_data_generator import TrainingDataset
    >>> full_data_pdf = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv").process()
    >>> dataset = TrainingDataset(full_data_pdf)
    >>> train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(dataset.train_indices))
    >>> next(iter(train_loader))
    {'user_label': tensor([8355, 4416, 7508, 6793, 1011, 7133, 2794, 1201]),
     'store_label': tensor([63634, 83770, 33454, 24558,  8657, 46895, 10226, 65825]),
     'label': tensor([0, 1, 0, 0, 0, 0, 0, 0])}
    """
    def __init__(self, full_data_pdf, split_ratio=(0.8, 0.1)):
        self.all_samples = full_data_pdf.drop(["user_id", "store_id"], axis=1)
        self.split_ratio = split_ratio

        # Split indices for train, validation, and test sets
        self.train_indices, self.valid_indices, self.test_indices = self._split_indices()

    def _split_indices(self):
        # Calculate lengths of splits
        train_size = int(self.split_ratio[0] * len(self.all_samples))
        valid_size = int(self.split_ratio[1] * len(self.all_samples))

        # Generate indices and split
        indices = torch.randperm(len(self.all_samples)).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        return train_indices, valid_indices, test_indices

    def __len__(self):
        # The total length is the sum of all samples
        return len(self.all_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return {
            'user_label': torch.tensor(self.all_samples.loc[idx, 'user_label']),
            'store_label': torch.tensor(self.all_samples.loc[idx, 'store_label']),
            'label': torch.tensor(self.all_samples.loc[idx, 'label'])
        }
