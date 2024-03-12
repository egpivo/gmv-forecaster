import torch
from forecaster.data.utils import generate_negative_samples
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """
    Examples
    --------
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> from forecaster.data.training_data_generator import TrainingDataset
    >>> full_data_pdf = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv").process()
    >>> dataset = TrainingDataset(full_data_pdf)
    >>> train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(dataset.train_indices))
    >>> train_loader
    """
    def __init__(self, full_data_pdf, num_negative_samples=5, random_state=42, split_ratio=(0.8, 0.1, 0.1)):
        self.full_data_pdf = full_data_pdf
        self.positive_pairs = full_data_pdf[["user_id", "store_id"]].drop_duplicates()
        self.num_negative_samples = num_negative_samples
        self.random_state = random_state

        # Generate negative samples once and store them
        self.all_samples = self._generate_samples(full_data_pdf)

        # Split indices for train, validation, and test sets
        self.train_indices, self.valid_indices, self.test_indices = self._split_indices()

    def _split_indices(self):
        # Calculate lengths of splits
        train_size = int(0.8 * len(self.all_samples))
        valid_size = int(0.1 * len(self.all_samples))

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
            'user_id': torch.tensor(self.all_samples.loc[idx, 'user_id']),
            'store_id': torch.tensor(self.all_samples.loc[idx, 'store_id']),
            'label': torch.tensor(self.all_samples.loc[idx, 'label'])
        }

    def _generate_samples(self, data):
        # Your logic to generate negative samples
        return generate_negative_samples(self.positive_pairs, data, self.num_negative_samples, self.random_state)
