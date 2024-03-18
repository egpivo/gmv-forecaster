from typing import Tuple

import pandas as pd
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

from forecaster.data.data_proprocessor import (
    CONTEXT_FIELDS,
    STORE_FIELDS,
    USER_FIELDS,
    DataPreprocessor,
)


def train_model(
    model: Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
    log_interval: int = 100,
) -> None:
    """
    Train the model using the provided data loader.

    Parameters
    ----------
    model : Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    data_loader : DataLoader
        DataLoader containing the training dataset.
    criterion : callable
        Loss function used for training.
    device : torch.device
        Device to run the training on (e.g., 'cpu', 'cuda').
    log_interval : int, optional
        Interval for logging training loss, by default 100.

    Returns
    -------
    None
    """
    model.train()
    total_loss = 0
    progressor = tqdm(data_loader, desc="Training", smoothing=0, mininterval=1.0)
    for i, (input_data, target) in enumerate(progressor):
        input_data, target = input_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            progressor.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def validate_model(
    model: Module, data_loader: DataLoader, device: torch.device
) -> float:
    """
    Evaluate the model using the provided data loader and return AUROC score.

    Parameters
    ----------
    model : Module
        The trained model to be evaluated.
    data_loader : DataLoader
        DataLoader containing the evaluation dataset.
    device : torch.device
        Device to run the evaluation on (e.g., 'cpu', 'cuda').

    Returns
    -------
    float
        AUROC score.
    """
    model.eval()
    all_targets = torch.tensor([]).to(device, dtype=torch.long)
    all_predictions = torch.tensor([]).to(device)
    with torch.no_grad():
        progressor = tqdm(data_loader, desc="Validation", smoothing=0, mininterval=1.0)
        for input_data, target in progressor:
            input_data, target = input_data.to(device), target.to(device)
            output = model(input_data)
            all_targets = torch.cat((all_targets, target))
            all_predictions = torch.cat((all_predictions, output))
    auroc = binary_auroc(all_predictions, all_targets)
    return auroc.item()


def calculate_field_dims(
    user_data_path: str,
    transaction_data_path: str,
    store_data_path: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate the field dimensions based on the provided data paths.

    Parameters
    ----------
    user_data_path : str
        Path to the user data file.
    transaction_data_path : str
        Path to the transaction data file.
    store_data_path : str
        Path to the store data file.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Series containing the field dimensions and DataFrame with processed data.
    """
    full_pdf = DataPreprocessor(
        user_data_path,
        transaction_data_path,
        store_data_path,
        is_negative_sampling=False,
    ).process()
    feature_pdf = full_pdf[[*USER_FIELDS, *STORE_FIELDS, *CONTEXT_FIELDS]]
    return feature_pdf.apply(max) + 2, feature_pdf
