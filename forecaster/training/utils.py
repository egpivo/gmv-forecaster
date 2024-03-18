import pandas as pd
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

from forecaster.data.data_proprocessor import DataPreprocessor


def train_model(
    model: Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
    log_interval: int = 100,
) -> None:
    """
    Train the model using the provided data loader.

    Args:
        model (Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating model parameters.
        data_loader (DataLoader): DataLoader containing the training dataset.
        criterion: Loss function used for training.
        device (torch.device): Device to run the training on (e.g., 'cpu', 'cuda').
        log_interval (int): Interval for logging training loss.

    Returns:
        None
    """
    model.train()
    total_loss = 0
    progressor = tqdm(data_loader, smoothing=0, mininterval=1.0)
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

    Args:
        model (Module): The trained model to be evaluated.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        device (torch.device): Device to run the evaluation on (e.g., 'cpu', 'cuda').

    Returns:
        AUROC score (float)
    """
    model.eval()
    all_targets = torch.tensor([]).to(device, dtype=torch.long)
    all_predictions = torch.tensor([]).to(device)
    with torch.no_grad():
        for input_data, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
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
) -> pd.Series:
    processor = DataPreprocessor(
        user_data_path,
        transaction_data_path,
        store_data_path,
        is_negative_sampling=False,
    )
    feature_pdf = processor.process()[
        [*processor._user_fields, *processor._store_fields, *processor._context_fields]
    ]

    return feature_pdf.apply(max) + 2, feature_pdf
