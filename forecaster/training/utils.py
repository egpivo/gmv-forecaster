import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm


def train_model(
    model: Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    criterion: _Loss,
    device: str,
    log_interval: int = 100,
) -> None:
    """
    Train the model using the provided data loader.

    Args:
        model (Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating model parameters.
        data_loader (DataLoader): DataLoader containing the training dataset.
        criterion (_Loss): Loss function used for training.
        device (str): Device to run the training on (e.g., 'cpu', 'cuda').
        log_interval (int): Interval for logging training loss.

    Returns:
        None
    """
    model.train()
    total_loss = 0
    wrapped_loader = tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (input_data, target) in enumerate(wrapped_loader):
        input_data, target = input_data.to(device), target.to(device)
        y = model(input_data)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            wrapped_loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def val_model(model: Module, data_loader: DataLoader, device: str) -> Tensor:
    """
    Evaluate the model using the provided data loader.

    Args:
        model (Module): The trained model to be evaluated.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        device (str): Device to run the evaluation on (e.g., 'cpu', 'cuda').

    Returns: AUROC score
    """
    model.eval()
    targets = torch.tensor([]).to(device, dtype=torch.long)
    predicts = torch.tensor([]).to(device)

    with torch.no_grad():
        for input_data, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            input_data, target = input_data.to(device), target.to(device)
            predict = model(input_data)
            targets = torch.cat((targets, target))
            predicts = torch.cat((predicts, predict))
    auroc = binary_auroc(predicts, targets)

    return auroc
