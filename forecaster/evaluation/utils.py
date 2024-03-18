import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.retrieval import retrieval_recall
from tqdm import tqdm


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


def test_model(
    model: torch.nn.Module, data_loader: DataLoader, device: str, top_k: int = 10
) -> float:
    """
    Evaluate the model using the provided data loader.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to be evaluated.
    data_loader : DataLoader
        DataLoader containing the evaluation dataset.
    device : str
        Device to run the evaluation on (e.g., 'cpu', 'cuda').
    top_k : int, optional
        Number of top predictions to consider for recall calculation, by default 10.

    Returns
    -------
    float
        recall@k score.
    """
    model.eval()
    targets = torch.tensor([]).to(device=device, dtype=torch.long)
    predicts = torch.tensor([]).to(device=device)

    with torch.no_grad():
        for input_data, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            input_data, target = input_data.to(device), target.to(device)
            predict = model(input_data)
            targets = torch.cat((targets, target))
            predicts = torch.cat((predicts, predict))

    recall_at_k = retrieval_recall(predicts, targets, top_k=top_k)
    return recall_at_k.item()
