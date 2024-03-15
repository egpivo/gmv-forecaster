import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional.retrieval import retrieval_recall
from tqdm import tqdm


def test_model(model, data_loader: DataLoader, device, top_k: int = 10) -> Tensor:
    """
    Evaluate the model using the provided data loader.

    Args:
        model: The trained model to be evaluated.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        device: Device to run the evaluation on (e.g., 'cpu', 'cuda').
        top_k (int): Number of top predictions to consider for recall calculation.

    Returns: recall@k score.
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

    recall_at_k = retrieval_recall(predicts, targets, top_k=top_k)

    return recall_at_k
