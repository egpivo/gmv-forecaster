from collections import defaultdict

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.retrieval import retrieval_recall
from tqdm import tqdm


def test_model(model, data_loader: DataLoader, device, top_k: int = 10) -> float:
    """
    Evaluate the model using the provided data loader.

    Args:
        model: The trained model to be evaluated.
        data_loader (DataLoader): DataLoader containing the evaluation dataset.
        device: Device to run the evaluation on (e.g., 'cpu', 'cuda').
        top_k (int): Number of top predictions to consider for recall calculation.

    Returns:
        float: recall@k score.
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

    return recall_at_k.item()


def calculate_embeddings(model, user_label_pdf, store_label_pdf):
    user_embeddings = {}
    for _, row in user_label_pdf.iterrows():
        user_embeddings[row[0]] = torch.sum(
            model.embedding.embedding.weight[row], dim=0
        ).tolist()

    store_embeddings = {}
    for _, row in store_label_pdf.iterrows():
        store_embeddings[row[0]] = torch.sum(
            model.embedding.embedding.weight[row], dim=0
        ).tolist()

    return user_embeddings, store_embeddings


def create_faiss_index(store_embeddings):
    store_embeddings_np = np.array(list(store_embeddings.values()), dtype=np.float32)
    index = faiss.IndexFlatL2(16)
    index.add(store_embeddings_np)
    return index


def estimate_gmv_per_user(user_embeddings, store_embeddings, index, top_k=5):
    top_stores_with_scores = defaultdict(list)

    def rescale_scores(scores, new_min, new_max):
        min_score = min(scores)
        max_score = max(scores)
        scaled_scores = []
        for score in scores:
            scaled_score = ((score - min_score) / (max_score - min_score)) * (
                new_max - new_min
            ) + new_min
            scaled_scores.append(scaled_score)
        return scaled_scores

    for user_id, user_embedding in user_embeddings.items():
        distances, indices = index.search(
            np.array([user_embedding], dtype=np.float32), k=top_k
        )
        similarity_scores = -distances.flatten()
        rescaled_similarity_scores = rescale_scores(similarity_scores, 0, 1)
        top_store_ids = [list(store_embeddings.keys())[i] for i in indices[0]]
        top_stores_with_scores[user_id] = list(
            zip(top_store_ids, rescaled_similarity_scores)
        )

    return top_stores_with_scores


def calculate_estimated_gmv(top_stores_with_scores, avg_store_amount, scale):
    estimated_gmv_per_user = defaultdict(float)

    for user_id, store_id_info in top_stores_with_scores.items():
        for store_id_label, store_prob in store_id_info:
            if store_id_label in avg_store_amount:
                avg_amount = avg_store_amount[store_id_label]
                estimated_gmv_per_user[user_id] += scale * store_prob * avg_amount

    return estimated_gmv_per_user
