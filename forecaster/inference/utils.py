from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.retrieval import retrieval_recall
from tqdm import tqdm

from forecaster.data.data_proprocessor import DataPreprocessor
from forecaster.training.utils import calculate_field_dims


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


def calculate_estimated_gmv(top_stores_with_scores, avg_store_amount):
    estimated_gmv_per_user = defaultdict(float)

    for user_id, store_id_info in top_stores_with_scores.items():
        for store_id_label, store_prob in store_id_info:
            if store_id_label in avg_store_amount:
                avg_amount = avg_store_amount[store_id_label]
                estimated_gmv_per_user[user_id] += store_prob * avg_amount

    return estimated_gmv_per_user


def get_context_features(
    users_csv_path, transactions_csv_path, stores_csv_path, predicted_date
):
    processor = DataPreprocessor(
        users_csv_path,
        transactions_csv_path,
        stores_csv_path,
        is_negative_sampling=False,
    )
    full_data_pdf = processor.process()
    full_data_pdf["event_occurrence"] = pd.to_datetime(
        full_data_pdf["event_occurrence"]
    )
    upper_date = pd.to_datetime(predicted_date, format="%Y%m%d")
    lower_date = upper_date - pd.DateOffset(years=1)
    condition = (full_data_pdf["event_occurrence"] >= lower_date) & (
        full_data_pdf["event_occurrence"] < upper_date
    )
    context_features_pdf = full_data_pdf[condition][
        ["user_id_label", "store_id_label", *processor._context_fields[5:]]
    ]
    user_features = [
        column for column in context_features_pdf.columns if "user" in column
    ]
    store_features = [
        column for column in context_features_pdf.columns if "store" in column
    ]

    user_context_pdf = context_features_pdf[
        ["transaction_age", *user_features]
    ].drop_duplicates(subset=["user_id_label"])
    store_context_pdf = context_features_pdf[store_features].drop_duplicates(
        subset=["store_id_label"]
    )
    return user_context_pdf, store_context_pdf


def preprocess_inference_data(
    users_csv_path, transactions_csv_path, stores_csv_path, predicted_date
):
    field_dims, feature_pdf = calculate_field_dims(
        users_csv_path, transactions_csv_path, stores_csv_path
    )
    feature_list = list(feature_pdf.columns)
    cumulative_field_dims = np.cumsum(field_dims)

    for index, column in enumerate(feature_list[1:]):
        feature_pdf[column] += cumulative_field_dims.iloc[index]

    # Calculate temporal features based on the predicted date
    temporal_labels = get_temporal_labels(predicted_date)
    user_context_pdf, store_context_pdf = get_context_features(
        users_csv_path, transactions_csv_path, stores_csv_path, predicted_date
    )

    user_label_pdf = feature_pdf[
        ["user_id_label", "gender_label", "age_label"]
    ].drop_duplicates(subset=["user_id_label"])

    # the label = 0 means nan
    merged_user_df = user_label_pdf.merge(
        user_context_pdf, on="user_id_label", how="left"
    ).fillna(0)

    # Add the calculated temporal labels to the user_label_pdf
    for key, label in temporal_labels.items():
        merged_user_df[key] = cumulative_field_dims[feature_list.index(key) - 1] + label

    store_label_pdf = feature_pdf[
        ["store_id_label", "nam_label", "laa_label", "category_label", "spatial_label"]
    ].drop_duplicates(subset=["store_id_label"])

    # the label = 0 means nan
    merged_store_df = store_label_pdf.merge(
        store_context_pdf, on="store_id_label", how="left"
    ).fillna(0)
    return merged_user_df, merged_store_df


def get_temporal_labels(predicted_date: str) -> dict:
    # Convert predicted date string to datetime object
    predicted_date = pd.to_datetime(predicted_date, format="%Y%m%d")

    weekday = predicted_date.weekday()
    is_weekend = int(predicted_date.weekday() >= 5)
    season = (predicted_date.month % 12 + 3) // 3
    month = predicted_date.month

    # Return the calculated labels as a dictionary
    labels = {
        "weekday": weekday,
        "is_weekend": is_weekend,
        "season": season,
        "month": month,
    }
    return labels


# a, b = preprocess_inference_data(users_csv_path, transactions_csv_path, stores_csv_path, predicted_date)
# user_context_pdf, store_context_pdf = get_context_features(users_csv_path, transactions_csv_path, stores_csv_path,
#                                                            predicted_date)
