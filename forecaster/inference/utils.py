from collections import defaultdict
from typing import Tuple

import faiss
import numpy as np
import pandas as pd
import torch

from forecaster.data.data_preprocessor import CONTEXT_FIELDS, DataPreprocessor
from forecaster.training.utils import calculate_field_dims


def calculate_embeddings(
    model: torch.nn.Module, user_label_pdf: pd.DataFrame, store_label_pdf: pd.DataFrame
) -> Tuple[dict, dict]:
    """
    Calculate embeddings for users and stores.

    Parameters
    ----------
    model : torch.nn.Module
        Trained embedding model.
    user_label_pdf : pd.DataFrame
        DataFrame containing user labels.
    store_label_pdf : pd.DataFrame
        DataFrame containing store labels.

    Returns
    -------
    Tuple[dict, dict]
        Tuple containing user and store embeddings.
    """
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


def create_faiss_index(store_embeddings: dict) -> faiss.IndexFlatL2:
    """
    Create a FAISS index for store embeddings.

    Parameters
    ----------
    store_embeddings : dict
        Dictionary containing store embeddings.

    Returns
    -------
    faiss.IndexFlatL2
        FAISS index for store embeddings.
    """
    store_embeddings_np = np.array(list(store_embeddings.values()), dtype=np.float32)
    index = faiss.IndexFlatL2(store_embeddings_np.shape[0])
    index.add(store_embeddings_np)
    return index


def estimate_gmv_per_user(
    user_embeddings: dict,
    store_embeddings: dict,
    index: faiss.IndexFlatL2,
    top_k: int = 5,
) -> dict:
    """
    Estimate Gross Merchandise Volume (GMV) per user.

    Parameters
    ----------
    user_embeddings : dict
        Dictionary containing user embeddings.
    store_embeddings : dict
        Dictionary containing store embeddings.
    index : faiss.IndexFlatL2
        FAISS index for store embeddings.
    top_k : int, optional
        Number of top stores to consider, by default 5.

    Returns
    -------
    dict
        Dictionary containing estimated GMV per user.
    """
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


def calculate_estimated_gmv(
    top_stores_with_scores: dict, avg_store_amount: dict
) -> dict:
    """
    Calculate estimated Gross Merchandise Volume (GMV) per user.

    Parameters
    ----------
    top_stores_with_scores : dict
        Dictionary containing top stores with scores for each user.
    avg_store_amount : dict
        Dictionary containing average store amounts.

    Returns
    -------
    dict
        Dictionary containing estimated GMV per user.
    """
    estimated_gmv_per_user = defaultdict(float)

    for user_id, store_id_info in top_stores_with_scores.items():
        for store_id_label, store_prob in store_id_info:
            if store_id_label in avg_store_amount:
                avg_amount = avg_store_amount[store_id_label]
                estimated_gmv_per_user[user_id] += store_prob * avg_amount

    return estimated_gmv_per_user


def get_context_features(
    users_csv_path: str,
    transactions_csv_path: str,
    stores_csv_path: str,
    predicted_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get context features from CSV files.

    Parameters
    ----------
    users_csv_path : str
        Path to the users CSV file.
    transactions_csv_path : str
        Path to the transactions CSV file.
    stores_csv_path : str
        Path to the stores CSV file.
    predicted_date : str
        Predicted date in 'YYYYMMDD' format.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing DataFrames for user and store context features.
    """
    full_data_pdf = DataPreprocessor(
        users_csv_path,
        transactions_csv_path,
        stores_csv_path,
        is_negative_sampling=False,
    ).process()
    full_data_pdf["event_occurrence"] = pd.to_datetime(
        full_data_pdf["event_occurrence"]
    )
    upper_date = pd.to_datetime(predicted_date, format="%Y%m%d")
    lower_date = upper_date - pd.DateOffset(years=1)
    condition = (full_data_pdf["event_occurrence"] >= lower_date) & (
        full_data_pdf["event_occurrence"] < upper_date
    )
    context_features_pdf = full_data_pdf[condition][
        ["user_id_label", "store_id_label", *CONTEXT_FIELDS[5:]]
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
    users_csv_path: str,
    transactions_csv_path: str,
    stores_csv_path: str,
    predicted_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data for inference.

    Parameters
    ----------
    users_csv_path : str
        Path to the users CSV file.
    transactions_csv_path : str
        Path to the transactions CSV file.
    stores_csv_path : str
        Path to the stores CSV file.
    predicted_date : str
        Predicted date in 'YYYYMMDD' format.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing preprocessed DataFrames for users and stores.
    """
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
    """
    Get temporal labels based on the predicted date.

    Parameters
    ----------
    predicted_date : str
        Predicted date in 'YYYYMMDD' format.

    Returns
    -------
    dict
        Dictionary containing temporal labels.
    """
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
