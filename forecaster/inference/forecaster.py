import torch

from forecaster.data.data_proprocessor import DataPreprocessor
from forecaster.data.utils import preprocess_inference_data
from forecaster.inference.utils import (
    calculate_embeddings,
    calculate_estimated_gmv,
    create_faiss_index,
    estimate_gmv_per_user,
)


class UserGmvForecaster:
    """
    Forecast GMV for each user based on their top-5 stores and average store GMV.

    Parameters:
        - model_path (str): Path to the trained model checkpoint.
        - user_csv_path (str): Path to the user data CSV file.
        - transactions_csv_path (str): Path to the transactions data CSV file.
        - stores_csv_path (str): Path to the stores data CSV file.
        - scale (int): Scale factor for estimation (e.g., number of days).

    Returns:
        dict: Estimated GMV for each user.

    Notes:
        - This method utilizes an Extreme Deep Factorization Machine (XDFM) model to calculate embeddings for users and stores.
        - The top-5 stores for each user are determined based on their embeddings and a Faiss index.
        - GMV is estimated by multiplying the probability of each store with its average transaction amount.
        - The estimation can be scaled based on a specified factor (e.g., number of days).
    """

    def __init__(
        self,
        model_path,
        user_csv_path,
        transactions_csv_path,
        stores_csv_path,
        scale=30,
    ):
        self.model = torch.load(model_path)
        self.model.eval()
        self.processor = DataPreprocessor(
            user_csv_path, transactions_csv_path, stores_csv_path
        )
        self.full_pdf = self.processor.process()
        self.user_label_pdf, self.store_label_pdf = preprocess_inference_data(
            self.processor, self.full_pdf
        )
        self.scale = scale

    def forecast(self):
        user_embeddings, store_embeddings = calculate_embeddings(
            self.model, self.user_label_pdf, self.store_label_pdf
        )
        index = create_faiss_index(store_embeddings)
        top_stores_with_scores = estimate_gmv_per_user(
            user_embeddings, store_embeddings, index
        )
        avg_store_amount = (
            self.full_pdf.groupby("store_id_label")["amount"].mean().to_dict()
        )
        estimated_gmv_per_user = calculate_estimated_gmv(
            top_stores_with_scores, avg_store_amount, self.scale
        )
        return estimated_gmv_per_user


if __name__ == "__main__":
    forecaster = UserGmvForecaster(
        "checkpoint/xdfm.pt",
        "data/users.csv",
        "data/transactions.csv",
        "data/stores.csv",
    )
    estimated_gmv_per_user = forecaster.forecast()
