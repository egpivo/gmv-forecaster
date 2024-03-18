import torch

from forecaster.data.data_preprocessor import DataPreprocessor
from forecaster.inference.utils import (
    calculate_embeddings,
    calculate_estimated_gmv,
    create_faiss_index,
    estimate_gmv_per_user,
    preprocess_inference_data,
)


class UserGmvForecaster:
    """
    Forecast GMV for each user based on their top-5 stores and average store GMV.

    Parameters
    ----------
    model_path : str
        Path to the trained XDFM model.
    users_csv_path : str
        Path to the CSV file containing user data.
    transactions_csv_path : str
        Path to the CSV file containing transaction data.
    stores_csv_path : str
        Path to the CSV file containing store data.
    top_k : int, optional
        Number of top stores to consider for each user. Defaults to 5.

    Returns
    -------
    dict
        Estimated GMV for each user.

    Notes
    -----
    - This method utilizes an Extreme Deep Factorization Machine (XDFM) model to calculate embeddings for users and stores.
    - The top-k stores for each user are determined based on their embeddings and a Faiss index.
    - GMV is estimated by multiplying the probability of each store with its average transaction amount.
    - The estimation can be scaled based on a specified factor (e.g., number of days).
    """

    def __init__(
        self,
        model_path: str,
        users_csv_path: str,
        transactions_csv_path: str,
        stores_csv_path: str,
        top_k: int = 2,
    ) -> None:
        """
        Initialize the UserGmvForecaster.

        Parameters
        ----------
        model_path : str
            Path to the trained XDFM model.
        users_csv_path : str
            Path to the CSV file containing user data.
        transactions_csv_path : str
            Path to the CSV file containing transaction data.
        stores_csv_path : str
            Path to the CSV file containing store data.
        top_k : int, optional
            Number of top stores to consider for each user, by default 2 based on EDA
        """
        assert top_k > 1, f"Please enter a larger top-k number, but got {top_k}"

        self.processor = DataPreprocessor(
            users_csv_path, transactions_csv_path, stores_csv_path
        )
        self.model = torch.load(model_path)
        self.model.eval()

        self.users_csv_path = users_csv_path
        self.transactions_csv_path = transactions_csv_path
        self.stores_csv_path = stores_csv_path

        self.top_k = top_k
        self.avg_store_amount = self._calculate_avg_store_amount()

    def forecast(self, predicted_date: str) -> dict:
        """
        Forecast GMV for each user based on their top-5 stores.

        Parameters
        ----------
        predicted_date : str
            The date for which to make the forecast (in format 'YYYYMMDD').

        Returns
        -------
        dict
            Estimated GMV for each user.
        """
        user_label_pdf, store_label_pdf = preprocess_inference_data(
            self.users_csv_path,
            self.transactions_csv_path,
            self.stores_csv_path,
            predicted_date,
        )
        user_embeddings, store_embeddings = calculate_embeddings(
            self.model, user_label_pdf, store_label_pdf
        )
        index = create_faiss_index(store_embeddings)
        top_stores_with_scores = estimate_gmv_per_user(
            user_embeddings, store_embeddings, index, self.top_k
        )
        estimated_gmv_per_user = calculate_estimated_gmv(
            top_stores_with_scores, self.avg_store_amount
        )
        return estimated_gmv_per_user

    def _calculate_avg_store_amount(self) -> dict:
        """
        Calculate the average transaction amount for each store.

        Returns
        -------
        dict
            Average transaction amount for each store.
        """
        full_pdf = self.processor.process()
        return full_pdf.groupby("store_id_label")["amount"].mean().to_dict()
