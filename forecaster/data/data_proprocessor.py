import pandas as pd
from sklearn.preprocessing import LabelEncoder

from forecaster.data.data_handler import DataHandler
from forecaster.data.utils import generate_negative_samples


class DataPreprocessor:
    """
    Notes
    -----
    - This class will take over the main data wrangling logic and create a new features


    Examples
    --------
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> processor = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv")
    >>> processor.process().shape
    (8531868, 16)
    """

    _return_columns = [
        "user_id",
        "store_id",
        "gender",
        "nam",
        "laa",
        "category",
        # User Field
        "user_id_label",
        "gender_label",
        "age",
        # Store Field
        "store_id_label",
        "nam_label",
        "laa_label",
        "category_label",
        "lat",
        "lon",
        # Contex Field
        "is_weekend",
        "season",
        "month",
        "amount",
        # Target
        "label",
    ]

    def __init__(
        self,
        user_data_path: str,
        transaction_data_path: str,
        store_data_path: str,
        num_negative_samples: int = 5,
    ) -> None:
        self.handler = DataHandler(
            user_data_path, transaction_data_path, store_data_path
        )
        self.num_negative_samples = num_negative_samples

    def _label_encode(self, column: pd.Series) -> pd.Series:
        return LabelEncoder().fit_transform(column)

    def _process_user_data(self) -> pd.DataFrame:
        user_pdf = (
            self.handler.fetch_user_data().rename({"id": "user_id"}, axis=1).dropna()
        )
        # Label encoding
        for column in ("user_id", "gender"):
            user_pdf[f"{column}_label"] = self._label_encode(user_pdf[column])
        return user_pdf

    def _process_store_data(self) -> pd.DataFrame:
        store_pdf = (
            self.handler.fetch_store_data().rename({"id": "store_id"}, axis=1).dropna()
        )
        # Label encoding
        for column in ("store_id", "nam", "laa", "category"):
            store_pdf[f"{column}_label"] = self._label_encode(store_pdf[column])
        return store_pdf

    def _process_transaction_data(self) -> pd.DataFrame:
        """Add negative sampling + temporal features"""
        transaction_pdf = self.handler.fetch_transaction_data().drop("id", axis=1)
        # Negative sampling
        label_data_pdf = generate_negative_samples(
            transaction_pdf, num_negative_samples=self.num_negative_samples
        )
        # Extracting temporal features
        label_data_pdf["is_weekend"] = (
            label_data_pdf["event_occurrence"].dt.weekday >= 5
        ) * 1
        label_data_pdf["season"] = (
            label_data_pdf["event_occurrence"].dt.month % 12 + 3
        ) // 3
        label_data_pdf["month"] = label_data_pdf["event_occurrence"].dt.month
        return label_data_pdf

    def process(self) -> pd.DataFrame:
        # [TODO] Check other methods to deal with Null data
        user_pdf = self._process_user_data()
        transaction_pdf = self._process_transaction_data()
        store_pdf = self._process_store_data()

        merged_data_pdf = pd.merge(
            pd.merge(transaction_pdf, user_pdf, on=["user_id"], how="inner"),
            store_pdf,
            on=["store_id"],
            how="left",
        )
        return merged_data_pdf[self._return_columns]
