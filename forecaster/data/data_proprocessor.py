import pandas as pd

from forecaster.data.data_handler import DataHandler


class DataPreprocessor:
    """
    Notes
    -----
    - This class will take over the main data wrangling logic and create a new features


    Examples
    --------
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> processor = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv")
    >>> processor.process().head(1)
                                    user_id  ... month
    0  93098549-3ff0-e579-01c3-df9183278f64  ...     1
    [1 rows x 14 columns]
    """

    _return_columns = [
        'user_id', 'store_id', 'event_occurrence', 'amount', 'gender',
        'age', 'nam', 'laa', 'category', 'lat', 'lon', 'is_weekend', 'season',
        'month'
    ]
    def __init__(
        self, user_data_path: str, transaction_data_path: str, store_data_path: str
    ) -> None:
        self.handler = DataHandler(user_data_path, transaction_data_path, store_data_path)

    def _add_temporal_features(self, pdf: pd.DataFrame) -> pd.DataFrame:
        pdf['event_occurrence'] = pd.to_datetime(pdf['event_occurrence'])

        # Extracting temporal features
        pdf['is_weekend'] = (pdf['event_occurrence'].dt.weekday >= 5) * 1  # True for weekdays, False for weekends
        pdf['season'] = (pdf['event_occurrence'].dt.month % 12 + 3) // 3  # Calculating the season based on months
        pdf['month'] = pdf['event_occurrence'].dt.month
        return pdf

    def process(self) -> pd.DataFrame:
        # [TODO] Check other methods to deal with Null data
        user_pdf = self.handler.fetch_user_data().rename({"id": "user_id"}, axis=1).dropna()
        transaction_pdf = self.handler.fetch_transaction_data()
        store_pdf = self.handler.fetch_store_data().rename({"id": "store_id"}, axis=1)
        merged_data =  pd.merge(
                pd.merge(transaction_pdf, user_pdf, on=["user_id"], how="inner"),
                store_pdf, on=["store_id"], how="left"
        )
        full_data = self._add_temporal_features(merged_data)
        return full_data[self._return_columns]
