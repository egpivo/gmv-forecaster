import pandas as pd


class DataHandler:
    """
    Examples
    --------
    >>> from forecaster.data.data_handler import DataHandler
    >>> DataHandler(user_data_path="data/users.csv", transaction_data_path="", store_data_path="")
    >>> DataHandler.fetch_user_data().head(1)
        user_id	gender	age
    0	3cf2d95c-851a-3e66-bd62-36050c1aa8dd	M	30.0
    """

    def __init__(self, user_data_path: str, transaction_data_path: str, store_data_path: str) -> None:
        self.user_data_path = user_data_path
        self.transaction_data_path = transaction_data_path
        self.store_data_path = store_data_path

    def fetch_user_data(self) -> pd.DataFrame:
        """Return: 'user_id', 'gender', 'age'
        Notes
        -----
        - Will remove missing data in this work, which can be enhanced later
        """
        default_columns = ['id', 'gender', 'age']
        user_pdf = pd.read_csv(self.user_data_path)

        assert any(user_pdf.columns == default_columns), f"Check the input columns of users, got {user_pdf.columns}"
        return user_pdf.rename({"id": "user_id"}, axis=1).dropna()

    def fetch_transaction_data(self) -> pd.DataFrame:
        """Return: 'id', 'user_id', 'store_id', 'event_occurrence', 'amount'"""
        default_columns = ['id', 'user_id', 'store_id', 'event_occurrence', 'amount']
        transaction_pdf = pd.read_csv(self.transaction_data_path, parse_dates=["event_occurrence"])

        assert any(transaction_pdf.columns == default_columns), f"Check the input columns of transaction, got {transaction_pdf.columns}"
        return transaction_pdf

    def fetch_store_data(self) -> pd.DataFrame:
        """Return: 'user_id', 'gender', 'age'"""
        default_columns = ['id', 'nam', 'laa', 'category', 'lat', 'lon']
        store_pdf = pd.read_csv(self.store_data_path)

        assert any(store_pdf.columns == default_columns), f"Check the input columns of stores, got {store_pdf.columns}"
        return store_pdf.rename({"id": "store_id"}, axis=1)
