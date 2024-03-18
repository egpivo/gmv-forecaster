import pandas as pd


class DataHandler:
    """
    Examples
    --------
    >>> from forecaster.data.data_handler import DataHandler
    >>> DataHandler.fetch_user_data("data/users.csv").head(1)
                                   user_id gender   age
    0  3cf2d95c-851a-3e66-bd62-36050c1aa8dd      M  30.0
    """

    @staticmethod
    def fetch_user_data(data_path: str) -> pd.DataFrame:
        """Return: 'user_id', 'gender', 'age'
        Notes
        -----
        - Will remove missing data in this work, which can be enhanced later
        """
        default_columns = ["id", "gender", "age"]
        user_pdf = pd.read_csv(data_path)

        assert any(
            user_pdf.columns == default_columns
        ), f"Check the input columns of users, got {user_pdf.columns}"
        return user_pdf

    @staticmethod
    def fetch_transaction_data(data_path: str) -> pd.DataFrame:
        """Return: 'id', 'user_id', 'store_id', 'event_occurrence', 'amount'"""
        default_columns = ["id", "user_id", "store_id", "event_occurrence", "amount"]
        transaction_pdf = pd.read_csv(data_path, parse_dates=["event_occurrence"])

        assert any(
            transaction_pdf.columns == default_columns
        ), f"Check the input columns of transactions, got {transaction_pdf.columns}"
        return transaction_pdf

    @staticmethod
    def fetch_store_data(data_path: str) -> pd.DataFrame:
        """Return: 'id', 'nam', 'laa', 'category', 'lat', 'lon'"""
        default_columns = ["id", "nam", "laa", "category", "lat", "lon"]
        store_pdf = pd.read_csv(data_path)

        assert any(
            store_pdf.columns == default_columns
        ), f"Check the input columns of stores, got {store_pdf.columns}"
        return store_pdf
