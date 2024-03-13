import pandas as pd
from sklearn.preprocessing import LabelEncoder

from forecaster.data.data_handler import DataHandler
from forecaster.data.utils import generate_negative_samples

UNSEEN_USER_ID = "-1"
UNSEEN_STORE_ID = "-1"


def label_encode(column: pd.Series) -> pd.Series:
    return LabelEncoder().fit_transform(column)


def transform_temporal_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf["is_weekend"] = (pdf["event_occurrence"].dt.weekday >= 5) * 1
    pdf["season"] = (pdf["event_occurrence"].dt.month % 12 + 3) // 3
    pdf["month"] = pdf["event_occurrence"].dt.month
    return pdf


class DataPreprocessor:
    """
    Notes
    -----
    - This class will take over the main data wrangling logic and create a new features


    Examples
    --------
    >>> from forecaster.data.data_proprocessor import DataPreprocessor
    >>> processor = DataPreprocessor("data/users.csv", "data/transactions.csv", "data/stores.csv")
    >>> pdf = processor.process()
    >>> pdf.shape
    (9493566, 21)
    >>> pdf.isnull().sum().sum()
    0
    """

    _user_fields = [
        "user_id_label",
        "gender_label",
        "age",
    ]
    _store_fields = [
        "store_id_label",
        "nam_label",
        "laa_label",
        "category_label",
        "lat",
        "lon",
    ]
    _context_fields = [
        "is_weekend",
        "season",
        "month",
    ]

    _return_columns = [
        "user_id",
        "store_id",
        "event_occurrence",
        "gender",
        "nam",
        "laa",
        "category",
        "amount",
        # User Field
        *_user_fields,
        # Store Field
        *_store_fields,
        # Contex Field
        *_context_fields,
        # Target
        "label",
    ]

    def __init__(
        self,
        user_data_path: str,
        transaction_data_path: str,
        store_data_path: str,
        num_negative_samples: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> None:

        self.user_pdf = UserDataPreprocessor(user_data_path).process()
        self.transaction_pdf = TransactionDataPreprocessor(
            transaction_data_path, start_date, end_date
        ).process()
        self.store_pdf = StoreDataPreprocessor(store_data_path).process()

        self.num_negative_samples = num_negative_samples

    @property
    def field_dims(self) -> list[int]:
        user_fields_dims = list(self.user_pdf[self._user_fields].nunique() + 1)
        store_fields_dims = list(self.store_pdf[self._store_fields].nunique() + 1)
        # is_weekends/seasons/months
        context_fields_dims = [3, 5, 13]
        return user_fields_dims + store_fields_dims + context_fields_dims

    def process(self) -> pd.DataFrame:
        label_data_pdf = generate_negative_samples(
            self.transaction_pdf, num_negative_samples=self.num_negative_samples
        )
        added_temporal_features = transform_temporal_features(label_data_pdf)
        merged_data_pdf = pd.merge(
            pd.merge(
                added_temporal_features, self.user_pdf, on=["user_id"], how="left"
            ),
            self.store_pdf,
            on=["store_id"],
            how="left",
        )
        return merged_data_pdf[self._return_columns]


class UserDataPreprocessor:

    _return_columns = [
        "user_id",
        "gender",
        "age",
        "user_id_label",
        "gender_label",
    ]

    def __init__(
        self,
        data_path: str,
    ) -> None:
        self.user_pdf = DataHandler.fetch_user_data(data_path)

    def process(self) -> pd.DataFrame:
        """Force the dataset with None tuples for resolving cold-start issues"""
        user_pdf = self.user_pdf.rename({"id": "user_id"}, axis=1)
        null_user_row = pd.DataFrame({key: [None] for key in user_pdf.columns})

        user_pdf = pd.concat([null_user_row, user_pdf], ignore_index=True)

        # Imputation
        user_pdf["user_id"] = user_pdf["user_id"].fillna(UNSEEN_USER_ID)
        user_pdf["age"] = user_pdf["age"].fillna(user_pdf["age"].median())
        user_pdf["gender"] = user_pdf["gender"].fillna("NULL")

        # Label encoding
        for column in ("user_id", "gender"):
            user_pdf[f"{column}_label"] = label_encode(user_pdf[column])

        return user_pdf[self._return_columns]


class StoreDataPreprocessor:

    _return_columns = [
        "store_id",
        "nam",
        "laa",
        "category",
        "lat",
        "lon",
        "store_id_label",
        "nam_label",
        "laa_label",
        "category_label",
    ]

    def __init__(
        self,
        data_path: str,
    ) -> None:
        self.store_pdf = DataHandler.fetch_store_data(data_path)

    def process(self) -> pd.DataFrame:
        store_pdf = self.store_pdf.rename({"id": "store_id"}, axis=1)
        null_store_row = pd.DataFrame({key: [None] for key in store_pdf.columns})

        store_pdf = pd.concat([store_pdf, null_store_row], ignore_index=True)

        store_pdf["store_id"] = store_pdf["store_id"].fillna(UNSEEN_STORE_ID)
        store_pdf["nam"] = store_pdf["nam"].fillna("NULL")
        store_pdf["laa"] = store_pdf["laa"].fillna("NULL")
        store_pdf["category"] = store_pdf["category"].fillna("NULL")
        store_pdf["lat"] = store_pdf["lat"].fillna(store_pdf["lat"].median())
        store_pdf["lon"] = store_pdf["lon"].fillna(store_pdf["lon"].median())

        # Label encoding
        for column in ("store_id", "nam", "laa", "category"):
            store_pdf[f"{column}_label"] = label_encode(store_pdf[column])
        return store_pdf


class TransactionDataPreprocessor:
    _return_columns = ["user_id", "store_id", "event_occurrence", "amount"]

    def __init__(
        self,
        data_path: str,
        start_date: str = None,
        end_date: str = None,
    ) -> None:
        self.transaction_pdf = DataHandler.fetch_transaction_data(data_path)
        self.start_date = start_date
        self.end_date = end_date

    def process(self) -> pd.DataFrame:
        # Filter data in a range
        if self.start_date:
            self.transaction_pdf = self.transaction_pdf[
                pd.to_datetime(self.start_date) <= self.transaction_pdf.event_occurrence
            ]
        if self.end_date:
            self.transaction_pdf = self.transaction_pdf[
                self.transaction_pdf.event_occurrence < pd.to_datetime(self.end_date)
            ]
        null_transaction_pdf_row = pd.DataFrame(
            {key: [None] for key in self.transaction_pdf.columns}
        )
        transaction_pdf = pd.concat(
            [self.transaction_pdf, null_transaction_pdf_row], ignore_index=True
        )

        transaction_pdf["user_id"] = transaction_pdf["user_id"].fillna(UNSEEN_USER_ID)
        transaction_pdf["store_id"] = transaction_pdf["store_id"].fillna(
            UNSEEN_STORE_ID
        )
        transaction_pdf["event_occurrence"] = transaction_pdf[
            "event_occurrence"
        ].fillna(transaction_pdf["event_occurrence"].median())
        transaction_pdf["amount"] = transaction_pdf["amount"].fillna(0)

        return transaction_pdf[self._return_columns]
