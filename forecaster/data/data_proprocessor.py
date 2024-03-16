import pandas as pd

from forecaster.data.data_handler import DataHandler
from forecaster.data.utils import (
    calculate_transaction_age_label,
    create_quantile_labels,
    create_spatial_labels_kmeans,
    generate_gmv_label_by_periods,
    generate_negative_samples,
    generate_purchase_label_by_periods,
    generate_recency_label,
    label_encode,
    transform_temporal_features,
)

UNSEEN_USER_ID = "-1"
UNSEEN_STORE_ID = "-1"


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
    (9493566, 44)
    >>> pdf.isnull().sum().sum()
    0
    """

    _user_fields = [
        "user_id_label",
        "gender_label",
        "age_label",
    ]
    _store_fields = [
        "store_id_label",
        "nam_label",
        "laa_label",
        "category_label",
        "spatial_label",
    ]
    _context_fields = [
        "hour",
        "weekday",
        "is_weekend",
        "season",
        "month",
        "transaction_age",
        "last_month_user_gmv_label",
        "last_month_store_gmv_label",
        "last_quarter_user_gmv_label",
        "last_quarter_store_gmv_label",
        "last_half_year_user_gmv_label",
        "last_half_year_store_gmv_label",
        "last_year_user_gmv_label",
        "last_year_store_gmv_label",
        "last_month_user_purchase_label",
        "last_month_store_purchase_label",
        "last_quarter_user_purchase_label",
        "last_quarter_store_purchase_label",
        "last_half_year_user_purchase_label",
        "last_half_year_store_purchase_label",
        "last_year_user_purchase_label",
        "last_year_store_purchase_label",
        "user_recency_label",
        "store_recency_label",
    ]

    _return_columns = [
        "user_id",
        "store_id",
        "event_occurrence",
        "gender",
        "age",
        "nam",
        "laa",
        "category",
        "amount",
        "lat",
        "lon",
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
        num_quantiles: int = 5,
        start_date: str = None,
        end_date: str = None,
    ) -> None:
        self.user_pdf = UserDataPreprocessor(user_data_path, num_quantiles).process()
        self.transaction_pdf = TransactionDataPreprocessor(
            transaction_data_path, start_date, end_date
        ).process()
        self.store_pdf = StoreDataPreprocessor(store_data_path).process()
        self.num_quantiles = num_quantiles
        self.num_negative_samples = num_negative_samples

    def add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return transaction age label
        df = calculate_transaction_age_label(df, self.num_quantiles)
        # The number of quantile set 3 for store
        df = generate_gmv_label_by_periods(df, "store_id", "store", 2)
        df = generate_gmv_label_by_periods(df, "user_id", "user", self.num_quantiles)
        df = generate_purchase_label_by_periods(
            df, "store_id", "store", self.num_quantiles
        )
        df = generate_purchase_label_by_periods(
            df, "user_id", "user", self.num_quantiles
        )
        df = generate_recency_label(df, "store_id", "store")
        df = generate_recency_label(df, "user_id", "user")
        return df

    def process(self) -> pd.DataFrame:
        label_data_pdf = generate_negative_samples(
            self.transaction_pdf, num_negative_samples=self.num_negative_samples
        )
        added_temporal_features = transform_temporal_features(label_data_pdf)
        merged_data_pdf = pd.merge(
            pd.merge(
                added_temporal_features, self.user_pdf, on=["user_id"], how="inner"
            ),
            self.store_pdf,
            on=["store_id"],
            how="inner",
        )
        full_data_pdf = self.add_context_features(merged_data_pdf)

        return full_data_pdf[self._return_columns]


class UserDataPreprocessor:

    _return_columns = [
        "user_id",
        "gender",
        "age",
        "user_id_label",
        "gender_label",
        "age_label",
    ]

    def __init__(
        self,
        data_path: str,
        num_quantiles: int = 5,
    ) -> None:
        self.user_pdf = DataHandler.fetch_user_data(data_path)
        self.num_quantiles = num_quantiles

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
        # Discretion
        user_pdf["age_label"] = create_quantile_labels(
            user_pdf, "age", num_quantiles=self.num_quantiles
        )

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
        "spatial_label",
    ]

    def __init__(self, data_path: str, num_clusters: int = 10) -> None:
        self.store_pdf = DataHandler.fetch_store_data(data_path)
        self.num_clusters = num_clusters

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

        store_pdf["spatial_label"] = create_spatial_labels_kmeans(
            store_pdf, num_clusters=self.num_clusters
        )
        return store_pdf


class TransactionDataPreprocessor:
    _return_columns = [
        "user_id",
        "store_id",
        "event_occurrence",
        "amount",
    ]

    def __init__(
        self,
        data_path: str,
        start_date: str = None,
        end_date: str = None,
        num_quantiles: int = 5,
    ) -> None:
        self.transaction_pdf = DataHandler.fetch_transaction_data(data_path)
        self.start_date = start_date
        self.end_date = end_date
        self.num_quantiles = num_quantiles

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
