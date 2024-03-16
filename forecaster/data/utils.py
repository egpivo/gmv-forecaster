import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def generate_negative_samples(data, num_negative_samples=5, random_state=42):
    np.random.seed(random_state)

    # Get user purchase frequency
    user_purchase_freq = data["user_id"].value_counts().to_frame("freq")

    # Create positive samples
    data["label"] = 1 * (data["amount"] > 0)

    # Generate negative samples
    all_stores = data["store_id"].unique()
    total_samples = num_negative_samples * user_purchase_freq["freq"].sum()

    # Sample stores and maintain temporal structure for event occurrences
    sampled_stores = np.random.choice(all_stores, total_samples, replace=True)
    sampled_event_times = np.random.choice(
        data["event_occurrence"], total_samples, replace=True
    )

    # Create negative samples DataFrame
    negative_samples = pd.DataFrame(
        {
            "user_id": np.repeat(
                user_purchase_freq.index.values,
                num_negative_samples * user_purchase_freq["freq"],
            ),
            "store_id": sampled_stores,
            "event_occurrence": sampled_event_times,
            "amount": np.zeros(total_samples, dtype=int),
            "label": np.zeros(total_samples, dtype=int),
        }
    )

    # Combine positive and negative samples
    balanced_dataset = pd.concat([data, negative_samples], ignore_index=True)
    balanced_dataset = balanced_dataset.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return balanced_dataset


def create_quantile_labels(dataframe, column, num_quantiles):
    """Create column labels based on quantiles for a DataFrame."""
    return pd.qcut(dataframe[column], num_quantiles, labels=False, duplicates="drop")


def create_spatial_labels_kmeans(dataframe, num_clusters):
    """
    Apply K-means clustering to latitude and longitude coordinates and add cluster labels to a DataFrame.

    Parameters:
    - dataframe: DataFrame containing the 'lat' and 'lon' columns.
    - num_clusters: Number of clusters for K-means clustering.

    Returns:
    - DataFrame with an additional column 'cluster_label' containing the cluster labels.
    """
    coordinates = dataframe[["lat", "lon"]].values

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(coordinates)


def generate_gmv_label_by_periods(df, group_column, preffix, num_quantiles):
    """
    Calculate GMV label by periods for each group (e.g., store or user).

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing transaction data.
    group_column : str
        The name of the column representing the group (e.g., 'store_id' or 'user_id').
    amount_column : str
        The name of the column representing the transaction amount (e.g., 'revenue' or 'contribution').
    num_quantiles : int
        The number of quantiles to use for discretizing the GMV.

    Returns:
    --------
    df : pandas DataFrame
        The DataFrame with GMV labels added for each period.

    Examples:
    ---------
    >>> transaction_pdf_with_revenue = calculate_gmv_label_by_periods(transaction_pdf, 'store_id', 'store', num_quantiles)
    """

    # Convert 'event_occurrence' to datetime
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])

    # Find the last date in the dataset
    last_date = df["event_occurrence"].max()

    # Define periods and their corresponding start dates
    periods = {
        f"last_month_{preffix}_gmv": 1,
        f"last_quarter_{preffix}_gmv": 3,
        f"last_half_year_{preffix}_gmv": 6,
        f"last_year_{preffix}_gmv": 12,
    }
    start_dates = {
        period: last_date - pd.offsets.DateOffset(months=months)
        for period, months in periods.items()
    }

    # Extract the month from the 'event_occurrence' column
    # Calculate the sum of 'amount' for each group in each period
    for period, start_date in start_dates.items():
        period_transactions = df[df["event_occurrence"] >= start_date]
        period_amount = period_transactions.groupby([group_column])["amount"].sum()
        df[period] = df[group_column].map(period_amount).fillna(0)

    # Create quantile labels for each period's amount
    for period in periods:
        df[f"{period}_label"] = create_quantile_labels(df, period, num_quantiles)

    return df


def generate_purchase_label_by_periods(df, group_column, preffix, num_quantiles):
    """
    Calculate purchase frequency label by periods for each group (e.g., store or user).

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing transaction data.
    group_column : str
        The name of the column representing the group (e.g., 'store_id' or 'user_id').
    preffix : str
        The prefix to add to the column names.
    num_quantiles : int
        The number of quantiles to use for discretizing the frequency.

    Returns:
    --------
    df : pandas DataFrame
        The DataFrame with frequency labels added for each period.

    Examples:
    ---------
    >>> transaction_pdf_with_frequency = generate_frequency_label_by_periods(transaction_pdf, 'store_id', 'store', num_quantiles)
    """

    # Convert 'event_occurrence' to datetime
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])

    # Find the last date in the dataset
    last_date = df["event_occurrence"].max()

    # Define periods and their corresponding start dates
    periods = {
        f"last_month_{preffix}_purchase": 1,
        f"last_quarter_{preffix}_purchase": 3,
        f"last_half_year_{preffix}_purchase": 6,
        f"last_year_{preffix}_purchase": 12,
    }
    start_dates = {
        period: last_date - pd.offsets.DateOffset(months=months)
        for period, months in periods.items()
    }

    # Extract the month from the 'event_occurrence' column
    df["month"] = df["event_occurrence"].dt.month

    # Calculate the frequency of transactions for each group in each period
    for period, start_date in start_dates.items():
        period_transactions = df[df["event_occurrence"] >= start_date]
        period_frequency = period_transactions.groupby([group_column]).size()
        df[period] = df[group_column].map(period_frequency).fillna(0)

    # Create quantile labels for each period's frequency
    for period in periods:
        df[f"{period}_label"] = create_quantile_labels(df, period, num_quantiles)

    return df


def generate_recency_label(df, group_column, prefix, num_quantiles):
    """
    Calculate recency for users or stores based on their transaction history.

    Args:
        df (pandas DataFrame): The DataFrame containing transaction data.
        group_column (str): The name of the column representing the group (e.g., 'user_id' or 'store_id').
        prefix : str
            The prefix to add to the column names.
        num_quantiles : int
            The number of quantiles to use for discretizing the frequency.

    Returns:
        pandas DataFrame: DataFrame with recency calculated for each group.

    """
    # Convert 'event_occurrence' to datetime
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])

    # Find the last date in the dataset
    last_date = df["event_occurrence"].max()
    earliest_date = df["event_occurrence"].min()

    # Calculate recency for each group
    recency_data = df.groupby(group_column)["event_occurrence"].max().reset_index()
    recency_data["recency"] = (last_date - recency_data["event_occurrence"]).dt.days
    column_name = f"{prefix}_recency"
    recency_data = recency_data.set_index(group_column)["recency"]
    df[column_name] = (
        df[group_column].map(recency_data).fillna((last_date - earliest_date).days)
    )
    df[f"{column_name}_label"] = create_quantile_labels(df, column_name, num_quantiles)
    return df


def calculate_transaction_age_label(df, num_quantiles):
    # Find the last transaction date
    last_transaction_date = df["event_occurrence"].max()

    # Calculate transaction age for each transaction
    df["transaction_age"] = (last_transaction_date - df["event_occurrence"]).dt.days
    df["transaction_age_label"] = create_quantile_labels(
        df, "transaction_age", num_quantiles
    )
    return df


def label_encode(column: pd.Series) -> pd.Series:
    return LabelEncoder().fit_transform(column)


def trim_hour(hour: int) -> int:
    if hour < 8:
        return 7
    elif hour > 16:
        return 16
    else:
        return hour


def transform_temporal_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf["event_occurrence"] = pd.to_datetime(pdf["event_occurrence"])

    pdf["hour"] = pdf["event_occurrence"].dt.hour.apply(trim_hour)
    pdf["weekday"] = pdf["event_occurrence"].dt.weekday
    pdf["is_weekend"] = (pdf["event_occurrence"].dt.weekday >= 5) * 1
    pdf["season"] = (pdf["event_occurrence"].dt.month % 12 + 3) // 3
    pdf["month"] = pdf["event_occurrence"].dt.month
    return pdf
