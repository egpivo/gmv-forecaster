import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from forecaster.data import GMV_BINS, PURCHASE_BINS, RECENCY_BINS, TRANSACTIONS_AGE_BINS


def generate_negative_samples(
    data: pd.DataFrame, num_negative_samples: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate negative samples for data balancing.

    Parameters:
        data (pd.DataFrame): Original data frame.
        num_negative_samples (int): Number of negative samples to generate per positive sample.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced data frame with negative samples.
    """
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


def create_quantile_labels(
    dataframe: pd.DataFrame, column: str, num_quantiles: int
) -> pd.Series:
    """
    Create quantile labels for a given column in a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column for which quantile labels are to be created.
        num_quantiles (int): Number of quantiles to divide the data into.

    Returns:
        pd.Series: Series containing quantile labels.
    """
    return pd.qcut(dataframe[column], num_quantiles, labels=False, duplicates="drop")


def create_bin_labels(df: pd.DataFrame, column: str, bins: int) -> pd.Series:
    """
    Create bin labels for a given column in a DataFrame where the label starts from 0 and NaN values are represented as -1.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column for which bin labels are to be created.
        bins (int): Number of bins to divide the data into.

    Returns:
        pd.Series: Series containing bin labels.
    """
    labels = pd.cut(df[column], bins=bins, labels=False)  # Create labels (0-indexed)
    labels = labels.fillna(-1)  # Fill NaNs with a value representing the first bin
    labels += 1  # Shift labels to start from 1
    return labels.astype(int)  # Convert to integer dtype


def create_spatial_labels_kmeans(
    dataframe: pd.DataFrame, num_clusters: int
) -> np.ndarray:
    """
    Apply K-means clustering to latitude and longitude coordinates and add cluster labels to a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the latitude and longitude columns.
        num_clusters (int): Number of clusters for K-means clustering.

    Returns:
        np.ndarray: Array containing cluster labels.
    """
    coordinates = dataframe[["lat", "lon"]].values

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(coordinates)


def generate_gmv_label_by_periods(
    df: pd.DataFrame, group_column: str, prefix: str
) -> pd.DataFrame:
    """
    Calculate GMV label by periods for each group (e.g., store or user).

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing transaction data.
    group_column : str
        The name of the column representing the group (e.g., 'store_id' or 'user_id').
    prefix : str
        The prefix to add to the column names.

    Returns:
    --------
    df : pandas DataFrame
        The DataFrame with GMV labels added for each period.

    Examples:
    ---------
    >>> transaction_pdf_with_revenue = generate_gmv_label_by_periods(transaction_pdf, 'store_id', 'store')
    """
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])
    last_date = df["event_occurrence"].max()
    periods = {
        f"last_month_{prefix}_gmv": 1,
        f"last_quarter_{prefix}_gmv": 3,
        f"last_half_year_{prefix}_gmv": 6,
        f"last_year_{prefix}_gmv": 12,
    }
    start_dates = {
        period: last_date - pd.offsets.DateOffset(months=months)
        for period, months in periods.items()
    }
    for period, start_date in start_dates.items():
        period_transactions = df[df["event_occurrence"] >= start_date]
        period_amount = period_transactions.groupby([group_column])["amount"].sum()
        df[period] = df[group_column].map(period_amount).fillna(0)
    for period in periods:
        df[f"{period}_label"] = create_bin_labels(df, period, GMV_BINS[prefix][period])
    return df


def generate_purchase_label_by_periods(
    df: pd.DataFrame, group_column: str, prefix: str
) -> pd.DataFrame:
    """
    Calculate purchase frequency label by periods for each group (e.g., store or user).

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing transaction data.
    group_column : str
        The name of the column representing the group (e.g., 'store_id' or 'user_id').
    prefix : str
        The prefix to add to the column names.

    Returns:
    --------
    df : pandas DataFrame
        The DataFrame with frequency labels added for each period.

    Examples:
    ---------
    >>> transaction_pdf_with_frequency = generate_purchase_label_by_periods(transaction_pdf, 'store_id', 'store', num_quantiles)
    """
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])
    last_date = df["event_occurrence"].max()
    periods = {
        f"last_month_{prefix}_purchase": 1,
        f"last_quarter_{prefix}_purchase": 3,
        f"last_half_year_{prefix}_purchase": 6,
        f"last_year_{prefix}_purchase": 12,
    }
    start_dates = {
        period: last_date - pd.offsets.DateOffset(months=months)
        for period, months in periods.items()
    }
    df["month"] = df["event_occurrence"].dt.month
    for period, start_date in start_dates.items():
        period_transactions = df[df["event_occurrence"] >= start_date]
        period_frequency = period_transactions.groupby([group_column]).size()
        df[period] = df[group_column].map(period_frequency).fillna(0)
    for period in periods:
        df[f"{period}_label"] = create_bin_labels(
            df, period, PURCHASE_BINS[prefix][period]
        )
    return df


def generate_recency_label(
    df: pd.DataFrame, group_column: str, prefix: str
) -> pd.DataFrame:
    """
    Calculate recency for users or stores based on their transaction history.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing transaction data.
    group_column : str
        The name of the column representing the group (e.g., 'user_id' or 'store_id').
    prefix : str
        The prefix to add to the column names.

    Returns
    -------
    pandas DataFrame
        DataFrame with recency calculated for each group.

    """
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])
    last_date = df["event_occurrence"].max()
    earliest_date = df["event_occurrence"].min()
    recency_data = df.groupby(group_column)["event_occurrence"].max().reset_index()
    recency_data["recency"] = (last_date - recency_data["event_occurrence"]).dt.days
    column_name = f"{prefix}_recency"
    recency_data = recency_data.set_index(group_column)["recency"]
    df[column_name] = (
        df[group_column].map(recency_data).fillna((last_date - earliest_date).days)
    )
    df[f"{column_name}_label"] = create_bin_labels(
        df, column_name, RECENCY_BINS[prefix]
    )
    return df


def calculate_transaction_age_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transaction age labels based on the difference between the last transaction date
    and the transaction occurrence date.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing transaction data.

    Returns:
    --------
    pd.DataFrame
        DataFrame with transaction age labels added.

    """
    last_transaction_date = df["event_occurrence"].max()
    df["transaction_age"] = (last_transaction_date - df["event_occurrence"]).dt.days
    df["transaction_age_label"] = create_bin_labels(
        df, "transaction_age", TRANSACTIONS_AGE_BINS
    )
    return df


def label_encode(column: pd.Series) -> pd.Series:
    """
    Encode labels in a column using scikit-learn's LabelEncoder.

    Parameters:
    -----------
    column : pd.Series
        Series containing labels to encode.

    Returns:
    --------
    pd.Series
        Encoded labels.

    """
    return pd.Series(LabelEncoder().fit_transform(column))


def trim_hour(hour: int) -> int:
    """
    Trim hour values to be within a specific range.

    Parameters:
    -----------
    hour : int
        Hour value.

    Returns:
    --------
    int
        Trimmed hour value.

    """
    if hour < 8:
        return 7
    elif hour > 16:
        return 16
    else:
        return hour


def transform_temporal_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Transform temporal features such as hour, weekday, is_weekend, season, and month.

    Parameters:
    -----------
    pdf : pd.DataFrame
        DataFrame containing temporal features.

    Returns:
    --------
    pd.DataFrame
        Transformed DataFrame with additional temporal features.

    """
    pdf["event_occurrence"] = pd.to_datetime(pdf["event_occurrence"])
    pdf["hour"] = pdf["event_occurrence"].dt.hour.apply(trim_hour)
    pdf["weekday"] = pdf["event_occurrence"].dt.weekday
    pdf["is_weekend"] = (pdf["event_occurrence"].dt.weekday >= 5) * 1
    pdf["season"] = (pdf["event_occurrence"].dt.month % 12 + 3) // 3
    pdf["month"] = pdf["event_occurrence"].dt.month
    return pdf
