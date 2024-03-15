import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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


def generate_gmv_label_by_periods(df, group_column, suffix, num_quantiles):
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
    >>> transaction_pdf_with_revenue = calculate_gmv_label_by_periods(transaction_pdf, 'store_id', 'revenue', num_quantiles)
    """

    # Convert 'event_occurrence' to datetime
    df["event_occurrence"] = pd.to_datetime(df["event_occurrence"])

    # Find the last date in the dataset
    last_date = df["event_occurrence"].max()

    # Define periods and their corresponding start dates
    periods = {"last_month": 1, "last_quarter": 3, "last_half_year": 6, "last_year": 12}
    start_dates = {
        period: last_date - pd.offsets.DateOffset(months=months)
        for period, months in periods.items()
    }

    # Extract the month from the 'event_occurrence' column
    df["month"] = df["event_occurrence"].dt.month

    # Calculate the sum of 'amount' for each group in each period
    for period, start_date in start_dates.items():
        period_transactions = df[df["event_occurrence"] >= start_date]
        period_amount = period_transactions.groupby([group_column])["amount"].sum()
        df[f"{period}_{suffix}"] = df[group_column].map(period_amount).fillna(0)

    # Create quantile labels for each period's amount
    for period in periods:
        column_name = f"{period}_{suffix}"
        df[column_name + "_label"] = create_quantile_labels(
            df, column_name, num_quantiles
        )

    return df
