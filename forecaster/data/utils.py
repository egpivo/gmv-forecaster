import pandas as pd


def generate_negative_samples(
    full_data_pdf: pd.DataFrame, num_negative_samples: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """
    Generate negative samples for a binary classification problem.

    Examples
    --------
    >>> from forecaster.data.utils import generate_negative_samples
    >>> balanced_dataset = generate_negative_samples(full_data_pdf)

    Parameters
    ----------
    - full_data_pdf (pd.DataFrame): The DataFrame containing positive instances.
    - num_negative_samples (int, optional): The number of negative samples to generate for each user. Default is 5.
    - random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns
    -------
    - pd.DataFrame: A balanced dataset with both positive and negative samples.
    """

    # Extract unique user and store pairs from positive instances
    positive_pairs = full_data_pdf[["user_id", "store_id"]].drop_duplicates()

    # Generate a list of all possible user and store pairs
    all_user_store_pairs = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [full_data_pdf["user_id"].unique(), full_data_pdf["store_id"].unique()],
            names=["user_id", "store_id"],
        )
    ).reset_index()

    # Identify negative pairs by finding those not present in positive pairs
    negative_pairs = all_user_store_pairs[
        ~all_user_store_pairs.set_index(["user_id", "store_id"]).index.isin(
            positive_pairs.set_index(["user_id", "store_id"]).index
        )
    ]

    # Randomly sample negative stores for each user
    users = negative_pairs["user_id"].unique()
    negative_samples = pd.DataFrame(columns=["user_id", "store_id", "label"])

    for user in users:
        # Randomly sample negative stores not present in positive instances for each user
        negative_stores = negative_pairs[negative_pairs["user_id"] == user].sample(
            n=num_negative_samples, random_state=random_state
        )
        negative_samples = pd.concat([negative_samples, negative_stores])

    # Add a 'label' column indicating negative instances
    negative_samples["label"] = 0
    # Combine positive and negative samples
    balanced_dataset = pd.concat(
        [full_data_pdf[["user_id", "store_id", "label"]], negative_samples]
    )
    # Shuffle the dataset to mix positive and negative samples
    balanced_dataset = balanced_dataset.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return balanced_dataset
