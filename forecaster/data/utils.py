import numpy as np
import pandas as pd


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
