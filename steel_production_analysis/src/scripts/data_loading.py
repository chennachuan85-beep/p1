"""
data_loading.py
---------------
Stage 1: Data loading
"""

import pandas as pd
import os


def load_data():
    """
    Load training and testing datasets.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data")

    train_path = os.path.join(data_dir, "normalized_train_data.csv")
    test_path = os.path.join(data_dir, "normalized_test_data.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
