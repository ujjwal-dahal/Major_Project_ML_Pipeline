"""
====================================================
Dataset Exploration
====================================================

Dataset ko basic information print garne
"""

import pandas as pd


def explore_dataset(df, dataset_name="Dataset"):
    """
    Dataset ko summary print garne
    """

    print("\n")
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    # Dataset shape print gareko
    print(f"\nShape : {df.shape}")

    # First 5 rows print gareko
    print("\nFirst Five Rows\n")
    print(df.head())

    # Data type print gareko
    print("\nData Types\n")
    print(df.dtypes)

    # Missing values count gareko
    print("\nMissing Values\n")
    print(df.isnull().sum())

    # Duplicate rows count gareko
    print("\nDuplicate Rows")

    print(df.duplicated().sum())

    # Label distribution print gareko
    print("\nLabel Distribution\n")

    print(df["label"].value_counts())

    # Total feature count
    feature_count = len(df.columns) - 2

    print(f"\nTotal Features : {feature_count}")