"""
====================================================
Dataset Loader
====================================================

CSV files load garne module
"""

import pandas as pd

#File Load gareko
from config import TRAIN_CSV
from config import DEV_CSV


def load_datasets():
    """
    Training ra Development dataset load garne
    """

    # CSV file load gareko
    train_df = pd.read_csv(TRAIN_CSV)

    # CSV file load gareko
    dev_df = pd.read_csv(DEV_CSV)

    return train_df, dev_df