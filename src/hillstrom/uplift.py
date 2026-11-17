"""Uplift modelling: train/test split, features, T-learner, and uplift-curve evaluation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hillstrom.paths import SEED

NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
# Fixed levels so train, test and any subset always get the same columns.
CATEGORICAL_FEATURES = {
    "zip_code": ["Rural", "Suburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
}
# history_segment and purchase_group are left out: they are coarser versions of
# history and of mens/womens, which are already included.


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split, stratified by arm so each part keeps the 1/3 split."""
    train, test = train_test_split(df, test_size=test_size, stratify=df["arm"], random_state=seed)
    return train, test


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment feature matrix: numeric columns plus one 0/1 column per category level."""
    dummies = [
        pd.get_dummies(pd.Categorical(df[col], categories=levels), prefix=col, prefix_sep="=", dtype=float)
        .set_axis(df.index)
        for col, levels in CATEGORICAL_FEATURES.items()
    ]
    return pd.concat([df[NUMERIC_FEATURES].astype(float), *dummies], axis=1)
