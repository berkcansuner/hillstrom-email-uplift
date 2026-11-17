"""Uplift modelling: train/test split, features, T-learner, and uplift-curve evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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


def _base_model(outcome: str, seed: int):
    """Gradient boosting with conservative settings, the same for every arm.

    Shallow trees and large leaves limit how much noise each arm's model can fit;
    early stopping uses a validation split carved out of the training data only.
    """
    params = dict(
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=100,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=seed,
    )
    if outcome == "binary":
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)


class TLearner:
    """T-learner: one outcome model per arm; uplift = prediction(treated) - prediction(control).

    outcome="binary" uses classifiers and predicted probabilities (e.g. visit);
    outcome="continuous" uses regressors (e.g. spend).
    """

    def __init__(self, outcome: str, seed: int = SEED):
        if outcome not in ("binary", "continuous"):
            raise ValueError(f"outcome must be 'binary' or 'continuous', got {outcome!r}")
        self.outcome = outcome
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: np.ndarray, treated: np.ndarray) -> "TLearner":
        treated = np.asarray(treated, dtype=bool)
        y = np.asarray(y)
        self.model_t_ = _base_model(self.outcome, self.seed).fit(X[treated], y[treated])
        self.model_c_ = _base_model(self.outcome, self.seed).fit(X[~treated], y[~treated])
        return self

    def _predict(self, model, X: pd.DataFrame) -> np.ndarray:
        if self.outcome == "binary":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.model_t_, X) - self._predict(self.model_c_, X)
