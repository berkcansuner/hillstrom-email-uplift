"""Uplift modelling: train/test split, features, T-learner, and uplift-curve evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def _linear_model(outcome: str):
    """Logistic regression (L2, default strength) for binary outcomes, OLS for continuous ones.

    Features are standardized so the L2 penalty treats them equally.
    """
    if outcome == "binary":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return make_pipeline(StandardScaler(), LinearRegression())


def _gbm_model(outcome: str, seed: int):
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
    base="gbm" uses gradient boosting, base="linear" logistic/linear regression.
    """

    def __init__(self, outcome: str, base: str = "gbm", seed: int = SEED):
        if outcome not in ("binary", "continuous"):
            raise ValueError(f"outcome must be 'binary' or 'continuous', got {outcome!r}")
        if base not in ("gbm", "linear"):
            raise ValueError(f"base must be 'gbm' or 'linear', got {base!r}")
        self.outcome = outcome
        self.base = base
        self.seed = seed

    def _new_model(self):
        if self.base == "linear":
            return _linear_model(self.outcome)
        return _gbm_model(self.outcome, self.seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray, treated: np.ndarray) -> "TLearner":
        treated = np.asarray(treated, dtype=bool)
        y = np.asarray(y)
        self.model_t_ = self._new_model().fit(X[treated], y[treated])
        self.model_c_ = self._new_model().fit(X[~treated], y[~treated])
        return self

    def _predict(self, model, X: pd.DataFrame) -> np.ndarray:
        if self.outcome == "binary":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.model_t_, X) - self._predict(self.model_c_, X)


def uplift_curve(y: np.ndarray, uplift: np.ndarray, treated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uplift curve: incremental outcome from targeting the top-k customers by predicted uplift.

    For the k customers with the highest predicted uplift,
        gain(k) = (mean outcome of treated among them - mean outcome of control among them) * k.
    Ties in `uplift` are kept together (one point per distinct score). Works for binary
    and continuous outcomes; on binary outcomes it equals `sklift.metrics.uplift_curve`
    (checked in the tests). Returns (k values starting at 0, gains starting at 0).
    """
    y = np.asarray(y, dtype=float)
    uplift = np.asarray(uplift, dtype=float)
    treated = np.asarray(treated, dtype=float)

    order = np.argsort(uplift, kind="mergesort")[::-1]
    y, uplift, treated = y[order], uplift[order], treated[order]

    last_of_each_score = np.r_[np.where(np.diff(uplift))[0], len(uplift) - 1]
    n_all = last_of_each_score + 1
    n_t = np.cumsum(treated)[last_of_each_score]
    n_c = n_all - n_t
    y_t = np.cumsum(y * treated)[last_of_each_score]
    y_c = np.cumsum(y * (1 - treated))[last_of_each_score]

    mean_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t != 0)
    mean_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c != 0)
    return np.r_[0, n_all], np.r_[0.0, (mean_t - mean_c) * n_all]


def gain_over_random(y: np.ndarray, uplift: np.ndarray, treated: np.ndarray) -> float:
    """Area between the uplift curve and random targeting, per customer.

    Both axes are divided by the number of customers (x = share targeted, y = incremental
    outcome per customer), so the result is in outcome units: the average extra outcome per
    customer, across all targeting depths, from ranking by the model instead of at random.
    Positive = better than random; 0 = no better than random.
    """
    n_targeted, gain = uplift_curve(y, uplift, treated)
    n = n_targeted[-1]
    share, gain_per_customer = n_targeted / n, gain / n
    random_line = share * gain_per_customer[-1]
    return float(np.trapezoid(gain_per_customer - random_line, share))


def bootstrap_gain_over_random(
    y: np.ndarray, uplift: np.ndarray, treated: np.ndarray, n_boot: int = 500, seed: int = SEED
) -> np.ndarray:
    """Bootstrap draws of `gain_over_random`, resampling customers with predictions held fixed.

    Captures sampling uncertainty of the evaluation set, not variation from re-training the
    model. The same seed gives the same resamples, so draws for two different score vectors
    are paired and their difference can be summarised directly.
    """
    y, uplift, treated = np.asarray(y), np.asarray(uplift), np.asarray(treated)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = gain_over_random(y[idx], uplift[idx], treated[idx])
    return draws
