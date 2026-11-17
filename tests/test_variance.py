import numpy as np
import pandas as pd
import pytest

from hillstrom.effects import diff_means_welch
from hillstrom.variance import cuped_adjust, regression_adjusted_effect


def _experiment(n: int = 20_000, rho_strength: float = 2.0, effect: float = 0.5, seed: int = 0):
    """Randomized two-arm data where a pre-treatment x predicts y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(10, 2, n)
    treated = rng.integers(0, 2, n).astype(bool)
    y = rho_strength * x + effect * treated + rng.normal(0, 3, n)
    return x, y, treated


def test_cuped_theta_is_regression_slope():
    x, y, _ = _experiment()
    _, theta = cuped_adjust(y, x)
    assert theta == pytest.approx(2.0, abs=0.05)


def test_cuped_keeps_the_mean_and_the_effect():
    x, y, treated = _experiment()
    y_adj, _ = cuped_adjust(y, x)
    assert y_adj.mean() == pytest.approx(y.mean())
    raw = y[treated].mean() - y[~treated].mean()
    adjusted = y_adj[treated].mean() - y_adj[~treated].mean()
    assert adjusted == pytest.approx(0.5, abs=0.06)
    assert adjusted == pytest.approx(raw, abs=0.15)


def test_cuped_variance_reduction_equals_rho_squared():
    x, y, _ = _experiment()
    y_adj, _ = cuped_adjust(y, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert 1 - y_adj.var() / y.var() == pytest.approx(rho**2, abs=0.01)


def test_cuped_with_unrelated_covariate_changes_little():
    rng = np.random.default_rng(1)
    y = rng.normal(size=5000)
    y_adj, theta = cuped_adjust(y, rng.normal(size=5000))
    assert abs(theta) < 0.05
    assert y_adj.var() == pytest.approx(y.var(), rel=0.01)


# --- regression adjustment ---------------------------------------------------


def _frame(seed: int = 0) -> pd.DataFrame:
    x, y, treated = _experiment(seed=seed)
    rng = np.random.default_rng(seed + 1)
    return pd.DataFrame(
        {
            "arm": np.where(treated, "mens", "control"),
            "y": y,
            "x": x,
            "channel": rng.choice(["Web", "Phone"], len(y)),
        }
    )


def test_regression_without_covariates_equals_raw_difference():
    df = _frame()
    result = regression_adjusted_effect(df, "y", "mens", covariates=[])
    welch = diff_means_welch(df.loc[df.arm == "mens", "y"], df.loc[df.arm == "control", "y"])
    assert result.estimate == pytest.approx(welch.estimate)
    assert result.ci_high - result.ci_low == pytest.approx(welch.ci_high - welch.ci_low, rel=0.01)


def test_regression_with_predictive_covariate_narrows_ci_and_stays_unbiased():
    df = _frame()
    raw = regression_adjusted_effect(df, "y", "mens", covariates=[])
    adjusted = regression_adjusted_effect(df, "y", "mens", covariates=["x"])
    assert adjusted.estimate == pytest.approx(0.5, abs=0.1)
    assert (adjusted.ci_high - adjusted.ci_low) < 0.8 * (raw.ci_high - raw.ci_low)


def test_regression_accepts_categorical_covariates():
    df = _frame()
    result = regression_adjusted_effect(df, "y", "mens", covariates=["x", "channel"])
    assert result.ci_low < 0.5 < result.ci_high
    assert (result.n_t, result.n_c) == ((df.arm == "mens").sum(), (df.arm == "control").sum())
