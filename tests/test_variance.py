import numpy as np
import pandas as pd
import pytest

from hillstrom.variance import cuped_adjust


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
