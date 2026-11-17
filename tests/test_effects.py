import numpy as np
import pytest
from scipy import stats

from hillstrom.effects import diff_proportions


def test_diff_proportions_known():
    y_t = np.r_[np.ones(200), np.zeros(800)]  # 0.2
    y_c = np.r_[np.ones(100), np.zeros(900)]  # 0.1
    result = diff_proportions(y_t, y_c)
    se = np.sqrt(0.2 * 0.8 / 1000 + 0.1 * 0.9 / 1000)
    z = stats.norm.ppf(0.975)
    assert result.estimate == pytest.approx(0.1)
    assert result.ci_low == pytest.approx(0.1 - z * se)
    assert result.ci_high == pytest.approx(0.1 + z * se)
    assert result.p_value == pytest.approx(2 * stats.norm.sf(0.1 / se))
    assert (result.n_t, result.n_c) == (1000, 1000)
    assert (result.mean_t, result.mean_c) == (pytest.approx(0.2), pytest.approx(0.1))


def test_diff_proportions_ci_and_p_agree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_t = rng.binomial(1, 0.12, 500)
        y_c = rng.binomial(1, 0.10, 500)
        r = diff_proportions(y_t, y_c)
        assert (r.p_value < 0.05) == (r.ci_low > 0 or r.ci_high < 0)


def test_diff_proportions_zero_events():
    result = diff_proportions(np.zeros(50), np.zeros(80))
    assert result.estimate == 0.0
    assert np.isfinite([result.ci_low, result.ci_high, result.p_value]).all()
    assert result.p_value == 1.0
