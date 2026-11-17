import numpy as np
import pytest
from scipy import stats

from hillstrom.effects import bootstrap_diff_ci, diff_means_welch, diff_proportions, holm_adjust


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


def test_diff_means_welch_matches_manual_formula():
    rng = np.random.default_rng(1)
    y_t = rng.normal(1.0, 2.0, 300)
    y_c = rng.normal(0.5, 1.0, 500)
    result = diff_means_welch(y_t, y_c)

    v_t, v_c = y_t.var(ddof=1) / 300, y_c.var(ddof=1) / 500
    se = np.sqrt(v_t + v_c)
    df = (v_t + v_c) ** 2 / (v_t**2 / 299 + v_c**2 / 499)
    t_crit = stats.t.ppf(0.975, df)
    estimate = y_t.mean() - y_c.mean()

    assert result.estimate == pytest.approx(estimate)
    assert result.ci_low == pytest.approx(estimate - t_crit * se)
    assert result.ci_high == pytest.approx(estimate + t_crit * se)
    assert result.p_value == pytest.approx(2 * stats.t.sf(abs(estimate) / se, df))


def _zero_inflated_spend(rng, n, buy_rate, scale):
    """Mostly zeros, lognormal amounts for buyers: mean = buy_rate * scale * exp(0.5)."""
    buys = rng.random(n) < buy_rate
    return np.where(buys, scale * rng.lognormal(0.0, 1.0, n), 0.0)


def test_bootstrap_same_seed_identical():
    rng = np.random.default_rng(2)
    y_t = _zero_inflated_spend(rng, 2000, 0.05, 100)
    y_c = _zero_inflated_spend(rng, 2000, 0.03, 100)
    assert bootstrap_diff_ci(y_t, y_c, n_boot=2000, seed=7) == bootstrap_diff_ci(
        y_t, y_c, n_boot=2000, seed=7
    )


def test_bootstrap_ci_covers_true_diff_and_matches_welch():
    rng = np.random.default_rng(3)
    y_t = _zero_inflated_spend(rng, 20_000, 0.02, 100)
    y_c = _zero_inflated_spend(rng, 20_000, 0.01, 100)
    true_diff = (0.02 - 0.01) * 100 * np.exp(0.5)

    low, high = bootstrap_diff_ci(y_t, y_c, n_boot=2000, seed=0)
    assert low < true_diff < high

    welch = diff_means_welch(y_t, y_c)
    width = welch.ci_high - welch.ci_low
    assert abs(low - welch.ci_low) < 0.1 * width
    assert abs(high - welch.ci_high) < 0.1 * width


def test_holm_matches_manual():
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert np.allclose(holm_adjust([0.5, 0.9]), [1.0, 1.0])
