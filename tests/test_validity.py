import numpy as np
import pandas as pd
import pytest

from hillstrom.validity import balance_table, srm_test, standardized_mean_diff


def test_srm_equal_counts_p_is_one():
    result = srm_test([100, 100, 100], [1 / 3, 1 / 3, 1 / 3])
    assert result.chi2 == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_srm_detects_imbalance():
    result = srm_test([1000, 1000, 1200], [1 / 3, 1 / 3, 1 / 3])
    assert result.p_value < 0.001


def test_srm_expected_counts():
    result = srm_test([90, 210], [0.3, 0.7])
    assert np.allclose(result.expected, [90, 210])
    assert result.p_value == pytest.approx(1.0)


def test_srm_rejects_ratios_not_summing_to_one():
    with pytest.raises(ValueError):
        srm_test([100, 100], [0.5, 0.6])


# --- standardized mean differences -------------------------------------------


def test_smd_identical_is_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardized_mean_diff(x, x) == pytest.approx(0.0)


def test_smd_known_value():
    # means 1 and 0, both sample variances 2 -> pooled SD sqrt(2)
    assert standardized_mean_diff(np.array([0.0, 2.0]), np.array([-1.0, 1.0])) == pytest.approx(
        1 / np.sqrt(2)
    )


def test_smd_sign_is_treat_minus_control():
    assert standardized_mean_diff(np.array([0.0, 1.0]), np.array([1.0, 2.0])) < 0


def test_smd_constant_in_both_groups_is_zero():
    assert standardized_mean_diff(np.zeros(5), np.zeros(5)) == 0.0


def test_balance_table_expands_categoricals():
    df = pd.DataFrame(
        {
            "arm": ["control", "control", "mens", "mens", "womens", "womens"],
            "x": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0],
            "channel": ["Web", "Phone", "Web", "Phone", "Web", "Web"],
        }
    )
    table = balance_table(df, ["x", "channel"])
    assert list(table.columns) == ["mens", "womens"]
    assert list(table.index) == ["x", "channel=Phone", "channel=Web"]
    assert table.loc["x", "mens"] == pytest.approx(0.0)
    assert table.loc["x", "womens"] > 0
