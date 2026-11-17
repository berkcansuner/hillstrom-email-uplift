import numpy as np
import pytest

from hillstrom.validity import srm_test


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
