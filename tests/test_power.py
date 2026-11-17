import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from hillstrom.power import mde_means, mde_proportions, required_n_means


def test_mde_matches_statsmodels():
    # statsmodels solves for the standardized effect size d = mde / sd.
    d = NormalIndPower().solve_power(nobs1=1000, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")
    assert mde_means(sd=2.0, n_t=1000, n_c=1000) == pytest.approx(2.0 * d, rel=1e-3)


def test_required_n_matches_statsmodels():
    n = NormalIndPower().solve_power(effect_size=0.1, alpha=0.05, power=0.8, ratio=1.0)
    assert required_n_means(sd=1.0, mde=0.1) == int(np.ceil(n))


def test_mde_and_required_n_are_inverse():
    n = required_n_means(sd=3.0, mde=0.25)
    assert mde_means(sd=3.0, n_t=n, n_c=n) <= 0.25
    assert mde_means(sd=3.0, n_t=n - 1, n_c=n - 1) > 0.25


def test_mde_proportions_uses_bernoulli_sd():
    assert mde_proportions(0.1, 5000, 5000) == pytest.approx(mde_means(np.sqrt(0.09), 5000, 5000))


def test_variance_reduction_shrinks_required_n_proportionally():
    # CUPED with variance reduction r: sd -> sd * sqrt(1 - r), so n -> n * (1 - r).
    n = required_n_means(sd=10.0, mde=0.1)
    n_cuped = required_n_means(sd=10.0 * np.sqrt(1 - 0.3), mde=0.1)
    assert n_cuped == pytest.approx(0.7 * n, rel=1e-3)


def test_stricter_alpha_raises_mde():
    assert mde_means(1.0, 1000, 1000, alpha=0.05 / 9) > mde_means(1.0, 1000, 1000)
