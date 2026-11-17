"""Power analysis for two-arm comparisons of means and proportions (normal approximation)."""

import numpy as np
from scipy import stats


def _z_total(alpha: float, power: float) -> float:
    """z_{1 - alpha/2} + z_{power}: two-sided test at level alpha with the given power."""
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def mde_means(sd: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable difference in means: (z_{1-a/2} + z_power) * sd * sqrt(1/n_t + 1/n_c)."""
    return float(_z_total(alpha, power) * sd * np.sqrt(1 / n_t + 1 / n_c))


def mde_proportions(p_base: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable difference in proportions, using the baseline rate's SD sqrt(p(1-p))."""
    return mde_means(np.sqrt(p_base * (1 - p_base)), n_t, n_c, alpha, power)


def required_n_means(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers needed per arm (equal arms) to detect a difference of `mde` in means."""
    return int(np.ceil(2 * (_z_total(alpha, power) * sd / mde) ** 2))
