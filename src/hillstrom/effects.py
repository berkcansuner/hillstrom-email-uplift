"""Treatment effect estimates (difference treated minus control) with uncertainty."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class EffectResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    mean_t: float
    mean_c: float
    n_t: int
    n_c: int


def _z_result(y_t: np.ndarray, y_c: np.ndarray, se: float, alpha: float) -> EffectResult:
    """Normal-approximation CI and two-sided p-value for mean(y_t) - mean(y_c)."""
    mean_t, mean_c = float(y_t.mean()), float(y_c.mean())
    estimate = mean_t - mean_c
    z = stats.norm.ppf(1 - alpha / 2)
    if se == 0:
        # No variation in either arm (e.g. zero events): the difference is exact.
        p_value = 1.0 if estimate == 0 else 0.0
    else:
        p_value = float(2 * stats.norm.sf(abs(estimate) / se))
    return EffectResult(
        estimate=estimate,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
        p_value=p_value,
        mean_t=mean_t,
        mean_c=mean_c,
        n_t=len(y_t),
        n_c=len(y_c),
    )


def diff_proportions(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> EffectResult:
    """Difference in proportions of 0/1 outcomes: Wald CI and z-test, both with unpooled SE.

    Using the same (unpooled) SE for the CI and the p-value keeps them consistent:
    the CI excludes 0 exactly when p < alpha.
    """
    y_t = np.asarray(y_t, dtype=float)
    y_c = np.asarray(y_c, dtype=float)
    p_t, p_c = y_t.mean(), y_c.mean()
    se = float(np.sqrt(p_t * (1 - p_t) / len(y_t) + p_c * (1 - p_c) / len(y_c)))
    return _z_result(y_t, y_c, se, alpha)


def diff_means_welch(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> EffectResult:
    """Difference in means with Welch's t-test (unequal variances) and its t-based CI."""
    y_t = np.asarray(y_t, dtype=float)
    y_c = np.asarray(y_c, dtype=float)
    test = stats.ttest_ind(y_t, y_c, equal_var=False)
    ci = test.confidence_interval(confidence_level=1 - alpha)
    return EffectResult(
        estimate=float(y_t.mean() - y_c.mean()),
        ci_low=float(ci.low),
        ci_high=float(ci.high),
        p_value=float(test.pvalue),
        mean_t=float(y_t.mean()),
        mean_c=float(y_c.mean()),
        n_t=len(y_t),
        n_c=len(y_c),
    )
