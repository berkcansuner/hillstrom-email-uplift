"""Variance reduction with pre-treatment covariates: CUPED and regression adjustment."""

import numpy as np


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: remove the part of y explained by a pre-treatment covariate x.

    y_adj = y - theta * (x - mean(x)), with theta = cov(y, x) / var(x) estimated on all
    customers pooled across arms. Because x is measured before treatment, it cannot be
    affected by it, so differences in mean y_adj between arms estimate the same effect
    as differences in mean y, with variance reduced by a factor of (1 - corr(x, y)^2).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)
