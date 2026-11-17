"""Experiment validity checks: sample ratio mismatch and covariate balance."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    chi2: float
    p_value: float
    counts: np.ndarray
    expected: np.ndarray


def srm_test(counts: Sequence[int], expected_ratios: Sequence[float]) -> SRMResult:
    """Chi-square goodness-of-fit test of observed arm sizes against the designed split."""
    counts = np.asarray(counts, dtype=float)
    ratios = np.asarray(expected_ratios, dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"expected_ratios must sum to 1, got {ratios.sum()}")
    expected = counts.sum() * ratios
    chi2, p_value = stats.chisquare(counts, expected)
    return SRMResult(chi2=float(chi2), p_value=float(p_value), counts=counts, expected=expected)


def standardized_mean_diff(x_treat: np.ndarray, x_ctrl: np.ndarray) -> float:
    """(mean_treat - mean_ctrl) / sqrt((var_treat + var_ctrl) / 2), sample variances.

    Returns 0.0 when the variable is constant and equal in both groups.
    """
    x_treat = np.asarray(x_treat, dtype=float)
    x_ctrl = np.asarray(x_ctrl, dtype=float)
    diff = x_treat.mean() - x_ctrl.mean()
    pooled_sd = np.sqrt((x_treat.var(ddof=1) + x_ctrl.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
    return float(diff / pooled_sd)


def balance_table(
    df: pd.DataFrame, covariates: list[str], arm_col: str = "arm", control: str = "control"
) -> pd.DataFrame:
    """SMD of each covariate for every treated arm vs control.

    Non-numeric covariates are expanded into one 0/1 indicator per level, named
    "<column>=<level>". Rows are covariates (levels), columns are treated arms.
    """
    parts = []
    for col in covariates:
        if pd.api.types.is_numeric_dtype(df[col]):
            parts.append(df[[col]].astype(float))
        else:
            parts.append(pd.get_dummies(df[col], prefix=col, prefix_sep="=", dtype=float))
    X = pd.concat(parts, axis=1)

    arms = df[arm_col]
    treated_arms = sorted(a for a in arms.unique() if a != control)
    X_ctrl = X[arms == control]
    return pd.DataFrame(
        {
            arm: [standardized_mean_diff(X.loc[arms == arm, c], X_ctrl[c]) for c in X.columns]
            for arm in treated_arms
        },
        index=X.columns,
    )
