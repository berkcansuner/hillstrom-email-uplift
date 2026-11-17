"""Experiment validity checks: sample ratio mismatch and covariate balance."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
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
