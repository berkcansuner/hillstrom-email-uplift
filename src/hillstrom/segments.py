"""Treatment effects within pre-specified segments, and a test of whether they differ."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from hillstrom.effects import bootstrap_diff_ci, diff_proportions
from hillstrom.paths import SEED


def effects_by_segment(
    df: pd.DataFrame,
    treat: str,
    ctrl: str,
    metric: str,
    by: str,
    n_boot: int = 10_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Effect of `treat` vs `ctrl` on `metric` within each level of `by`, one row per level.

    0/1 metrics: difference in proportions with Wald CI and p-value.
    Other metrics (spend): difference in means with percentile bootstrap CI and no
    p-value, since small segments have few buyers and the normal approximation is weaker.
    """
    binary = df[metric].isin([0, 1]).all()
    rows = []
    for level, g in df.groupby(by, sort=True):
        y_t = g.loc[g["arm"] == treat, metric].to_numpy()
        y_c = g.loc[g["arm"] == ctrl, metric].to_numpy()
        if binary:
            r = diff_proportions(y_t, y_c)
            estimate, ci_low, ci_high, p_value, method = r.estimate, r.ci_low, r.ci_high, r.p_value, "wald"
        else:
            estimate = y_t.mean() - y_c.mean()
            ci_low, ci_high = bootstrap_diff_ci(y_t, y_c, n_boot=n_boot, seed=seed)
            p_value, method = np.nan, "bootstrap"
        rows.append(
            {
                "level": level,
                "n_t": len(y_t),
                "n_c": len(y_c),
                "mean_t": y_t.mean(),
                "mean_c": y_c.mean(),
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "ci_method": method,
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_test(df: pd.DataFrame, treat: str, ctrl: str, metric: str, by: str) -> float:
    """p-value for "the effect is the same in every level of `by`".

    Fits OLS `metric ~ treated * C(by)` with HC1 robust standard errors on the two arms
    and jointly tests all treated-by-segment interaction terms with a Wald test.
    """
    sub = df[df["arm"].isin([treat, ctrl])].assign(treated=lambda d: (d["arm"] == treat).astype(int))
    model = smf.ols(f"{metric} ~ treated * C({by})", data=sub).fit(cov_type="HC1")
    names = list(model.params.index)
    interactions = [i for i, name in enumerate(names) if name.startswith("treated:")]
    restriction = np.zeros((len(interactions), len(names)))
    for row, col in enumerate(interactions):
        restriction[row, col] = 1.0
    return float(model.wald_test(restriction, scalar=True).pvalue)
