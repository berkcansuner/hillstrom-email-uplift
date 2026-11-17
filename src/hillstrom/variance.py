"""Variance reduction with pre-treatment covariates: CUPED and regression adjustment."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from hillstrom.effects import EffectResult


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


def regression_adjusted_effect(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    ctrl: str = "control",
    covariates: list[str] | None = None,
    alpha: float = 0.05,
) -> EffectResult:
    """Treatment effect from OLS `outcome ~ treated + covariates`, with HC1 robust standard errors.

    Non-numeric covariates enter as categorical dummies. With no covariates the estimate is
    the plain difference in means. Covariates are not centered: in this additive model
    (no treated-by-covariate interactions) centering does not change the treated coefficient.
    """
    covariates = covariates or []
    sub = df[df["arm"].isin([treat, ctrl])].assign(treated=lambda d: (d["arm"] == treat).astype(int))
    terms = ["treated"] + [c if pd.api.types.is_numeric_dtype(sub[c]) else f"C({c})" for c in covariates]
    model = smf.ols(f"{outcome} ~ {' + '.join(terms)}", data=sub).fit(cov_type="HC1")
    ci_low, ci_high = model.conf_int(alpha=alpha).loc["treated"]
    is_t = sub["treated"] == 1
    return EffectResult(
        estimate=float(model.params["treated"]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(model.pvalues["treated"]),
        mean_t=float(sub.loc[is_t, outcome].mean()),
        mean_c=float(sub.loc[~is_t, outcome].mean()),
        n_t=int(is_t.sum()),
        n_c=int((~is_t).sum()),
    )
