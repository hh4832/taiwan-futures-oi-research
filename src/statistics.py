import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def hac_mean_test(series: pd.Series, maxlags: int = 5) -> dict:
    y = series.dropna().astype(float)
    if len(y) < 10:
        return {
            "n": len(y), "mean": np.nan, "se_hac": np.nan,
            "t": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan
        }

    X = np.ones((len(y), 1))
    model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    beta = float(model.params[0])
    se = float(model.bse[0])
    return {
        "n": len(y),
        "mean": beta,
        "se_hac": se,
        "t": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
    }


def group_by_percentile(
    df: pd.DataFrame,
    predictor_pr_col: str,
    target_col: str,
    bins=(0.0, 0.2, 0.8, 1.0),
    labels=("PR_0_20", "PR_20_80", "PR_80_100"),
    maxlags: int = 5,
) -> pd.DataFrame:
    work = df[[predictor_pr_col, target_col]].dropna().copy()
    work["group"] = pd.cut(
        work[predictor_pr_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    for label in labels:
        s = work.loc[work["group"] == label, target_col]
        r = hac_mean_test(s, maxlags=maxlags)
        r["group"] = label
        rows.append(r)
    return pd.DataFrame(rows)


def add_fdr_qvalues(results: pd.DataFrame, p_col="p_value") -> pd.DataFrame:
    out = results.copy()
    mask = out[p_col].notna()
    out["q_value_bh"] = np.nan
    if mask.any():
        out.loc[mask, "q_value_bh"] = multipletests(
            out.loc[mask, p_col].values,
            method="fdr_bh",
        )[1]
    return out
