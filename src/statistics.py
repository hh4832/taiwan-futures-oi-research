import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def hac_lag(signal_window: int, outcome_horizon: int) -> int:
    """Conservative lower bound for overlapping signal and outcome windows."""
    return max(signal_window - 1, outcome_horizon - 1)


def _cohens_d(group: pd.Series, nongroup: pd.Series) -> float:
    n1, n0 = len(group), len(nongroup)
    if n1 < 2 or n0 < 2:
        return np.nan
    pooled_var = (
        (n1 - 1) * group.var(ddof=1) + (n0 - 1) * nongroup.var(ddof=1)
    ) / (n1 + n0 - 2)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan
    return float((group.mean() - nongroup.mean()) / np.sqrt(pooled_var))


def compare_group_to_nongroup(
    group_values: pd.Series,
    target: pd.Series,
    maxlags: int,
) -> dict:
    """Compare a group with its non-group using chronological HAC regression."""
    work = pd.DataFrame({"group": group_values, "target": target}).dropna()
    work["group"] = work["group"].astype(int)
    group = work.loc[work["group"].eq(1), "target"].astype(float)
    nongroup = work.loc[work["group"].eq(0), "target"].astype(float)
    result = {
        "n": len(group),
        "n_nongroup": len(nongroup),
        "positive_rate": float(group.gt(0).mean()) if len(group) else np.nan,
        "mean_return": float(group.mean()) if len(group) else np.nan,
        "median_return": float(group.median()) if len(group) else np.nan,
        "q1_return": float(group.quantile(0.25)) if len(group) else np.nan,
        "q3_return": float(group.quantile(0.75)) if len(group) else np.nan,
        "std_return": float(group.std(ddof=1)) if len(group) > 1 else np.nan,
        "mean_nongroup": float(nongroup.mean()) if len(nongroup) else np.nan,
        "positive_rate_nongroup": float(nongroup.gt(0).mean()) if len(nongroup) else np.nan,
        "cohens_d": _cohens_d(group, nongroup),
        "hac_lag": int(maxlags),
    }
    result["mean_diff_vs_nongroup"] = result["mean_return"] - result["mean_nongroup"]
    result["positive_rate_diff_vs_nongroup"] = (
        result["positive_rate"] - result["positive_rate_nongroup"]
    )
    if len(group) < 5 or len(nongroup) < 5:
        result.update(
            welch_t=np.nan, welch_p_value=np.nan, hac_coef=np.nan,
            hac_se=np.nan, hac_t=np.nan, hac_p_value=np.nan,
            hac_ci_low=np.nan, hac_ci_high=np.nan,
        )
        return result

    welch = stats.ttest_ind(group, nongroup, equal_var=False, nan_policy="omit")
    X = sm.add_constant(work["group"].astype(float))
    model = sm.OLS(work["target"].astype(float), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    coef = float(model.params["group"])
    se = float(model.bse["group"])
    result.update(
        welch_t=float(welch.statistic),
        welch_p_value=float(welch.pvalue),
        hac_coef=coef,
        hac_se=se,
        hac_t=float(model.tvalues["group"]),
        hac_p_value=float(model.pvalues["group"]),
        hac_ci_low=coef - 1.96 * se,
        hac_ci_high=coef + 1.96 * se,
    )
    return result


def grouped_comparisons(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    bins,
    labels,
    maxlags: int,
) -> pd.DataFrame:
    work = df[[score_col, target_col]].copy()
    groups = pd.cut(
        work[score_col], bins=bins, labels=labels, include_lowest=True, right=True
    )
    rows = []
    valid_target = work[target_col].dropna()
    unconditional_mean = float(valid_target.mean()) if len(valid_target) else np.nan
    unconditional_positive = float(valid_target.gt(0).mean()) if len(valid_target) else np.nan
    for label in labels:
        indicator = groups.eq(label)
        row = compare_group_to_nongroup(indicator, work[target_col], maxlags)
        row.update(
            group=str(label),
            unconditional_mean=unconditional_mean,
            unconditional_positive_rate=unconditional_positive,
            mean_diff_vs_unconditional=row["mean_return"] - unconditional_mean,
            positive_rate_diff_vs_unconditional=(
                row["positive_rate"] - unconditional_positive
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_fdr_by_family(
    results: pd.DataFrame,
    family_cols=("fdr_family",),
    p_col="hac_p_value",
) -> pd.DataFrame:
    out = results.copy()
    out["q_value_bh"] = np.nan
    for _, index in out.groupby(list(family_cols), dropna=False).groups.items():
        valid = out.loc[index, p_col].dropna()
        if len(valid):
            out.loc[valid.index, "q_value_bh"] = multipletests(
                valid.to_numpy(), method="fdr_bh"
            )[1]
    return out
