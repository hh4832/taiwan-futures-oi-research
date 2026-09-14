"""Frozen v4 comparison of Foreign and Foreign–Dealer horizon retention."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .config import CONFIG, ResearchConfig
from .incremental_decay import INTERVAL_HORIZONS
from .institutional_divergence import assign_divergence_bucket
from .statistics import compare_group_to_nongroup, hac_lag


DAY1_EFFECT_EPSILON = 1e-12
PREDICTORS = ("Foreign", "Foreign-Dealer divergence")
SIDES = ("bearish", "bullish")


def decay_ratio(effect: float, day1_effect: float) -> float:
    """Absolute within-predictor retention; tiny Day-1 effects are not evaluable."""
    if not np.isfinite(effect) or not np.isfinite(day1_effect):
        return np.nan
    if abs(day1_effect) <= DAY1_EFFECT_EPSILON:
        return np.nan
    return abs(effect) / abs(day1_effect)


def _signal_masks(frame: pd.DataFrame, accumulation: int, rolling: int):
    foreign = frame[f"foreign_pr_{accumulation}d_{rolling}d"]
    divergence = frame[f"fd_divergence_{accumulation}d_{rolling}d"]
    buckets = assign_divergence_bucket(divergence)
    return {
        ("Foreign", "bearish"): foreign.le(.20).where(foreign.notna()),
        ("Foreign", "bullish"): foreign.ge(.80).where(foreign.notna()),
        ("Foreign-Dealer divergence", "bearish"): (
            buckets.eq("Strong bearish").where(buckets.notna())
        ),
        ("Foreign-Dealer divergence", "bullish"): (
            buckets.eq("Strong bullish").where(buckets.notna())
        ),
    }


def _apply_fdr(results: pd.DataFrame) -> pd.DataFrame:
    """Independent v4 universes: 120 fixed hypotheses for each side."""
    out = results.copy()
    out["q_value_family"] = np.nan
    out["q_value_global"] = np.nan
    for _, index in out.groupby(["side", "interval"], sort=False).groups.items():
        valid = out.loc[index, "hac_p_value"].dropna()
        if len(valid):
            out.loc[valid.index, "q_value_family"] = multipletests(
                valid.to_numpy(), method="fdr_bh"
            )[1]
    for _, index in out.groupby("side", sort=False).groups.items():
        valid = out.loc[index, "hac_p_value"].dropna()
        if len(valid):
            out.loc[valid.index, "q_value_global"] = multipletests(
                valid.to_numpy(), method="fdr_bh"
            )[1]
    out["evidence_level"] = np.select(
        [
            out["q_value_global"].lt(.05),
            out["q_value_family"].lt(.05),
            out["hac_p_value"].lt(.05),
        ],
        ["Level A", "Level B", "Level C"],
        default="Level D",
    )
    out.loc[out["hac_p_value"].isna(), "evidence_level"] = "Not evaluable"
    return out


def _add_decay(results: pd.DataFrame, effect_col: str, output_col: str) -> pd.DataFrame:
    out = results.copy()
    keys = ["predictor", "accumulation_window", "rolling_window"]
    if "side" in out.columns:
        keys.insert(1, "side")
    day1 = (
        out.loc[out["interval"].eq("day_1"), keys + [effect_col]]
        .rename(columns={effect_col: "day1_effect"})
    )
    out = out.merge(day1, on=keys, how="left", validate="many_to_one")
    out[output_col] = [
        decay_ratio(effect, baseline)
        for effect, baseline in zip(out[effect_col], out["day1_effect"])
    ]
    return out


def analyze_horizon_extension(
    frame: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> pd.DataFrame:
    """Run the parallel 2-predictor × 2-side × 12-cell × 5-interval grid."""
    rows = []
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            for (predictor, side), mask in _signal_masks(
                frame, accumulation, rolling
            ).items():
                for interval, endpoint in INTERVAL_HORIZONS.items():
                    target = frame[interval]
                    row = compare_group_to_nongroup(
                        mask, target, hac_lag(accumulation, endpoint)
                    )
                    valid = pd.DataFrame({"mask": mask, "target": target}).dropna()
                    nongroup = valid.loc[~valid["mask"].astype(bool), "target"]
                    row["median_nongroup"] = (
                        float(nongroup.median()) if len(nongroup) else np.nan
                    )
                    row.update({
                        "analysis_role": "confirmatory_horizon_extension",
                        "fdr_universe": f"v4_horizon_extension_{side}_120",
                        "predictor": predictor,
                        "side": side,
                        "accumulation_window": accumulation,
                        "rolling_window": rolling,
                        "parameter_cell": f"{accumulation}D_x_{rolling}D",
                        "interval": interval,
                        "interval_end_horizon": endpoint,
                    })
                    rows.append(row)
    results = _apply_fdr(pd.DataFrame(rows))
    results = _add_decay(
        results, "mean_diff_vs_nongroup", "decay_ratio"
    )
    results["expected_direction"] = np.where(
        results["side"].eq("bearish"),
        results["mean_diff_vs_nongroup"].lt(0),
        results["mean_diff_vs_nongroup"].gt(0),
    )
    return results


def _continuous_row(
    target: pd.Series,
    foreign: pd.Series,
    divergence: pd.Series,
    predictor: str,
    accumulation: int,
    rolling: int,
    interval: str,
    endpoint: int,
) -> dict:
    work = pd.concat(
        [target.rename("target"), foreign.rename("foreign"), divergence.rename("divergence")],
        axis=1,
    ).dropna()
    source = work["foreign"] if predictor == "Foreign" else work["divergence"]
    standardized = (source - source.mean()) / source.std(ddof=1)
    base = {
        "analysis_role": "secondary_same_sample_mechanism",
        "predictor": predictor,
        "accumulation_window": accumulation,
        "rolling_window": rolling,
        "parameter_cell": f"{accumulation}D_x_{rolling}D",
        "interval": interval,
        "interval_end_horizon": endpoint,
        "n": len(work),
        "hac_lag": hac_lag(accumulation, endpoint),
    }
    if len(work) < 20 or not np.isfinite(standardized.std(ddof=1)):
        return dict(base, standardized_beta=np.nan, hac_se=np.nan, hac_t=np.nan,
                    hac_p_value=np.nan, hac_ci_low=np.nan, hac_ci_high=np.nan,
                    r_squared=np.nan)
    model = sm.OLS(
        work["target"].astype(float), sm.add_constant(standardized.astype(float))
    ).fit(cov_type="HAC", cov_kwds={"maxlags": base["hac_lag"]})
    beta = float(model.params[source.name])
    se = float(model.bse[source.name])
    return dict(
        base, standardized_beta=beta, hac_se=se,
        hac_t=float(model.tvalues[source.name]),
        hac_p_value=float(model.pvalues[source.name]),
        hac_ci_low=beta - 1.96 * se, hac_ci_high=beta + 1.96 * se,
        r_squared=float(model.rsquared),
    )


def continuous_horizon_regressions(
    frame: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> pd.DataFrame:
    """Compare standardized coefficients on identical date samples."""
    rows = []
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            foreign = frame[f"foreign_pr_{accumulation}d_{rolling}d"]
            divergence = frame[f"fd_divergence_{accumulation}d_{rolling}d"]
            for interval, endpoint in INTERVAL_HORIZONS.items():
                for predictor in PREDICTORS:
                    rows.append(_continuous_row(
                        frame[interval], foreign, divergence, predictor,
                        accumulation, rolling, interval, endpoint,
                    ))
    results = _add_decay(
        pd.DataFrame(rows), "standardized_beta", "beta_retention_ratio"
    )
    return results


def _side_summary(results: pd.DataFrame, side: str) -> pd.DataFrame:
    subset = results.loc[results["side"].eq(side)]
    return (
        subset.groupby(["predictor", "interval"], sort=False, as_index=False)
        .agg(
            parameter_cells=("parameter_cell", "count"),
            median_effect=("mean_diff_vs_nongroup", "median"),
            median_decay_ratio=("decay_ratio", "median"),
            min_effect=("mean_diff_vs_nongroup", "min"),
            max_effect=("mean_diff_vs_nongroup", "max"),
            direction_consistency_cells=("expected_direction", "sum"),
            global_fdr_cells=("q_value_global", lambda s: int(s.lt(.05).sum())),
            median_n=("n", "median"),
            median_cohens_d=("cohens_d", "median"),
        )
    )


def _classification(row: pd.Series) -> str:
    if row["interval"] == "day_1":
        return "Not evaluable"
    required = ["foreign_median_decay_ratio", "divergence_median_decay_ratio"]
    if any(not np.isfinite(row[name]) for name in required):
        return "Not evaluable"
    expected = row["divergence_median_effect"] < 0 if row["side"] == "bearish" else row["divergence_median_effect"] > 0
    retention_better = row["divergence_median_decay_ratio"] > row["foreign_median_decay_ratio"]
    direction_ok = row["divergence_direction_cells"] >= row["foreign_direction_cells"]
    fdr_better = row["divergence_global_fdr_cells"] > row["foreign_global_fdr_cells"]
    if expected and retention_better and direction_ok and fdr_better:
        return "Yes"
    if (not expected) or (not retention_better) or (
        row["divergence_direction_cells"] < row["foreign_direction_cells"]
        and row["divergence_global_fdr_cells"] < row["foreign_global_fdr_cells"]
    ):
        return "No"
    return "Mixed"


def summarize_horizon_extension(results: pd.DataFrame):
    bearish = _side_summary(results, "bearish")
    bullish = _side_summary(results, "bullish")
    combined = pd.concat([bearish.assign(side="bearish"), bullish.assign(side="bullish")])
    foreign = combined.loc[combined["predictor"].eq("Foreign")].set_index(["side", "interval"])
    divergence = combined.loc[
        combined["predictor"].eq("Foreign-Dealer divergence")
    ].set_index(["side", "interval"])
    comparison = foreign.add_prefix("foreign_").join(
        divergence.add_prefix("divergence_"), how="outer"
    ).reset_index()
    comparison = comparison.rename(columns={
        "foreign_direction_consistency_cells": "foreign_direction_cells",
        "divergence_direction_consistency_cells": "divergence_direction_cells",
    })
    comparison["horizon_extension_supported"] = comparison.apply(_classification, axis=1)
    return bearish, bullish, comparison


def summarize_by_accumulation(results: pd.DataFrame) -> pd.DataFrame:
    """Surface summary across all three rolling windows; 3D is post-hoc."""
    out = (
        results.groupby(
            ["predictor", "side", "accumulation_window", "interval"],
            sort=False, as_index=False,
        )
        .agg(
            rolling_cells=("rolling_window", "count"),
            median_effect=("mean_diff_vs_nongroup", "median"),
            median_decay_ratio=("decay_ratio", "median"),
            direction_consistency_cells=("expected_direction", "sum"),
            global_fdr_cells=("q_value_global", lambda s: int(s.lt(.05).sum())),
            median_n=("n", "median"),
        )
    )
    out["interpretation"] = np.where(
        out["accumulation_window"].eq(3),
        "post-hoc mechanistic observation",
        "pre-specified surface context",
    )
    return out


def overall_horizon_answer(comparison: pd.DataFrame) -> str:
    post_day1 = comparison.loc[~comparison["interval"].eq("day_1"), "horizon_extension_supported"]
    if post_day1.eq("Not evaluable").all():
        return "Unable to determine"
    if post_day1.eq("Yes").any():
        return "Yes" if not post_day1.eq("No").any() else "Mixed"
    if post_day1.eq("No").all():
        return "No"
    return "Mixed"
