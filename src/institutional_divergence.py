import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .config import CONFIG, ResearchConfig
from .statistics import add_fdr_by_family, compare_group_to_nongroup, hac_lag


EXTREME_GROUPS = ("Strong bearish", "Strong bullish")


def assign_divergence_bucket(divergence: pd.Series) -> pd.Series:
    """Assign the five frozen buckets with the exact pre-specified boundaries."""
    out = pd.Series(pd.NA, index=divergence.index, dtype="object")
    out.loc[divergence.le(-.60)] = "Strong bearish"
    out.loc[divergence.gt(-.60) & divergence.le(-.20)] = "Moderate bearish"
    out.loc[divergence.gt(-.20) & divergence.lt(.20)] = "Neutral"
    out.loc[divergence.ge(.20) & divergence.lt(.60)] = "Moderate bullish"
    out.loc[divergence.ge(.60)] = "Strong bullish"
    return out


def build_divergence_features(
    foreign: pd.DataFrame,
    dealer: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> pd.DataFrame:
    """Subtract causal percentiles computed on matching accumulation/rolling cells."""
    if not foreign.index.equals(dealer.index):
        raise ValueError("Foreign and Dealer rows must have identical trading-date alignment")
    out = pd.DataFrame(index=foreign.index)
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            base = f"net_change_ratio_{accumulation}d_pr{rolling}"
            out[f"foreign_pr_{accumulation}d_{rolling}d"] = foreign[base]
            out[f"dealer_pr_{accumulation}d_{rolling}d"] = dealer[base]
            out[f"fd_divergence_{accumulation}d_{rolling}d"] = foreign[base] - dealer[base]
    return out


def _module_fdr(primary: pd.DataFrame) -> pd.DataFrame:
    out = add_fdr_by_family(primary, family_cols=("group",))
    out["q_value_global"] = np.nan
    eligible = out["group"].isin(EXTREME_GROUPS) & out["hac_p_value"].notna()
    if eligible.any():
        out.loc[eligible, "q_value_global"] = multipletests(
            out.loc[eligible, "hac_p_value"], method="fdr_bh"
        )[1]
    out["evidence_level"] = "Not evaluable"
    evaluable = out["group"].isin(EXTREME_GROUPS) & out["hac_p_value"].notna()
    out.loc[evaluable, "evidence_level"] = "Level D"
    out.loc[evaluable & out["hac_p_value"].lt(.05), "evidence_level"] = "Level C"
    out.loc[evaluable & out["q_value_family"].lt(.05), "evidence_level"] = "Level B"
    out.loc[evaluable & out["q_value_global"].lt(.05), "evidence_level"] = "Level A"
    return out


def analyze_divergence_surface(
    frame: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            col = f"fd_divergence_{accumulation}d_{rolling}d"
            buckets = assign_divergence_bucket(frame[col])
            for group in config.divergence_labels:
                row = compare_group_to_nongroup(
                    buckets.eq(group).where(buckets.notna()), frame["o1_c1"],
                    hac_lag(accumulation, 1),
                )
                row.update({
                    "analysis_role": "exploratory_new_predictor",
                    "predictor": "foreign_percentile_minus_dealer_percentile",
                    "accumulation_window": accumulation,
                    "rolling_window": rolling,
                    "parameter_cell": f"{accumulation}D_x_{rolling}D",
                    "group": group,
                    "outcome": "o1_c1",
                    "fdr_universe": "D_divergence_primary_24",
                })
                rows.append(row)
    surface = pd.DataFrame(rows)
    primary = _module_fdr(surface.loc[surface["group"].isin(EXTREME_GROUPS)].copy())
    order = {label: i for i, label in enumerate(config.divergence_labels)}
    shape = surface.assign(order=surface["group"].map(order)).sort_values(
        ["accumulation_window", "rolling_window", "order"]
    )
    monotonic = (
        shape.groupby(["accumulation_window", "rolling_window"])["mean_return"]
        .apply(lambda s: int((np.diff(s.to_numpy()) >= 0).sum()))
        .rename("monotonic_order_score")
        .reset_index()
    )
    monotonic["comparisons_total"] = 4
    return surface, primary, monotonic


def _regression_row(target, predictors, model_name, accumulation, rolling):
    work = pd.concat([target.rename("target"), predictors], axis=1).dropna()
    row = {
        "analysis_role": "exploratory_new_predictor",
        "model": model_name,
        "accumulation_window": accumulation,
        "rolling_window": rolling,
        "n": len(work),
    }
    if len(work) < max(20, len(predictors.columns) + 3):
        return [dict(row, term=term, beta=np.nan, se=np.nan, hac_p_value=np.nan, r_squared=np.nan) for term in predictors]
    fit = sm.OLS(work["target"], sm.add_constant(work[predictors.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lag(accumulation, 1)}
    )
    return [dict(
        row, term=term, beta=float(fit.params[term]), se=float(fit.bse[term]),
        hac_p_value=float(fit.pvalues[term]), r_squared=float(fit.rsquared),
    ) for term in predictors]


def divergence_regressions(frame: pd.DataFrame, config: ResearchConfig = CONFIG) -> pd.DataFrame:
    rows = []
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            foreign = frame[f"foreign_pr_{accumulation}d_{rolling}d"].rename("foreign_pr")
            dealer = frame[f"dealer_pr_{accumulation}d_{rolling}d"].rename("dealer_pr")
            divergence = frame[f"fd_divergence_{accumulation}d_{rolling}d"].rename("fd_divergence")
            specs = {
                "foreign_only": foreign.to_frame(),
                "dealer_only": dealer.to_frame(),
                "foreign_plus_dealer": pd.concat([foreign, dealer], axis=1),
                "fd_divergence": divergence.to_frame(),
            }
            for name, predictors in specs.items():
                rows.extend(_regression_row(frame["o1_c1"], predictors, name, accumulation, rolling))
    return pd.DataFrame(rows)


def institutional_composite_comparison(
    frames: dict[str, pd.DataFrame],
    divergence: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> pd.DataFrame:
    """Compare fixed composites; Foreign+Trust is built before normalization upstream."""
    rows = []
    sources = {
        "foreign_alone": frames["外資及陸資"],
        "dealer_alone": frames["自營商"],
        "aggregate": frames["三大法人合計"],
        "aggregate_minus_dealer_foreign_plus_trust": frames["外資+投信"],
    }
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            base = f"net_change_ratio_{accumulation}d_pr{rolling}"
            for predictor_name, source in sources.items():
                score = source[base]
                for group, mask in {
                    "low": score.le(.20), "high": score.ge(.80)
                }.items():
                    stat = compare_group_to_nongroup(
                        mask.where(score.notna()), source["o1_c1"],
                        hac_lag(accumulation, 1),
                    )
                    stat.update({
                        "analysis_role": "exploratory_comparison",
                        "predictor": predictor_name,
                        "definition": (
                            "underlying_foreign_plus_trust_oi_then_change_ratio_then_percentile"
                            if predictor_name.startswith("aggregate_minus")
                            else "underlying_oi_then_change_ratio_then_percentile"
                        ),
                        "accumulation_window": accumulation,
                        "rolling_window": rolling,
                        "group": group,
                    })
                    rows.append(stat)
            buckets = assign_divergence_bucket(
                divergence[f"fd_divergence_{accumulation}d_{rolling}d"]
            )
            for group in EXTREME_GROUPS:
                stat = compare_group_to_nongroup(
                    buckets.eq(group).where(buckets.notna()), divergence["o1_c1"],
                    hac_lag(accumulation, 1),
                )
                stat.update({
                    "analysis_role": "exploratory_comparison",
                    "predictor": "foreign_minus_dealer_percentile",
                    "definition": "foreign_percentile_minus_dealer_percentile",
                    "accumulation_window": accumulation,
                    "rolling_window": rolling,
                    "group": group,
                })
                rows.append(stat)
    return pd.DataFrame(rows)


def divergence_robustness(
    frame: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> dict[str, pd.DataFrame]:
    annual, era, outlier, event = [], [], [], []
    target = frame["o1_c1"]
    qlow, qhigh = target.quantile([.01, .99])
    without_top5 = ~frame.index.isin(target.nlargest(5).index)
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            col = f"fd_divergence_{accumulation}d_{rolling}d"
            buckets = assign_divergence_bucket(frame[col])
            for group in EXTREME_GROUPS:
                mask = buckets.eq(group)
                values = target.loc[mask].dropna()
                base = {"accumulation_window": accumulation, "rolling_window": rolling, "group": group}
                for year, sample in values.groupby(values.index.year):
                    annual.append(dict(base, year=year, n=len(sample), mean_return=sample.mean(), median_return=sample.median(), positive_rate=sample.gt(0).mean()))
                years = values.index.year
                eras = pd.Series(np.select([years < 2013, years < 2020], ["before_2013", "2013_2019"], default="2020_onward"), index=values.index)
                for era_name, sample in values.groupby(eras):
                    era.append(dict(base, era=era_name, n=len(sample), mean_return=sample.mean(), median_return=sample.median(), positive_rate=sample.gt(0).mean()))
                for scenario, valid in {
                    "raw": target.notna(),
                    "trim_1pct_each_tail": target.between(qlow, qhigh),
                    "remove_top5_gains": target.notna() & without_top5,
                }.items():
                    sample = target.loc[mask & valid]
                    other = target.loc[~mask & buckets.notna() & valid]
                    outlier.append(dict(base, scenario=scenario, n=len(sample), mean_return=sample.mean(), mean_diff_vs_nongroup=sample.mean()-other.mean(), positive_rate=sample.gt(0).mean()))
                entry = mask & ~mask.shift(1, fill_value=False)
                sample = target.loc[entry].dropna()
                event.append(dict(base, n=len(sample), mean_return=sample.mean(), median_return=sample.median(), positive_rate=sample.gt(0).mean()))
    return {
        "annual": pd.DataFrame(annual), "era": pd.DataFrame(era),
        "outlier": pd.DataFrame(outlier), "event_entry": pd.DataFrame(event),
    }
