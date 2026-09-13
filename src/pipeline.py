from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import CONFIG, ResearchConfig
from .features import build_oi_features
from .horizon_nonoverlap import analyze_horizon_nonoverlap
from .incremental_decay import analyze_incremental_decay
from .institutional_divergence import (
    analyze_divergence_surface,
    build_divergence_features,
    divergence_regressions,
    divergence_robustness,
    institutional_composite_comparison,
)
from .finlab_loader import (
    FUTURES_SYMBOL_MAP,
    LONG_OI_KEY,
    NET_OI_KEY,
    SHORT_OI_KEY,
    inspect_futures_schema,
    load_0050_price,
    load_futures_raw,
    login_finlab,
    standardize_futures_oi,
    validate_latest_non_null_dates,
)
from .prior_return import analyze_prior_return_conditioning
from .returns import build_forward_returns, build_incremental_returns, build_prior_returns
from .statistics import add_fdr_by_family, add_global_fdr, grouped_comparisons, hac_lag
from .visualization import create_parameter_surface_figures


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _direction(side: str, group: str) -> str:
    low = group in {"PR_0_20", "PR_0_5", "PR_5_20", "Z_LT_M2_5"}
    high = group in {"PR_80_100", "PR_80_95", "PR_95_100", "Z_GE_P2_5"}
    if not (low or high):
        return "middle"
    if side == "short":
        return "short_covering" if low else "short_increase"
    if side == "long":
        return "long_exit" if low else "long_increase"
    return "toward_short" if low else "toward_long"


def _metadata(
    table: pd.DataFrame,
    institution: str,
    side: str,
    accumulation: int,
    rolling_window: int,
    method: str,
    horizon: int,
    bin_set: str,
) -> pd.DataFrame:
    out = table.copy()
    out["institution"] = institution
    out["predictor"] = f"{side}_change_ratio_{accumulation}d"
    out["side"] = side
    out["accumulation_window"] = accumulation
    out["rolling_window"] = rolling_window
    out["standardization"] = method
    out["outcome"] = f"o1_c{horizon}"
    out["outcome_horizon"] = horizon
    out["outcome_role"] = "primary" if horizon in CONFIG.primary_outcomes else "secondary"
    out["bin_set"] = bin_set
    out["direction"] = out["group"].map(lambda group: _direction(side, group))
    out["fdr_family"] = out["outcome_role"] + "_" + method + "_" + bin_set
    return out


def analyze_finite_grid(
    oi: pd.DataFrame,
    returns: pd.DataFrame,
    institution: str = "外資及陸資",
    config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pre-specified grid without loading external data."""
    features = build_oi_features(
        oi,
        accumulation_windows=config.accumulation_windows,
        rolling_windows=config.rolling_windows,
        min_periods_by_window=config.min_periods_by_window,
    )
    merged = features.join(returns, how="inner")
    merged["institution"] = institution
    tables = []
    for side in config.predictor_sides:
        for accumulation in config.accumulation_windows:
            predictor = f"{side}_change_ratio_{accumulation}d"
            for rolling_window in config.rolling_windows:
                scores = {
                    "percentile": f"{predictor}_pr{rolling_window}",
                    "zscore": f"{predictor}_z{rolling_window}",
                }
                for method, score_col in scores.items():
                    for horizon in config.outcome_horizons:
                        target = f"o1_c{horizon}"
                        lag = hac_lag(accumulation, horizon)
                        if method == "percentile":
                            specifications = (
                                (
                                    config.percentile_bins,
                                    config.percentile_labels,
                                    "coarse",
                                ),
                                (
                                    config.full_percentile_bins,
                                    config.full_percentile_labels,
                                    "full",
                                ),
                            )
                        else:
                            specifications = (
                                (config.zscore_bins, config.zscore_labels, "zscore"),
                            )
                        for bins, labels, bin_set in specifications:
                            table = grouped_comparisons(
                                merged, score_col, target, bins, labels, lag
                            )
                            tables.append(
                                _metadata(
                                    table, institution, side, accumulation,
                                    rolling_window, method, horizon, bin_set,
                                )
                            )
    results = pd.concat(tables, ignore_index=True)
    results = add_fdr_by_family(
        results,
        family_cols=("institution", "fdr_family"),
        p_col="hac_p_value",
    )
    return merged, results


def _build_analysis_oi(raw_futures, config: ResearchConfig) -> dict[str, pd.DataFrame]:
    """Build individual institutions and the date-aligned three-institution total."""
    individual = {
        institution: standardize_futures_oi(raw_futures, institution)
        for institution in config.institutions
    }
    validate_latest_non_null_dates(individual)
    aligned = pd.concat(
        {
            institution: frame[["long_oi", "short_oi"]]
            for institution, frame in individual.items()
        },
        axis=1,
        join="inner",
    )
    if aligned.empty:
        raise ValueError("No common OI dates across the three institutions")

    total = pd.DataFrame(index=aligned.index)
    total["long_oi"] = aligned.xs("long_oi", axis=1, level=1).sum(axis=1)
    total["short_oi"] = aligned.xs("short_oi", axis=1, level=1).sum(axis=1)
    total_name = config.comparison_institutions[-1]
    total["institution"] = total_name
    if total[["long_oi", "short_oi"]].isna().any().any():
        raise ValueError("Institutional total contains missing OI after date alignment")

    result = {**individual, total_name: total}
    if tuple(result) != config.comparison_institutions:
        raise ValueError(
            "Configured comparison institutions do not match constructed OI datasets"
        )
    return result


def _build_foreign_trust_oi(
    oi_by_institution: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build Foreign+Trust underlying OI before normalization."""
    aligned = oi_by_institution["外資及陸資"][["long_oi", "short_oi"]].join(
        oi_by_institution["投信"][["long_oi", "short_oi"]],
        how="inner", lsuffix="_foreign", rsuffix="_trust",
    )
    out = pd.DataFrame(index=aligned.index)
    out["long_oi"] = aligned["long_oi_foreign"] + aligned["long_oi_trust"]
    out["short_oi"] = aligned["short_oi_foreign"] + aligned["short_oi_trust"]
    out["institution"] = "外資+投信"
    return out


def _write_v3_summary(archive: Path, tables: dict[str, pd.DataFrame], commit: str):
    prior = tables["prior"]
    decay = tables["decay"]
    nonoverlap = tables["nonoverlap"]
    divergence = tables["divergence_primary"]
    monotonic = tables["monotonic"]
    regressions = tables["regressions"]
    comparison = tables["comparison"]
    strongest_prior = (
        prior.groupby("lookback")["effect_attenuation_ratio"].median().abs().idxmax()
        if prior["effect_attenuation_ratio"].notna().any() else "無法判定"
    )
    dealer_joint = regressions.loc[
        regressions["model"].eq("foreign_plus_dealer")
        & regressions["term"].eq("dealer_pr")
    ]
    best_decay = (
        decay.groupby("interval")["median_effect_across_12_cells"].median().abs().idxmax()
        if decay["median_effect_across_12_cells"].notna().any() else "無法判定"
    )
    nonoverlap_consistency = (
        nonoverlap["number_offsets_expected_direction"].sum()
        / nonoverlap["total_offsets"].sum()
        if nonoverlap["total_offsets"].sum() else np.nan
    )
    expected_divergence = np.where(
        divergence["group"].eq("Strong bearish"),
        divergence["mean_diff_vs_nongroup"].lt(0),
        divergence["mean_diff_vs_nongroup"].gt(0),
    )
    med_effect = comparison.groupby("predictor")["mean_diff_vs_nongroup"].apply(
        lambda s: s.abs().median()
    )
    divergence_vs_foreign = (
        "較大" if med_effect.get("foreign_minus_dealer_percentile", np.nan)
        > med_effect.get("foreign_alone", np.nan) else "未較大"
    )
    aggregate_removed = (
        "改善" if med_effect.get("aggregate_minus_dealer_foreign_plus_trust", np.nan)
        > med_effect.get("aggregate", np.nan) else "未改善"
    )
    text = f"""# v3 Foreign robustness and Dealer divergence

- Commit: {commit}
- Phase A: confirmatory robustness
- Phase B: exploratory new predictor
- New Z-score analysis: False

## Phase A

1. Prior-return conditioning: {int(prior['q_value_global'].lt(.05).sum())} adjusted cells pass Universe-A Global FDR; direction and magnitude are retained in the full table.
2. The largest median absolute attenuation is associated with Prior {strongest_prior}D.
3. Incremental decay uses compounded price-ratio buckets; the largest median absolute separation occurs in `{best_decay}`. Cumulative C20 evidence is not interpreted as Day-20 alpha.
4. Horizon-specific non-overlap retains every offset for H=2/3/5/10/20; {nonoverlap_consistency:.1%} of offsets have the expected direction, with no best-offset selection.

## Phase B

5. Foreign–Dealer divergence is Foreign causal percentile minus Dealer causal percentile; {int(expected_divergence.sum())}/24 extreme cells have the expected direction.
6. The independent 24-hypothesis divergence universe has {int(divergence['q_value_global'].lt(.05).sum())} Level-A cells.
7. Median five-bucket monotonicity is {monotonic['monotonic_order_score'].median():.1f}/4 across the frozen 12-cell surface.
8. Dealer has raw HAC p < 0.05 in {int(dealer_joint['hac_p_value'].lt(.05).sum())}/12 joint-model cells after controlling Foreign.
9. By median absolute group separation, divergence is {divergence_vs_foreign} than Foreign alone.
10. Removing Dealer from Aggregate is classified as `{aggregate_removed}` by the same descriptive comparison. Foreign+Trust is constructed from underlying OI before normalization.

## Research judgment

- Phase A modules: 修改後再測，應依 q-values、attenuation、decay 與 multi-offset consistency 判讀。
- Foreign–Dealer divergence: 保留作為 prospective-validation candidate at most; it is not a validated trading signal.
"""
    (archive / "research_summary_v3.md").write_text(text, encoding="utf-8")
    (archive / "research_summary.md").write_text(text, encoding="utf-8")


def _data_quality(
    oi_by_institution: dict[str, pd.DataFrame],
    price: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for institution, frame in oi_by_institution.items():
        gross = frame["long_oi"] + frame["short_oi"]
        rows.append(
            {
                "dataset": f"oi_{institution}",
                "institution": institution,
                "source_api_type": "field_specific",
                "long_source_key": LONG_OI_KEY,
                "short_source_key": SHORT_OI_KEY,
                "target_column": FUTURES_SYMBOL_MAP.get(institution, "derived_sum"),
                "first_date": str(frame.index.min().date()),
                "last_date": str(frame.index.max().date()),
                "missing_long_oi": int(frame["long_oi"].isna().sum()),
                "missing_short_oi": int(frame["short_oi"].isna().sum()),
                "duplicate_dates": int(frame.index.duplicated().sum()),
                "date_monotonic": bool(frame.index.is_monotonic_increasing),
                "zero_gross_oi": int(gross.eq(0).sum()),
            }
        )
    rows.append(
        {
            "dataset": "price_0050",
            "institution": "",
            "source_api_type": "adjusted_price",
            "first_date": str(price.index.min().date()),
            "last_date": str(price.index.max().date()),
            "missing_open_0050": int(price["open_0050"].isna().sum()),
            "missing_close_0050": int(price["close_0050"].isna().sum()),
            "duplicate_dates": int(price.index.duplicated().sum()),
            "date_monotonic": bool(price.index.is_monotonic_increasing),
        }
    )
    return pd.DataFrame(rows)


def _annual_and_era(
    merged: pd.DataFrame,
    institution: str,
    config: ResearchConfig,
):
    annual_rows, era_rows = [], []
    for side in config.predictor_sides:
        for accumulation in config.accumulation_windows:
            predictor = f"{side}_change_ratio_{accumulation}d"
            for rolling_window in config.rolling_windows:
                for method, suffix in (("percentile", f"pr{rolling_window}"), ("zscore", f"z{rolling_window}")):
                    score_col = f"{predictor}_{suffix}"
                    if method == "percentile":
                        groups = pd.cut(
                            merged[score_col], config.percentile_bins,
                            labels=config.percentile_labels, include_lowest=True,
                        )
                        extremes = ("PR_0_20", "PR_80_100")
                    else:
                        groups = pd.cut(
                            merged[score_col], config.zscore_bins,
                            labels=config.zscore_labels, include_lowest=True,
                        )
                        extremes = ("Z_LT_M2_5", "Z_GE_P2_5")
                    for group in extremes:
                        mask = groups.eq(group)
                        sample = merged.loc[mask, "o1_c1"].dropna()
                        for year, values in sample.groupby(sample.index.year):
                            annual_rows.append(
                                {
                                    "institution": institution,
                                    "side": side, "accumulation_window": accumulation,
                                    "rolling_window": rolling_window, "standardization": method,
                                    "group": group, "year": year, "n": len(values),
                                    "mean_return": values.mean(), "median_return": values.median(),
                                    "positive_rate": values.gt(0).mean(),
                                }
                            )
                        years = sample.index.year
                        eras = pd.Series(
                            np.select(
                                [years < 2013, years < 2020],
                                ["before_2013", "2013_2019"],
                                default="2020_onward",
                            ),
                            index=sample.index,
                        )
                        for era, values in sample.groupby(eras):
                            era_rows.append(
                                {
                                    "institution": institution,
                                    "side": side, "accumulation_window": accumulation,
                                    "rolling_window": rolling_window, "standardization": method,
                                    "group": group, "era": era, "n": len(values),
                                    "mean_return": values.mean(), "median_return": values.median(),
                                    "positive_rate": values.gt(0).mean(),
                                }
                            )
    return pd.DataFrame(annual_rows), pd.DataFrame(era_rows)


def _event_and_nonoverlap(
    merged: pd.DataFrame,
    institution: str,
    config: ResearchConfig,
):
    event_rows, nonoverlap_rows = [], []
    for side in config.predictor_sides:
        for accumulation in config.accumulation_windows:
            predictor = f"{side}_change_ratio_{accumulation}d"
            for rolling_window in config.rolling_windows:
                score = merged[f"{predictor}_pr{rolling_window}"]
                groups = pd.cut(
                    score, config.percentile_bins, labels=config.percentile_labels,
                    include_lowest=True,
                )
                for group in ("PR_0_20", "PR_80_100"):
                    mask = groups.eq(group)
                    entry = mask & ~mask.shift(1, fill_value=False)
                    for sample_name, sample_mask in (
                        ("event_entry", entry),
                        ("nonoverlap", mask & (np.arange(len(mask)) % max(1, accumulation) == 0)),
                    ):
                        values = merged.loc[sample_mask, "o1_c1"].dropna()
                        row = {
                            "institution": institution,
                            "side": side, "accumulation_window": accumulation,
                            "rolling_window": rolling_window, "group": group,
                            "n": len(values), "mean_return": values.mean(),
                            "median_return": values.median(),
                            "positive_rate": values.gt(0).mean(),
                        }
                        (event_rows if sample_name == "event_entry" else nonoverlap_rows).append(row)
    return pd.DataFrame(event_rows), pd.DataFrame(nonoverlap_rows)


def _outlier_robustness(
    merged: pd.DataFrame,
    results: pd.DataFrame,
    institution: str,
    config: ResearchConfig,
) -> pd.DataFrame:
    primary = results.loc[
        results["outcome_role"].eq("primary")
        & results["bin_set"].eq("coarse")
        & results["group"].isin(["PR_0_20", "PR_80_100"])
    ].copy()
    target = merged["o1_c1"]
    low, high = target.quantile([0.01, 0.99])
    trimmed_valid = target.between(low, high)
    without_top5_valid = ~merged.index.isin(target.nlargest(5).index)
    rows = []
    for row in primary.itertuples(index=False):
        score_col = f"{row.predictor}_pr{row.rolling_window}"
        groups = pd.cut(
            merged[score_col], config.percentile_bins,
            labels=config.percentile_labels, include_lowest=True,
        )
        group_mask = groups.eq(row.group)
        for scenario, valid in (
            ("raw", target.notna()),
            ("trim_1pct_each_tail", trimmed_valid),
            ("remove_top5_gains", without_top5_valid & target.notna()),
        ):
            group = target.loc[group_mask & valid]
            nongroup = target.loc[~group_mask & groups.notna() & valid]
            rows.append(
                {
                    "institution": institution,
                    "side": row.side,
                    "accumulation_window": row.accumulation_window,
                    "rolling_window": row.rolling_window,
                    "group": row.group,
                    "scenario": scenario,
                    "n": len(group),
                    "n_nongroup": len(nongroup),
                    "mean_return": group.mean(),
                    "mean_nongroup": nongroup.mean(),
                    "mean_diff_vs_nongroup": group.mean() - nongroup.mean(),
                    "positive_rate": group.gt(0).mean(),
                    "trim_low": low,
                    "trim_high": high,
                }
            )
    return pd.DataFrame(rows)


def _post_period_validation(
    merged: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Compare the frozen foreign-net grid before and after the legacy cutoff."""
    cutoff = pd.Timestamp("2024-04-30")
    periods = {
        "original_baseline": merged.index <= cutoff,
        "post_period": merged.index > cutoff,
        "updated_full_sample": pd.Series(True, index=merged.index),
    }
    rows = []
    for accumulation in config.accumulation_windows:
        predictor = f"net_change_ratio_{accumulation}d"
        for rolling_window in config.rolling_windows:
            specifications = (
                (
                    "percentile",
                    f"{predictor}_pr{rolling_window}",
                    config.percentile_bins,
                    config.percentile_labels,
                    ("PR_0_20", "PR_80_100"),
                ),
                (
                    "zscore",
                    f"{predictor}_z{rolling_window}",
                    config.zscore_bins,
                    config.zscore_labels,
                    ("Z_LT_M2_5", "Z_GE_P2_5"),
                ),
            )
            for method, score_col, bins, labels, extremes in specifications:
                for period_name, period_mask in periods.items():
                    sample = merged.loc[period_mask]
                    table = grouped_comparisons(
                        sample, score_col, "o1_c1", bins, labels,
                        hac_lag(accumulation, 1),
                    )
                    table = table.loc[table["group"].isin(extremes)].copy()
                    table["sample_period"] = period_name
                    table["institution"] = "外資及陸資"
                    table["side"] = "net"
                    table["predictor"] = predictor
                    table["accumulation_window"] = accumulation
                    table["rolling_window"] = rolling_window
                    table["standardization"] = method
                    table["outcome"] = "o1_c1"
                    rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    out = add_fdr_by_family(out, family_cols=("sample_period", "standardization"))
    out["outcome_horizon"] = 1
    return add_global_fdr(out)


def _make_archive(output_root: str | Path, config: ResearchConfig) -> tuple[Path, str]:
    commit = _git_commit()
    timestamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    archive = Path(output_root) / f"{timestamp}_{config.research_version}_{commit[:8]}"
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "figures").mkdir()
    return archive, commit


def run_research(output_root="outputs", inspect_schema=True, config: ResearchConfig = CONFIG):
    np.random.seed(config.random_seed)
    login_finlab()
    raw_futures, futures_keys = load_futures_raw()
    if inspect_schema:
        print(f"Selected futures keys: {futures_keys}")
        inspect_futures_schema(raw_futures)
    price, price_keys = load_0050_price(config.target_symbol)
    expected_price_keys = {"open": "etl:adj_open", "close": "etl:adj_close"}
    if price_keys != expected_price_keys:
        raise RuntimeError(f"Adjusted-price validation failed: {price_keys}")
    print("price_source_open=etl:adj_open")
    print("price_source_close=etl:adj_close")
    print("outcome_price_adjusted=True")
    returns = build_forward_returns(price, config.outcome_horizons)
    oi_by_institution = _build_analysis_oi(raw_futures, config)
    merged_parts = {}
    result_parts = []
    annual_parts, era_parts = [], []
    event_parts, nonoverlap_parts, outlier_parts = [], [], []

    for institution, oi in oi_by_institution.items():
        print(f"Running finite grid: {institution}")
        merged, institution_results = analyze_finite_grid(
            oi, returns, institution, config
        )
        merged_parts[institution] = merged
        result_parts.append(institution_results)

        annual, eras = _annual_and_era(merged, institution, config)
        events, nonoverlap = _event_and_nonoverlap(merged, institution, config)
        outliers = _outlier_robustness(
            merged, institution_results, institution, config
        )
        annual_parts.append(annual)
        era_parts.append(eras)
        event_parts.append(events)
        nonoverlap_parts.append(nonoverlap)
        outlier_parts.append(outliers)

    results = pd.concat(result_parts, ignore_index=True)
    results = add_global_fdr(results)
    merged = pd.concat(merged_parts, names=["institution", "date"])
    annual = pd.concat(annual_parts, ignore_index=True)
    eras = pd.concat(era_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    nonoverlap = pd.concat(nonoverlap_parts, ignore_index=True)
    outliers = pd.concat(outlier_parts, ignore_index=True)
    post_period = _post_period_validation(
        merged_parts[config.primary_institutions[0]], config
    )

    foreign = merged_parts[config.primary_institutions[0]].join(
        build_prior_returns(price, config.prior_return_windows), how="left"
    ).join(build_incremental_returns(price), how="left")
    prior_results, prior_summary = analyze_prior_return_conditioning(foreign, config)
    decay_results, decay_summary = analyze_incremental_decay(foreign, config)
    horizon_nonoverlap, horizon_nonoverlap_summary = analyze_horizon_nonoverlap(
        foreign, config
    )

    divergence_frame = build_divergence_features(
        merged_parts["外資及陸資"], merged_parts["自營商"], config
    ).join(returns[["o1_c1"]], how="left")
    divergence_surface, divergence_primary, divergence_monotonic = (
        analyze_divergence_surface(divergence_frame, config)
    )
    divergence_regression = divergence_regressions(divergence_frame, config)
    divergence_robust = divergence_robustness(divergence_frame, config)
    foreign_trust = build_oi_features(
        _build_foreign_trust_oi(oi_by_institution),
        accumulation_windows=config.accumulation_windows,
        rolling_windows=config.rolling_windows,
        min_periods_by_window=config.min_periods_by_window,
    ).join(returns, how="inner")
    comparison_frames = dict(merged_parts)
    comparison_frames["外資+投信"] = foreign_trust
    composite_comparison = institutional_composite_comparison(
        comparison_frames, divergence_frame, config
    )
    archive, commit = _make_archive(output_root, config)

    quality = _data_quality(oi_by_institution, price)
    primary = results.loc[results["outcome_role"].eq("primary")]
    secondary = results.loc[results["outcome_role"].eq("secondary")]
    full_quantiles = results.loc[results["bin_set"].eq("full")]
    surface = results.loc[
        results["bin_set"].isin(["coarse", "zscore"])
        & ~results["direction"].eq("middle")
    ]
    global_columns = [
        "institution", "side", "predictor", "accumulation_window",
        "rolling_window", "standardization", "group", "outcome",
        "outcome_horizon", "n", "mean_return", "median_return",
        "positive_rate", "mean_nongroup", "mean_diff_vs_nongroup",
        "hac_p_value", "q_value_family", "q_value_global", "evidence_level",
    ]

    outputs = {
        "data_quality_report.csv": quality,
        "grid_full_results.csv": results,
        "primary_results.csv": primary,
        "secondary_persistence_results.csv": secondary,
        "full_quantile_results.csv": full_quantiles,
        "fdr_results.csv": results.loc[results["hac_p_value"].notna()],
        "annual_results.csv": annual,
        "era_results.csv": eras,
        "event_entry_results.csv": events,
        "nonoverlap_results.csv": nonoverlap,
        "outlier_robustness.csv": outliers,
        "parameter_surface_summary.csv": surface,
        "global_fdr_summary.csv": results.loc[:, global_columns],
        "post_period_validation.csv": post_period,
        "prior_return_conditioning.csv": prior_results,
        "prior_return_summary.csv": prior_summary,
        "incremental_decay_results.csv": decay_results,
        "incremental_decay_summary.csv": decay_summary,
        "horizon_nonoverlap_results.csv": horizon_nonoverlap,
        "horizon_nonoverlap_summary.csv": horizon_nonoverlap_summary,
        "foreign_dealer_divergence_primary.csv": divergence_primary,
        "foreign_dealer_divergence_surface.csv": divergence_surface,
        "foreign_dealer_divergence_fdr.csv": divergence_primary,
        "foreign_dealer_divergence_regression.csv": divergence_regression,
        "foreign_dealer_divergence_annual.csv": divergence_robust["annual"],
        "foreign_dealer_divergence_era.csv": divergence_robust["era"],
        "foreign_dealer_divergence_outlier.csv": divergence_robust["outlier"],
        "foreign_dealer_divergence_event_entry.csv": divergence_robust["event_entry"],
        "institutional_composite_comparison.csv": composite_comparison,
        "foreign_dealer_divergence_monotonicity.csv": divergence_monotonic,
    }
    for filename, frame in outputs.items():
        frame.to_csv(archive / filename, index=False, encoding="utf-8-sig")
    merged.to_parquet(archive / "daily_dataset.parquet")
    create_parameter_surface_figures(results, archive / "figures")

    config_dict = asdict(config)
    with (archive / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, ensure_ascii=False, indent=2)
    date_index = merged.index.get_level_values("date")
    run_info = {
        "repository": "hh4832/taiwan-futures-oi-research",
        "git_branch": _git_branch(),
        "git_commit": commit,
        "research_version": config.research_version,
        "price_source_open": price_keys["open"],
        "price_source_close": price_keys["close"],
        "outcome_price_adjusted": True,
        "outcome_definition": "signal t; O1=adjusted_open[t+1]; Ck=adjusted_close[t+k] on trading-date rows",
        "timezone": config.timezone,
        "run_time": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "futures_long_oi_key": futures_keys["long"],
        "futures_short_oi_key": futures_keys["short"],
        "futures_net_oi_validation_key": futures_keys["net"],
        "price_keys": price_keys,
        "data_start": str(date_index.min().date()),
        "data_end": str(date_index.max().date()),
        "primary_outcome": "o1_c1",
        "multiple_testing_method": "BH",
        "family_fdr": True,
        "global_fdr": True,
        "global_fdr_primary_outcome": "o1_c1",
        "secondary_global_fdr_mode": "by_horizon",
        "analysis_institutions": ",".join(oi_by_institution),
        "drive_root_folder_id": config.drive_folder_id,
        "archive_name": archive.name,
        "oi_source": "field_specific",
        "phase_a_prior_return": True,
        "phase_a_incremental_decay": True,
        "phase_a_horizon_nonoverlap": True,
        "phase_b_foreign_dealer_divergence": True,
        "zscore_new_analysis": False,
        "accumulation_windows": ",".join(map(str, config.accumulation_windows)),
        "rolling_windows": ",".join(map(str, config.rolling_windows)),
        "prior_return_windows": ",".join(map(str, config.prior_return_windows)),
        "divergence_definition": "foreign_percentile_minus_dealer_percentile",
        "divergence_primary_outcome": "o1_c1",
        "divergence_primary_extreme_threshold": 0.60,
    }
    prefixes = {"外資及陸資": "foreign", "投信": "trust", "自營商": "dealer"}
    for institution, prefix in prefixes.items():
        frame = oi_by_institution[institution]
        run_info[f"{prefix}_first_long_date"] = str(frame.attrs["first_long_date"].date())
        run_info[f"{prefix}_last_long_date"] = str(frame.attrs["last_long_date"].date())
        run_info[f"{prefix}_first_short_date"] = str(frame.attrs["first_short_date"].date())
        run_info[f"{prefix}_last_short_date"] = str(frame.attrs["last_short_date"].date())
    (archive / "run_info.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in run_info.items()) + "\n",
        encoding="utf-8",
    )
    _write_v3_summary(archive, {
        "prior": prior_results,
        "decay": decay_summary,
        "nonoverlap": horizon_nonoverlap_summary,
        "divergence_primary": divergence_primary,
        "monotonic": divergence_monotonic,
        "regressions": divergence_regression,
        "comparison": composite_comparison,
    }, commit)
    print(f"Saved archive: {archive}")
    return merged, results, archive
