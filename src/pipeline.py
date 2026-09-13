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
from .finlab_loader import (
    inspect_futures_schema,
    load_0050_price,
    load_futures_raw,
    login_finlab,
    standardize_futures_oi,
)
from .returns import build_forward_returns
from .statistics import add_fdr_by_family, grouped_comparisons, hac_lag
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


def _data_quality(
    oi_by_institution: dict[str, pd.DataFrame],
    price: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    datasets = [(f"oi_{institution}", frame) for institution, frame in oi_by_institution.items()]
    datasets.append(("price", price))
    for name, frame in datasets:
        rows.extend(
            {
                "dataset": name,
                "check": f"missing_{column}",
                "value": int(frame[column].isna().sum()),
            }
            for column in frame.columns
        )
        rows.extend(
            [
                {"dataset": name, "check": "duplicate_dates", "value": int(frame.index.duplicated().sum())},
                {"dataset": name, "check": "date_monotonic", "value": bool(frame.index.is_monotonic_increasing)},
                {"dataset": name, "check": "first_date", "value": str(frame.index.min().date())},
                {"dataset": name, "check": "last_date", "value": str(frame.index.max().date())},
            ]
        )
        if name.startswith("oi_") and {"long_oi", "short_oi"}.issubset(frame.columns):
            gross = frame["long_oi"] + frame["short_oi"]
            rows.append(
                {"dataset": name, "check": "zero_gross_oi", "value": int(gross.eq(0).sum())}
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


def _make_archive(output_root: str | Path, config: ResearchConfig) -> tuple[Path, str]:
    commit = _git_commit()
    timestamp = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    archive = Path(output_root) / f"{timestamp}_{commit[:8]}"
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "figures").mkdir()
    return archive, commit


def run_research(output_root="outputs", inspect_schema=True, config: ResearchConfig = CONFIG):
    np.random.seed(config.random_seed)
    login_finlab()
    raw_futures, futures_key = load_futures_raw()
    if inspect_schema:
        print(f"Selected futures key: {futures_key}")
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
    merged = pd.concat(merged_parts, names=["institution", "date"])
    annual = pd.concat(annual_parts, ignore_index=True)
    eras = pd.concat(era_parts, ignore_index=True)
    events = pd.concat(event_parts, ignore_index=True)
    nonoverlap = pd.concat(nonoverlap_parts, ignore_index=True)
    outliers = pd.concat(outlier_parts, ignore_index=True)
    archive, commit = _make_archive(output_root, config)

    quality = _data_quality(oi_by_institution, price)
    primary = results.loc[results["outcome_role"].eq("primary")]
    secondary = results.loc[results["outcome_role"].eq("secondary")]
    full_quantiles = results.loc[results["bin_set"].eq("full")]
    surface = results.loc[
        results["bin_set"].isin(["coarse", "zscore"])
        & ~results["direction"].eq("middle")
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
        "branch": _git_branch(),
        "git_commit": commit,
        "price_source_open": price_keys["open"],
        "price_source_close": price_keys["close"],
        "outcome_price_adjusted": True,
        "outcome_definition": "signal t; O1=adjusted_open[t+1]; Ck=adjusted_close[t+k] on trading-date rows",
        "timezone": config.timezone,
        "run_time": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "futures_key": futures_key,
        "price_keys": price_keys,
        "data_start": str(date_index.min().date()),
        "data_end": str(date_index.max().date()),
        "primary_outcome": "o1_c1",
        "analysis_institutions": ",".join(oi_by_institution),
        "drive_folder_id": config.drive_folder_id,
    }
    (archive / "run_info.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in run_info.items()) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# Futures OI finite-grid run\n\n"
        f"- Commit: {commit}\n"
        f"- Data: {run_info['data_start']} to {run_info['data_end']}\n"
        f"- Institutions: {', '.join(oi_by_institution)}\n"
        f"- Primary rows: {len(primary):,}\n"
        f"- Secondary rows: {len(secondary):,}\n\n"
        "Interpretation and freeze decision require review of the generated robustness tables.\n"
    )
    (archive / "research_summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved archive: {archive}")
    return merged, results, archive
