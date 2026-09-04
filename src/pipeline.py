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


def _data_quality(oi: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in (("oi", oi), ("price", price)):
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
    if {"long_oi", "short_oi"}.issubset(oi.columns):
        gross = oi["long_oi"] + oi["short_oi"]
        rows.append({"dataset": "oi", "check": "zero_gross_oi", "value": int(gross.eq(0).sum())})
    return pd.DataFrame(rows)


def _annual_and_era(merged: pd.DataFrame, config: ResearchConfig):
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
                                    "side": side, "accumulation_window": accumulation,
                                    "rolling_window": rolling_window, "standardization": method,
                                    "group": group, "era": era, "n": len(values),
                                    "mean_return": values.mean(), "median_return": values.median(),
                                    "positive_rate": values.gt(0).mean(),
                                }
                            )
    return pd.DataFrame(annual_rows), pd.DataFrame(era_rows)


def _event_and_nonoverlap(merged: pd.DataFrame, config: ResearchConfig):
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
                            "side": side, "accumulation_window": accumulation,
                            "rolling_window": rolling_window, "group": group,
                            "n": len(values), "mean_return": values.mean(),
                            "median_return": values.median(),
                            "positive_rate": values.gt(0).mean(),
                        }
                        (event_rows if sample_name == "event_entry" else nonoverlap_rows).append(row)
    return pd.DataFrame(event_rows), pd.DataFrame(nonoverlap_rows)


def _outlier_robustness(merged: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
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
            merged[score_col], CONFIG.percentile_bins,
            labels=CONFIG.percentile_labels, include_lowest=True,
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
    returns = build_forward_returns(price, config.outcome_horizons)
    oi = standardize_futures_oi(raw_futures, config.primary_institutions[0])
    merged, results = analyze_finite_grid(oi, returns, config.primary_institutions[0], config)
    archive, commit = _make_archive(output_root, config)

    quality = _data_quality(oi, price)
    annual, eras = _annual_and_era(merged, config)
    events, nonoverlap = _event_and_nonoverlap(merged, config)
    outliers = _outlier_robustness(merged, results)
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
    run_info = {
        "git_commit": commit,
        "timezone": config.timezone,
        "run_time": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "futures_key": futures_key,
        "price_keys": price_keys,
        "data_start": str(merged.index.min().date()),
        "data_end": str(merged.index.max().date()),
        "primary_outcome": "o1_c1",
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
        f"- Primary rows: {len(primary):,}\n"
        f"- Secondary rows: {len(secondary):,}\n\n"
        "Interpretation and freeze decision require review of the generated robustness tables.\n"
    )
    (archive / "research_summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved archive: {archive}")
    return merged, results, archive
