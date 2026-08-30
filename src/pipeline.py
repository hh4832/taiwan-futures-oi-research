from pathlib import Path
import pandas as pd

from .config import CONFIG
from .finlab_loader import (
    login_finlab,
    load_futures_raw,
    inspect_futures_schema,
    standardize_futures_oi,
    load_0050_price,
)
from .features import build_oi_features
from .returns import build_forward_returns
from .statistics import group_by_percentile, add_fdr_qvalues


def _analyze_one(label, oi, returns, result_tables, datasets):
    feat = build_oi_features(oi, CONFIG.rolling_window, CONFIG.min_periods)
    merged = feat.join(returns, how="inner")
    merged["institution"] = label
    datasets.append(merged)

    for predictor in CONFIG.primary_predictors:
        pr_col = f"{predictor}_pr"
        for h in CONFIG.primary_horizons:
            for target in [
                f"close_to_close_fwd_{h}",
                f"next_open_to_close_fwd_{h}",
            ]:
                if target not in merged.columns:
                    continue
                table = group_by_percentile(
                    merged,
                    predictor_pr_col=pr_col,
                    target_col=target,
                    bins=CONFIG.primary_pr_bins,
                    labels=("PR_0_20", "PR_20_80", "PR_80_100"),
                    maxlags=max(1, h),
                )
                table["institution"] = label
                table["predictor"] = predictor
                table["target"] = target
                result_tables.append(table)


def run_research(output_dir="outputs", inspect_schema=True):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    login_finlab()
    raw_futures, futures_key = load_futures_raw()
    if inspect_schema:
        print(f"Selected futures key: {futures_key}")
        inspect_futures_schema(raw_futures)

    price, price_keys = load_0050_price(CONFIG.target_symbol)
    returns = build_forward_returns(price, CONFIG.horizons)

    datasets = []
    result_tables = []
    standardized = {}

    for institution in CONFIG.institutions:
        oi = standardize_futures_oi(raw_futures, institution)
        standardized[institution] = oi
        _analyze_one(institution, oi, returns, result_tables, datasets)

    # Exploratory aggregate of the three institutions.
    agg = pd.concat(
        [
            df[["long_oi", "short_oi"]].rename(
                columns={
                    "long_oi": f"long_{i}",
                    "short_oi": f"short_{i}",
                }
            )
            for i, df in standardized.items()
        ],
        axis=1,
    )
    long_cols = [c for c in agg.columns if c.startswith("long_")]
    short_cols = [c for c in agg.columns if c.startswith("short_")]
    agg_oi = pd.DataFrame(
        {
            "long_oi": agg[long_cols].sum(axis=1, min_count=len(long_cols)),
            "short_oi": agg[short_cols].sum(axis=1, min_count=len(short_cols)),
            "institution": "三大法人合計",
        }
    )
    _analyze_one("三大法人合計", agg_oi, returns, result_tables, datasets)

    daily = pd.concat(datasets).sort_index()
    results = pd.concat(result_tables, ignore_index=True)
    results = add_fdr_qvalues(results)

    daily.to_parquet(outdir / "daily_dataset.parquet")
    results.to_csv(outdir / "primary_results.csv", index=False, encoding="utf-8-sig")

    print("Price keys:", price_keys)
    print("Saved:", outdir / "daily_dataset.parquet")
    print("Saved:", outdir / "primary_results.csv")
    return daily, results
