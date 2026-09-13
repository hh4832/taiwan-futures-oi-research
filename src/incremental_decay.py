import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import CONFIG, ResearchConfig
from .statistics import add_fdr_by_family, compare_group_to_nongroup, hac_lag


INTERVAL_HORIZONS = {
    "day_1": 1,
    "day_2_3": 3,
    "day_4_5": 5,
    "day_6_10": 10,
    "day_11_20": 20,
}


def analyze_incremental_decay(
    foreign: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for accumulation in config.accumulation_windows:
        for rolling in config.rolling_windows:
            score = foreign[f"net_change_ratio_{accumulation}d_pr{rolling}"]
            for group, mask in {
                "PR_0_20": score.le(0.20),
                "PR_80_100": score.ge(0.80),
            }.items():
                for interval, horizon in INTERVAL_HORIZONS.items():
                    row = compare_group_to_nongroup(
                        mask.where(score.notna()), foreign[interval],
                        hac_lag(accumulation, horizon),
                    )
                    row.update({
                        "analysis_role": "confirmatory_robustness",
                        "fdr_universe": "B_incremental_decay",
                        "interval": interval,
                        "interval_end_horizon": horizon,
                        "group": group,
                        "accumulation_window": accumulation,
                        "rolling_window": rolling,
                        "fdr_family": interval,
                    })
                    rows.append(row)
    results = add_fdr_by_family(pd.DataFrame(rows), family_cols=("interval",))
    results["q_value_global"] = np.nan
    valid = results["hac_p_value"].dropna()
    if len(valid):
        results.loc[valid.index, "q_value_global"] = multipletests(valid, method="fdr_bh")[1]
    results["family_q"] = results["q_value_family"]
    results["global_q"] = results["q_value_global"]
    results["evidence_level"] = np.select(
        [results["q_value_global"].lt(.05), results["q_value_family"].lt(.05), results["hac_p_value"].lt(.05)],
        ["Level A", "Level B", "Level C"], default="Level D",
    )
    results.loc[results["hac_p_value"].isna(), "evidence_level"] = "Not evaluable"
    summary = (
        results.groupby(["interval", "group"], as_index=False)
        .agg(
            parameter_cells=("mean_diff_vs_nongroup", "count"),
            median_effect_across_12_cells=("mean_diff_vs_nongroup", "median"),
            min_effect=("mean_diff_vs_nongroup", "min"),
            max_effect=("mean_diff_vs_nongroup", "max"),
            global_fdr_cells=("q_value_global", lambda s: int((s < .05).sum())),
        )
    )
    expected = results.assign(ok=np.where(results["group"].eq("PR_0_20"), results["mean_diff_vs_nongroup"].lt(0), results["mean_diff_vs_nongroup"].gt(0)))
    counts = expected.groupby(["interval", "group"])["ok"].sum().rename("direction_consistency_cells")
    summary = summary.join(counts, on=["interval", "group"])
    return results, summary
