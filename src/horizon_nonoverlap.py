import numpy as np
import pandas as pd

from .config import CONFIG, ResearchConfig


def nonoverlap_positions(length: int, horizon: int, offset: int) -> np.ndarray:
    """Positions whose future H-day outcome windows do not share trading days."""
    if horizon <= 0 or not 0 <= offset < horizon:
        raise ValueError("Require horizon > 0 and 0 <= offset < horizon")
    return np.arange(offset, length, horizon, dtype=int)


def all_nonoverlap_offsets(length: int, horizon: int) -> dict[int, np.ndarray]:
    return {offset: nonoverlap_positions(length, horizon, offset) for offset in range(horizon)}


def analyze_horizon_nonoverlap(
    foreign: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for horizon in config.nonoverlap_horizons:
        target_col = f"o1_c{horizon}"
        for offset, positions in all_nonoverlap_offsets(len(foreign), horizon).items():
            sampled = foreign.iloc[positions]
            for accumulation in config.accumulation_windows:
                for rolling in config.rolling_windows:
                    score = sampled[f"net_change_ratio_{accumulation}d_pr{rolling}"]
                    for group, mask in {
                        "PR_0_20": score.le(.20),
                        "PR_80_100": score.ge(.80),
                    }.items():
                        valid = score.notna() & sampled[target_col].notna()
                        group_values = sampled.loc[valid & mask, target_col]
                        nongroup = sampled.loc[valid & ~mask, target_col]
                        rows.append({
                            "analysis_role": "confirmatory_robustness",
                            "fdr_universe": "C_nonoverlap_descriptive_no_formal_test",
                            "horizon": horizon,
                            "signal": group,
                            "parameter_cell": f"{accumulation}D_x_{rolling}D",
                            "accumulation_window": accumulation,
                            "rolling_window": rolling,
                            "offset": offset,
                            "n": len(group_values),
                            "n_nongroup": len(nongroup),
                            "mean_return": group_values.mean(),
                            "mean_nongroup": nongroup.mean(),
                            "mean_diff": group_values.mean() - nongroup.mean(),
                            "positive_rate": group_values.gt(0).mean(),
                        })
    results = pd.DataFrame(rows)
    expected = np.where(
        results["signal"].eq("PR_0_20"),
        results["mean_diff"].lt(0),
        results["mean_diff"].gt(0),
    )
    results["expected_direction"] = expected
    summary = (
        results.groupby(
            ["horizon", "signal", "parameter_cell", "accumulation_window", "rolling_window"],
            as_index=False,
        )
        .agg(
            number_offsets_expected_direction=("expected_direction", "sum"),
            total_offsets=("offset", "count"),
            median_effect=("mean_diff", "median"),
            min_effect=("mean_diff", "min"),
            max_effect=("mean_diff", "max"),
        )
    )
    return results, summary
