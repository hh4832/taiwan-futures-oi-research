from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2007-07-02"
    target_symbol: str = "0050"
    rolling_window: int = 252
    min_periods: int = 60
    horizons: Tuple[int, ...] = (1, 3, 5, 10, 20)

    institutions: Tuple[str, ...] = (
        "外資及陸資",
        "投信",
        "自營商",
    )

    # Primary, pre-specified analysis.
    primary_predictors: Tuple[str, ...] = (
        "oi_ratio",
        "oi_change_ratio",
    )
    primary_horizons: Tuple[int, ...] = (1, 5)
    primary_pr_bins: Tuple[float, ...] = (0.0, 0.2, 0.8, 1.0)

    # Exploratory seven-bin percentile analysis.
    exploratory_pr_bins: Tuple[float, ...] = (
        0.0, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 1.0
    )

CONFIG = ResearchConfig()
