from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ResearchConfig:
    """Pre-specified finite grid for the futures OI freeze candidate."""

    start_date: str = "2007-07-02"
    target_symbol: str = "0050"
    timezone: str = "Asia/Taipei"
    random_seed: int = 4832
    institutions: Tuple[str, ...] = ("外資及陸資", "投信", "自營商")
    primary_institutions: Tuple[str, ...] = ("外資及陸資",)
    comparison_institutions: Tuple[str, ...] = (
        "外資及陸資", "投信", "自營商", "三大法人合計"
    )
    predictor_sides: Tuple[str, ...] = ("net", "long", "short")
    accumulation_windows: Tuple[int, ...] = (1, 3, 5, 10)
    rolling_windows: Tuple[int, ...] = (60, 120, 252)
    min_periods_by_window: Mapping[int, int] = None
    primary_outcomes: Tuple[int, ...] = (1,)
    secondary_outcomes: Tuple[int, ...] = (2, 3, 5, 10, 20)
    percentile_bins: Tuple[float, ...] = (0.0, 0.20, 0.80, 1.0)
    percentile_labels: Tuple[str, ...] = ("PR_0_20", "PR_20_80", "PR_80_100")
    full_percentile_bins: Tuple[float, ...] = (
        0.0, 0.05, 0.20, 0.40, 0.60, 0.80, 0.95, 1.0
    )
    full_percentile_labels: Tuple[str, ...] = (
        "PR_0_5", "PR_5_20", "PR_20_40", "PR_40_60",
        "PR_60_80", "PR_80_95", "PR_95_100",
    )
    zscore_bins: Tuple[float, ...] = (
        float("-inf"), -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, float("inf")
    )
    zscore_labels: Tuple[str, ...] = (
        "Z_LT_M2_5", "Z_M2_5_M1_5", "Z_M1_5_M0_5", "Z_M0_5_P0_5",
        "Z_P0_5_P1_5", "Z_P1_5_P2_5", "Z_GE_P2_5",
    )
    drive_folder_id: str = "1JrDCBf__DZA5aIlp3jxUqATsuQohtpLG"
    research_version: str = "v2_data_refresh_global_fdr"

    def __post_init__(self):
        if self.min_periods_by_window is None:
            object.__setattr__(
                self, "min_periods_by_window", {60: 40, 120: 80, 252: 126}
            )

    @property
    def outcome_horizons(self) -> Tuple[int, ...]:
        return self.primary_outcomes + self.secondary_outcomes


CONFIG = ResearchConfig()
