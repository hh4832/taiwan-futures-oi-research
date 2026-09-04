import pandas as pd

from src.config import ResearchConfig
from src.pipeline import analyze_finite_grid
from src.returns import build_forward_returns


def test_finite_grid_shape_and_fdr_families():
    idx = pd.date_range("2020-01-01", periods=90, freq="B")
    oi = pd.DataFrame(
        {
            "long_oi": 1000 + pd.Series(range(90), index=idx).mod(13) * 3,
            "short_oi": 900 + pd.Series(range(90), index=idx).mod(11) * 2,
        },
        index=idx,
    )
    price = pd.DataFrame(
        {
            "open_0050": 100 + pd.Series(range(90), index=idx) * 0.1,
            "close_0050": 100.2 + pd.Series(range(90), index=idx) * 0.1,
        },
        index=idx,
    )
    returns = build_forward_returns(price, horizons=(1, 2))
    config = ResearchConfig(
        accumulation_windows=(1,),
        rolling_windows=(60,),
        min_periods_by_window={60: 40},
        primary_outcomes=(1,),
        secondary_outcomes=(2,),
    )
    _, results = analyze_finite_grid(oi, returns, config=config)
    # Per side/outcome: 3 coarse PR + 7 full PR + 7 Z-score rows.
    assert len(results) == 3 * 2 * 17
    assert set(results["outcome_role"]) == {"primary", "secondary"}
    assert results["q_value_bh"].notna().any()
