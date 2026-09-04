import numpy as np
import pandas as pd

from src.features import build_oi_features, rolling_percentile, rolling_zscore


def test_multiday_changes_use_endpoint_difference_and_lagged_gross():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    oi = pd.DataFrame(
        {"long_oi": [100, 103, 105, 110, 108, 115],
         "short_oi": [80, 82, 81, 84, 86, 83]},
        index=idx,
    )
    result = build_oi_features(
        oi, accumulation_windows=(3,), rolling_windows=(3,),
        min_periods_by_window={3: 2},
    )
    date = idx[4]
    expected_net_change = (108 - 86) - (103 - 82)
    expected_ratio = expected_net_change / (103 + 82)
    assert result.loc[date, "delta_net_3d"] == expected_net_change
    assert np.isclose(result.loc[date, "net_change_ratio_3d"], expected_ratio)
    assert np.isclose(
        result.loc[date, "delta_net_3d"],
        result.loc[date, "delta_long_3d"] - result.loc[date, "delta_short_3d"],
    )


def test_rolling_scores_do_not_use_current_observation():
    s = pd.Series([1.0, 2.0, 3.0, 100.0])
    z = rolling_zscore(s, window=3, min_periods=3)
    pr = rolling_percentile(s, window=3, min_periods=3)
    expected_z = (100.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=1)
    assert np.isclose(z.iloc[-1], expected_z)
    assert pr.iloc[-1] == 1.0
