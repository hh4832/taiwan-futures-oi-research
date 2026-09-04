import numpy as np
import pandas as pd

from src.returns import build_forward_returns
from src.statistics import compare_group_to_nongroup, hac_lag


def test_o1_to_ch_uses_next_trading_open():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    price = pd.DataFrame(
        {"open_0050": [100.0, 110.0, 120.0], "close_0050": [105.0, 115.0, 126.0]},
        index=idx,
    )
    result = build_forward_returns(price, horizons=(1, 2))
    assert np.isclose(result.iloc[0]["o1_c1"], 115.0 / 110.0 - 1)
    assert np.isclose(result.iloc[0]["o1_c2"], 126.0 / 110.0 - 1)


def test_hac_lag_covers_signal_and_outcome_overlap():
    assert hac_lag(10, 1) == 9
    assert hac_lag(3, 20) == 19


def test_hac_coefficient_is_group_minus_nongroup():
    target = pd.Series([0.01] * 10 + [-0.01] * 10)
    group = pd.Series([True] * 10 + [False] * 10)
    result = compare_group_to_nongroup(group, target, maxlags=0)
    assert np.isclose(result["hac_coef"], 0.02)
    assert np.isclose(result["mean_diff_vs_nongroup"], 0.02)
