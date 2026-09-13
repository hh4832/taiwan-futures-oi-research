import numpy as np
import pandas as pd

from src.returns import build_forward_returns, build_incremental_returns


def _price():
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    return pd.DataFrame({
        "open_0050": np.arange(100., 125.),
        "close_0050": np.arange(101., 126.),
    }, index=idx)


def test_incremental_return_o1_c1():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_1"], 102 / 101 - 1)


def test_incremental_return_c1_c3():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_2_3"], 104 / 102 - 1)


def test_incremental_return_c3_c5():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_4_5"], 106 / 104 - 1)


def test_incremental_return_c5_c10():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_6_10"], 111 / 106 - 1)


def test_incremental_return_c10_c20():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_11_20"], 121 / 111 - 1)


def test_incremental_returns_use_price_ratios_not_return_subtraction():
    price = _price()
    incremental = build_incremental_returns(price).iloc[0]["day_2_3"]
    cumulative = build_forward_returns(price, (1, 3)).iloc[0]
    assert not np.isclose(incremental, cumulative["o1_c3"] - cumulative["o1_c1"])
