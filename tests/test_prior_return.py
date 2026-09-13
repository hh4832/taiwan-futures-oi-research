import numpy as np
import pandas as pd

from src.returns import build_prior_returns


def _price():
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    return pd.DataFrame({"close_0050": np.arange(100., 112.)}, index=idx)


def test_prior_return_uses_only_past_prices():
    price = _price()
    original = build_prior_returns(price, (1, 3, 5, 10))
    changed = price.copy()
    changed.iloc[-1, 0] = 9999
    revised = build_prior_returns(changed, (1, 3, 5, 10))
    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_prior_return_1d_definition():
    out = build_prior_returns(_price(), (1,))
    assert np.isclose(out.iloc[5, 0], 105 / 104 - 1)


def test_prior_return_3d_definition():
    out = build_prior_returns(_price(), (3,))
    assert np.isclose(out.iloc[5, 0], 105 / 102 - 1)


def test_prior_return_5d_definition():
    out = build_prior_returns(_price(), (5,))
    assert np.isclose(out.iloc[5, 0], 105 / 100 - 1)


def test_prior_return_10d_definition():
    out = build_prior_returns(_price(), (10,))
    assert np.isclose(out.iloc[10, 0], 110 / 100 - 1)


def test_prior_return_regression_alignment():
    out = build_prior_returns(_price(), (1,))
    assert out.index.equals(_price().index)
    assert out.iloc[0].isna().all()


def test_prior_return_no_lookahead():
    price = _price()
    out = build_prior_returns(price, (3,))
    assert np.isclose(out.iloc[7, 0], price.iloc[7, 0] / price.iloc[4, 0] - 1)
