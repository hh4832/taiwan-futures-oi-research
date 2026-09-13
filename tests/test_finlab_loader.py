import pandas as pd
import pytest

from src import finlab_loader as loader


def _wide_history(periods=1100):
    dates = pd.date_range("2007-07-02", periods=periods, freq="B")
    columns = list(loader.FUTURES_SYMBOL_MAP.values())
    long = pd.DataFrame({column: range(periods) for column in columns}, index=dates)
    short = pd.DataFrame({column: range(periods) for column in columns}, index=dates)
    net = long - short
    return {"long": long, "short": short, "net": net}


def test_loader_uses_field_specific_oi_keys(monkeypatch):
    requested = []
    frames = _wide_history()

    class FakeData:
        @staticmethod
        def get(key):
            requested.append(key)
            lookup = {
                loader.LONG_OI_KEY: frames["long"],
                loader.SHORT_OI_KEY: frames["short"],
                loader.NET_OI_KEY: frames["net"],
            }
            return lookup[key]

    import sys
    import types
    monkeypatch.setitem(sys.modules, "finlab", types.SimpleNamespace(data=FakeData))
    _, keys = loader.load_futures_raw()
    assert set(requested) == {loader.LONG_OI_KEY, loader.SHORT_OI_KEY, loader.NET_OI_KEY}
    assert "futures_institutional_investors_trading_summary" not in requested
    assert keys["long"] == loader.LONG_OI_KEY


def test_loader_selects_correct_taifex_columns():
    assert loader.FUTURES_SYMBOL_MAP == {
        "外資及陸資": "臺股期貨_外資及陸資",
        "投信": "臺股期貨_投信",
        "自營商": "臺股期貨_自營商",
    }
    result = loader.standardize_futures_oi(_wide_history(), "外資及陸資")
    assert list(result.columns) == ["long_oi", "short_oi", "institution"]


def test_loader_aligns_long_and_short_dates():
    frames = _wide_history()
    frames["short"] = frames["short"].iloc[:-1]
    with pytest.raises(ValueError, match="alignment"):
        loader.standardize_futures_oi(frames, "外資及陸資")


def test_loader_fails_if_target_column_missing():
    frames = _wide_history()
    frames["long"] = frames["long"].drop(columns="臺股期貨_外資及陸資")
    with pytest.raises(KeyError, match="Target column"):
        loader.standardize_futures_oi(frames, "外資及陸資")


def test_loader_fails_on_duplicate_dates():
    frames = _wide_history()
    frames["long"] = pd.concat([frames["long"], frames["long"].iloc[[-1]]])
    with pytest.raises(ValueError, match="Duplicate dates"):
        loader.standardize_futures_oi(frames, "外資及陸資")


def test_loader_preserves_full_history():
    result = loader.standardize_futures_oi(_wide_history(periods=5000), "外資及陸資")
    assert len(result) == 5000
    assert result.index.year.min() == 2007
    assert result.index.year.max() >= 2026


def test_loader_checks_latest_non_null_date_per_institution():
    frames = _wide_history()
    outputs = {
        institution: loader.standardize_futures_oi(frames, institution)
        for institution in loader.FUTURES_SYMBOL_MAP
    }
    outputs["投信"].attrs["last_short_date"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Latest non-null short date"):
        loader.validate_latest_non_null_dates(outputs)


def test_price_candidates_forbid_raw_fallback():
    assert loader.PRICE_DATASET_CANDIDATES == {
        "open": ["etl:adj_open"],
        "close": ["etl:adj_close"],
    }
