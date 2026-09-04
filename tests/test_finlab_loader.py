import pandas as pd

from src.finlab_loader import standardize_futures_oi


def test_loader_combines_legacy_and_current_symbol_names():
    legacy_dates = pd.date_range("2007-07-02", periods=1000, freq="B")
    current_dates = pd.date_range(legacy_dates[-1] + pd.Timedelta(days=1), periods=10, freq="B")
    raw = pd.DataFrame(
        {
            "symbol": ["台指_外資"] * len(legacy_dates)
            + ["臺股期貨_外資及陸資"] * len(current_dates),
            "date": list(legacy_dates) + list(current_dates),
            "多方未平倉口數": range(1010),
            "空方未平倉口數": range(1010),
        }
    )
    result = standardize_futures_oi(raw, "外資及陸資")
    assert len(result) == 1010
    assert result.index.min() == legacy_dates.min()
    assert result.index.max() == current_dates.max()


def test_loader_rejects_overlap_between_old_and_new_labels():
    dates = pd.date_range("2007-07-02", periods=1000, freq="B")
    raw = pd.DataFrame(
        {
            "symbol": ["台指_外資"] * len(dates) + ["臺股期貨_外資及陸資"],
            "date": list(dates) + [dates[-1]],
            "多方未平倉口數": list(range(1000)) + [1],
            "空方未平倉口數": list(range(1000)) + [1],
        }
    )
    try:
        standardize_futures_oi(raw, "外資及陸資")
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected overlapping labels to fail")
