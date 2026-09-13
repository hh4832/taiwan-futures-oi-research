import pandas as pd

from src.config import ResearchConfig
from src.finlab_loader import FUTURES_SYMBOL_MAP
from src.pipeline import _build_analysis_oi


def test_build_analysis_oi_includes_three_institutions_and_total():
    dates = pd.date_range("2007-07-02", periods=1000, freq="B")
    columns = list(FUTURES_SYMBOL_MAP.values())
    long = pd.DataFrame(
        {column: pd.Series(1000 * (i + 1), index=dates) for i, column in enumerate(columns)},
        index=dates,
    )
    short = pd.DataFrame(
        {column: pd.Series(100 * (i + 1), index=dates) for i, column in enumerate(columns)},
        index=dates,
    )
    raw = {"long": long, "short": short, "net": long - short}

    result = _build_analysis_oi(raw, ResearchConfig())

    assert tuple(result) == ("外資及陸資", "投信", "自營商", "三大法人合計")
    assert all(len(frame) == len(dates) for frame in result.values())
    total = result["三大法人合計"]
    assert total["long_oi"].eq(6000).all()
    assert total["short_oi"].eq(600).all()
    assert total["institution"].eq("三大法人合計").all()
