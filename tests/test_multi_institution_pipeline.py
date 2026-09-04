import pandas as pd

from src.config import ResearchConfig
from src.pipeline import _build_analysis_oi


def test_build_analysis_oi_includes_three_institutions_and_total():
    dates = pd.date_range("2007-07-02", periods=1000, freq="B")
    symbols = {
        "外資及陸資": "台指_外資",
        "投信": "台指_投信",
        "自營商": "台指_自營商",
    }
    blocks = []
    for offset, (institution, symbol) in enumerate(symbols.items(), start=1):
        blocks.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "多方未平倉口數": 1000 * offset,
                    "空方未平倉口數": 100 * offset,
                }
            )
        )

    result = _build_analysis_oi(pd.concat(blocks, ignore_index=True), ResearchConfig())

    assert tuple(result) == ("外資及陸資", "投信", "自營商", "三大法人合計")
    assert all(len(frame) == len(dates) for frame in result.values())
    total = result["三大法人合計"]
    assert total["long_oi"].eq(6000).all()
    assert total["short_oi"].eq(600).all()
    assert total["institution"].eq("三大法人合計").all()
