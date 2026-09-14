import numpy as np
import pandas as pd

from src.config import ResearchConfig
from src.features import rolling_percentile
from src.horizon_extension import (
    analyze_horizon_extension,
    continuous_horizon_regressions,
    decay_ratio,
    summarize_horizon_extension,
)
from src.institutional_divergence import build_divergence_features
from src.returns import build_incremental_returns


INTERVALS = ("day_1", "day_2_3", "day_4_5", "day_6_10", "day_11_20")


def _price(periods=30):
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.DataFrame({
        "open_0050": np.arange(100.0, 100.0 + periods),
        "close_0050": np.arange(101.0, 101.0 + periods),
    }, index=index)


def _horizon_frame(periods=300):
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    frame = pd.DataFrame(index=index)
    base = np.tile(np.linspace(0, 1, 20), periods // 20 + 1)[:periods]
    for accumulation in (1, 3, 5, 10):
        for rolling in (60, 120, 252):
            frame[f"foreign_pr_{accumulation}d_{rolling}d"] = base
            frame[f"dealer_pr_{accumulation}d_{rolling}d"] = 1 - base
            frame[f"fd_divergence_{accumulation}d_{rolling}d"] = 2 * base - 1
    for number, interval in enumerate(INTERVALS, start=1):
        frame[interval] = (base - .5) / (100 * number)
    return frame


def test_divergence_incremental_day1():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_1"], 102 / 101 - 1)


def test_divergence_incremental_day2_3():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_2_3"], 104 / 102 - 1)


def test_divergence_incremental_day4_5():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_4_5"], 106 / 104 - 1)


def test_divergence_incremental_day6_10():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_6_10"], 111 / 106 - 1)


def test_divergence_incremental_day11_20():
    assert np.isclose(build_incremental_returns(_price()).iloc[0]["day_11_20"], 121 / 111 - 1)


def test_incremental_returns_use_price_ratios():
    result = build_incremental_returns(_price()).iloc[0]
    assert np.isclose(result["day_2_3"], 104 / 102 - 1)


def test_no_cumulative_return_subtraction():
    result = build_incremental_returns(_price()).iloc[0]
    wrong = (104 / 101 - 1) - (102 / 101 - 1)
    assert not np.isclose(result["day_2_3"], wrong)


def test_horizon_extension_uses_12_cell_grid():
    results = analyze_horizon_extension(_horizon_frame())
    assert results["parameter_cell"].nunique() == 12
    assert len(results) == 2 * 2 * 12 * 5
    assert results.groupby("side").size().eq(120).all()


def test_no_new_accumulation_window():
    assert ResearchConfig().accumulation_windows == (1, 3, 5, 10)


def test_no_new_rolling_window():
    assert ResearchConfig().rolling_windows == (60, 120, 252)


def test_no_zscore():
    results = analyze_horizon_extension(_horizon_frame())
    assert not any("zscore" in column.lower() for column in results.columns)


def test_foreign_signal_definition_unchanged():
    results = analyze_horizon_extension(_horizon_frame())
    row = results.query("predictor == 'Foreign' and side == 'bearish'").iloc[0]
    assert row["n"] == 60  # Three observations per 20-date cycle are <= PR20.


def test_divergence_definition_unchanged():
    frame = _horizon_frame()
    assert np.allclose(
        frame["fd_divergence_1d_60d"],
        frame["foreign_pr_1d_60d"] - frame["dealer_pr_1d_60d"],
    )


def test_divergence_threshold_unchanged():
    results = analyze_horizon_extension(_horizon_frame())
    row = results.query("predictor == 'Foreign-Dealer divergence' and side == 'bearish'").iloc[0]
    assert row["n"] == 60  # D <= -0.60 is exactly Foreign PR <= 0.20 here.


def test_decay_ratio_day1_equals_one():
    assert decay_ratio(-.002, -.002) == 1


def test_decay_ratio_uses_absolute_effect():
    assert decay_ratio(.001, -.002) == .5


def test_decay_ratio_handles_zero_day1_effect():
    assert np.isnan(decay_ratio(.001, 0.0))


def test_horizon_extension_no_lookahead():
    prices = _price()
    original = build_incremental_returns(prices)
    changed = prices.copy()
    changed.iloc[-1] = 9999
    pd.testing.assert_frame_equal(original.iloc[:-21], changed.pipe(build_incremental_returns).iloc[:-21])


def test_divergence_percentile_uses_shifted_history():
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    percentile = rolling_percentile(values, window=3, min_periods=3)
    assert percentile.iloc[-1] == 1.0
    changed = values.copy()
    changed.iloc[-1] = 4.0
    assert rolling_percentile(changed, 3, 3).iloc[-1] == percentile.iloc[-1]


def test_same_sample_continuous_regression_grid_and_retention():
    results = continuous_horizon_regressions(_horizon_frame())
    assert len(results) == 2 * 12 * 5
    assert results["parameter_cell"].nunique() == 12
    assert results.groupby(["parameter_cell", "interval"])["n"].nunique().eq(1).all()
    assert results.loc[results["interval"].eq("day_1"), "beta_retention_ratio"].eq(1).all()


def test_head_to_head_has_both_sides_and_fixed_classification():
    results = analyze_horizon_extension(_horizon_frame())
    bearish, bullish, comparison = summarize_horizon_extension(results)
    assert set(bearish["predictor"]) == {"Foreign", "Foreign-Dealer divergence"}
    assert set(bullish["predictor"]) == {"Foreign", "Foreign-Dealer divergence"}
    assert set(comparison["side"]) == {"bearish", "bullish"}
    assert set(comparison["horizon_extension_supported"]).issubset(
        {"Yes", "No", "Mixed", "Not evaluable"}
    )


def test_build_divergence_preserves_causal_input_alignment():
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    foreign = pd.DataFrame({"net_change_ratio_1d_pr60": [np.nan, .1, .2, .3, .4]}, index=index)
    dealer = pd.DataFrame({"net_change_ratio_1d_pr60": [np.nan, .4, .3, .2, .1]}, index=index)
    config = ResearchConfig(accumulation_windows=(1,), rolling_windows=(60,), min_periods_by_window={60: 1})
    result = build_divergence_features(foreign, dealer, config)
    assert result.index.equals(index)
    assert np.isnan(result.iloc[0]["fd_divergence_1d_60d"])
