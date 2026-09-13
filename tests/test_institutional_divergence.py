import numpy as np
import pandas as pd

from src.config import ResearchConfig
from src.institutional_divergence import (
    analyze_divergence_surface,
    assign_divergence_bucket,
    build_divergence_features,
)


def _frames(periods=80):
    idx = pd.date_range("2020-01-01", periods=periods, freq="B")
    foreign, dealer = pd.DataFrame(index=idx), pd.DataFrame(index=idx)
    for a in (1, 3, 5, 10):
        for w in (60, 120, 252):
            foreign[f"net_change_ratio_{a}d_pr{w}"] = np.linspace(0, 1, periods)
            dealer[f"net_change_ratio_{a}d_pr{w}"] = np.linspace(1, 0, periods)
    return foreign, dealer


def test_divergence_definition():
    foreign, dealer = _frames()
    out = build_divergence_features(foreign, dealer)
    assert np.isclose(out.iloc[10]["fd_divergence_1d_60d"], foreign.iloc[10, 0] - dealer.iloc[10, 0])


def test_divergence_range_minus1_plus1():
    foreign, dealer = _frames()
    out = build_divergence_features(foreign, dealer)
    values = out.filter(like="fd_divergence")
    assert values.min().min() >= -1 and values.max().max() <= 1


def test_divergence_same_accumulation_window():
    foreign, dealer = _frames()
    out = build_divergence_features(foreign, dealer)
    assert "fd_divergence_3d_60d" in out


def test_divergence_same_rolling_window():
    foreign, dealer = _frames()
    out = build_divergence_features(foreign, dealer)
    assert "fd_divergence_3d_120d" in out


def test_divergence_12_cell_grid():
    foreign, dealer = _frames()
    out = build_divergence_features(foreign, dealer)
    assert len(out.filter(like="fd_divergence").columns) == 12


def test_divergence_bucket_boundaries():
    values = pd.Series([-1, -.60, -.59, -.20, -.19, .19, .20, .59, .60, 1])
    assert list(assign_divergence_bucket(values)) == [
        "Strong bearish", "Strong bearish", "Moderate bearish", "Moderate bearish",
        "Neutral", "Neutral", "Moderate bullish", "Moderate bullish",
        "Strong bullish", "Strong bullish",
    ]


def test_divergence_no_zscore():
    foreign, dealer = _frames()
    assert not any("zscore" in c or "_z" in c for c in build_divergence_features(foreign, dealer))


def test_divergence_global_fdr_universe():
    foreign, dealer = _frames(120)
    frame = build_divergence_features(foreign, dealer)
    frame["o1_c1"] = np.sin(np.arange(len(frame))) / 100
    _, primary, _ = analyze_divergence_surface(frame)
    assert len(primary) == 24
    assert primary["fdr_universe"].eq("D_divergence_primary_24").all()


def test_divergence_not_added_to_v2_global_fdr():
    foreign, dealer = _frames(120)
    frame = build_divergence_features(foreign, dealer)
    frame["o1_c1"] = np.sin(np.arange(len(frame))) / 100
    _, primary, _ = analyze_divergence_surface(frame)
    assert "outcome_horizon" not in primary.columns
