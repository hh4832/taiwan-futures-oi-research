import numpy as np

from src.horizon_nonoverlap import all_nonoverlap_offsets, nonoverlap_positions


def test_nonoverlap_uses_outcome_horizon():
    positions = nonoverlap_positions(30, horizon=5, offset=0)
    assert np.all(np.diff(positions) == 5)


def test_nonoverlap_all_offsets_generated():
    assert set(all_nonoverlap_offsets(30, 5)) == set(range(5))


def test_nonoverlap_no_shared_outcome_windows():
    for positions in all_nonoverlap_offsets(50, 10).values():
        windows = [set(range(t + 1, t + 11)) for t in positions]
        assert all(windows[i].isdisjoint(windows[i + 1]) for i in range(len(windows) - 1))


def test_nonoverlap_offset_counts():
    offsets = all_nonoverlap_offsets(23, 5)
    assert sum(map(len, offsets.values())) == 23
    assert max(map(len, offsets.values())) - min(map(len, offsets.values())) <= 1
