import pandas as pd
import pytest

import src.finlab_loader as loader


def test_price_candidates_forbid_raw_fallback():
    assert loader.PRICE_DATASET_CANDIDATES == {
        "open": ["etl:adj_open"],
        "close": ["etl:adj_close"],
    }


def test_missing_adjusted_price_fails(monkeypatch):
    monkeypatch.setattr(loader, "get_first_available", lambda keys: (_ for _ in ()).throw(RuntimeError("missing adjusted")))
    with pytest.raises(RuntimeError, match="missing adjusted"):
        loader.load_0050_price("0050")
