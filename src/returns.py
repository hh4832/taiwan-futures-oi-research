import pandas as pd


def build_forward_returns(price: pd.DataFrame, horizons=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """Build outcomes for a signal first known after the d0 close."""
    required = {"open_0050", "close_0050"}
    missing = required.difference(price.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    out = price.copy().sort_index()
    if out.index.has_duplicates:
        raise ValueError("Price index contains duplicate dates")
    close = out["close_0050"].astype(float)
    open_ = out["open_0050"].astype(float)
    next_open = open_.shift(-1)
    out["c0_o1"] = next_open / close - 1.0
    for horizon in horizons:
        out[f"c0_c{horizon}"] = close.shift(-horizon) / close - 1.0
        out[f"o1_c{horizon}"] = close.shift(-horizon) / next_open - 1.0
    return out
