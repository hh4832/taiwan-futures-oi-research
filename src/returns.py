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


def build_prior_returns(price: pd.DataFrame, windows=(1, 3, 5, 10)) -> pd.DataFrame:
    """Return C[t] / C[t-k] - 1 using only information known on signal date t."""
    if "close_0050" not in price.columns:
        raise ValueError("Missing price column: close_0050")
    out = pd.DataFrame(index=price.sort_index().index)
    close = price.sort_index()["close_0050"].astype(float)
    for window in windows:
        if window <= 0:
            raise ValueError("Prior-return windows must be positive")
        out[f"prior_{window}d_return"] = close / close.shift(window) - 1.0
    return out


def build_incremental_returns(price: pd.DataFrame) -> pd.DataFrame:
    """Build pre-specified non-overlapping future return buckets from price ratios."""
    required = {"open_0050", "close_0050"}
    missing = required.difference(price.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    ordered = price.sort_index()
    close = ordered["close_0050"].astype(float)
    open_1 = ordered["open_0050"].astype(float).shift(-1)
    out = pd.DataFrame(index=ordered.index)
    out["day_1"] = close.shift(-1) / open_1 - 1.0
    out["day_2_3"] = close.shift(-3) / close.shift(-1) - 1.0
    out["day_4_5"] = close.shift(-5) / close.shift(-3) - 1.0
    out["day_6_10"] = close.shift(-10) / close.shift(-5) - 1.0
    out["day_11_20"] = close.shift(-20) / close.shift(-10) - 1.0
    out["c1_c2"] = close.shift(-2) / close.shift(-1) - 1.0
    out["c2_c3"] = close.shift(-3) / close.shift(-2) - 1.0
    return out
