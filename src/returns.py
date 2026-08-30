import pandas as pd


def build_forward_returns(price: pd.DataFrame, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """
    Required columns:
      open_0050, close_0050

    Statistical returns:
      close_to_close_fwd_{h} = d0 close -> d+h close
    Tradability-oriented returns:
      next_open_to_close = d1 open -> d1 close
      next_open_to_close_fwd_{h} = d1 open -> d+h close

    Signal assumption:
      institutional OI signal is known after d0 close.
    """
    required = {"open_0050", "close_0050"}
    missing = required.difference(price.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")

    out = price.copy().sort_index()
    c = out["close_0050"]
    o = out["open_0050"]

    out["next_gap"] = o.shift(-1) / c - 1.0
    out["next_open_to_close"] = c.shift(-1) / o.shift(-1) - 1.0

    for h in horizons:
        out[f"close_to_close_fwd_{h}"] = c.shift(-h) / c - 1.0
        out[f"next_open_to_close_fwd_{h}"] = c.shift(-h) / o.shift(-1) - 1.0

    return out
