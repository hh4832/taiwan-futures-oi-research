import os
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd


FUTURES_DATASET_CANDIDATES = [
    "futures_institutional_investors_trading_summary",
]

PRICE_DATASET_CANDIDATES = {
    "open": ["etl:adj_open", "price:開盤價"],
    "close": ["etl:adj_close", "price:收盤價"],
}


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def login_finlab():
    token = os.getenv("FINLAB_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "Missing FINLAB_API_TOKEN. In Colab, set it with a secret/environment variable; "
            "do not hard-code the token into the notebook or GitHub."
        )
    from finlab import login
    login(token)


def get_first_available(keys: Iterable[str]):
    from finlab import data
    errors = []
    for key in keys:
        try:
            obj = data.get(key)
            if obj is not None:
                return obj, key
        except Exception as exc:
            errors.append((key, repr(exc)))
    raise RuntimeError(f"No FinLab key worked. Tried: {errors}")


def load_0050_price(symbol: str = "0050") -> tuple[pd.DataFrame, Dict[str, str]]:
    frames = {}
    selected = {}

    for side in ["open", "close"]:
        raw, key = get_first_available(PRICE_DATASET_CANDIDATES[side])
        raw = _normalize_index(pd.DataFrame(raw))
        if symbol not in raw.columns:
            raise KeyError(f"{symbol} not found in FinLab dataset {key}.")
        frames[side] = raw[symbol].astype(float)
        selected[side] = key

    price = pd.concat(
        [
            frames["open"].rename("open_0050"),
            frames["close"].rename("close_0050"),
        ],
        axis=1,
    ).sort_index()
    return price, selected


def inspect_futures_schema(raw) -> None:
    df = pd.DataFrame(raw)
    print("Futures dataset shape:", df.shape)
    print("Index:", type(df.index), getattr(df.index, "names", None))
    print("Columns:", type(df.columns), getattr(df.columns, "names", None))
    print("First columns:", list(df.columns[:20]))
    print(df.head())


def load_futures_raw():
    raw, key = get_first_available(FUTURES_DATASET_CANDIDATES)
    return raw, key


def standardize_futures_oi(raw, institution: str) -> pd.DataFrame:
    """
    FinLab futures table schemas can change. This adapter handles common shapes but
    intentionally fails loudly if the required long/short open-interest fields
    cannot be identified.

    The first Colab run prints the raw schema. If FinLab changes naming, update only
    this adapter rather than the analysis logic.
    """
    df = pd.DataFrame(raw).copy()

    # MultiIndex columns are common in institutional datasets.
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            flat.append(" | ".join([str(x) for x in col if str(x) != ""]))
        df.columns = flat

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Keep only columns mentioning requested institution when possible.
    inst_cols = [c for c in df.columns if institution in str(c)]
    work = df[inst_cols].copy() if inst_cols else df.copy()

    def find_col(keywords_all, keywords_any):
        for c in work.columns:
            s = str(c)
            if all(k in s for k in keywords_all) and any(k in s for k in keywords_any):
                return c
        return None

    long_col = find_col(["多"], ["未平倉", "未沖銷", "OI"])
    short_col = find_col(["空"], ["未平倉", "未沖銷", "OI"])

    if long_col is None or short_col is None:
        raise ValueError(
            "Unable to identify long/short OI columns automatically. "
            "Run inspect_futures_schema(raw), then adjust standardize_futures_oi()."
        )

    out = pd.DataFrame({
        "long_oi": pd.to_numeric(work[long_col], errors="coerce"),
        "short_oi": pd.to_numeric(work[short_col], errors="coerce"),
    })
    out["institution"] = institution
    return out
