
import os
from typing import Dict, Iterable

import pandas as pd


FUTURES_DATASET_CANDIDATES = [
    "futures_institutional_investors_trading_summary",
]

PRICE_DATASET_CANDIDATES = {
    "open": ["etl:adj_open", "price:開盤價"],
    "close": ["etl:adj_close", "price:收盤價"],
}


# 我們研究的是「臺股期貨」，不是 ETF、小台、MSCI 或其他商品
FUTURES_SYMBOL_MAP = {
    # FinLab changed the Taiwan-index-futures labels on 2024-04-17.
    # Both names are required to preserve the continuous history.
    "外資及陸資": ("台指_外資", "臺股期貨_外資及陸資"),
    "投信": ("台指_投信", "臺股期貨_投信"),
    "自營商": ("台指_自營商", "臺股期貨_自營商"),
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
            "Missing FINLAB_API_TOKEN."
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

    raise RuntimeError(
        f"No FinLab key worked. Tried: {errors}"
    )


def load_0050_price(
    symbol: str = "0050"
) -> tuple[pd.DataFrame, Dict[str, str]]:

    frames = {}
    selected = {}

    for side in ["open", "close"]:

        raw, key = get_first_available(
            PRICE_DATASET_CANDIDATES[side]
        )

        raw = _normalize_index(
            pd.DataFrame(raw)
        )

        if symbol not in raw.columns:
            raise KeyError(
                f"{symbol} not found in {key}"
            )

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
    print("Columns:", list(df.columns))
    print()

    print("Target symbols:")

    for symbols in FUTURES_SYMBOL_MAP.values():
        for symbol in symbols:
            n = (df["symbol"] == symbol).sum()
            print(f"{symbol}: {n:,} rows")


def load_futures_raw():

    raw, key = get_first_available(
        FUTURES_DATASET_CANDIDATES
    )

    return raw, key


def standardize_futures_oi(
    raw,
    institution: str
) -> pd.DataFrame:

    if institution not in FUTURES_SYMBOL_MAP:
        raise ValueError(
            f"Unknown institution: {institution}"
        )

    target_symbols = FUTURES_SYMBOL_MAP[institution]

    df = pd.DataFrame(raw).copy()

    required = {
        "symbol",
        "date",
        "多方未平倉口數",
        "空方未平倉口數",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing futures columns: {sorted(missing)}"
        )

    # Exact match，非常重要
    work = df.loc[
        df["symbol"].isin(target_symbols),
        [
            "date",
            "多方未平倉口數",
            "空方未平倉口數",
        ],
    ].copy()

    if work.empty:
        raise ValueError(
            f"No data found for {target_symbols}"
        )

    work["date"] = pd.to_datetime(work["date"])

    work["long_oi"] = pd.to_numeric(
        work["多方未平倉口數"],
        errors="coerce"
    )

    work["short_oi"] = pd.to_numeric(
        work["空方未平倉口數"],
        errors="coerce"
    )

    work = (
        work
        .set_index("date")
        [["long_oi", "short_oi"]]
        .sort_index()
    )

    duplicate_dates = work.index[work.index.duplicated(keep=False)].unique()
    if len(duplicate_dates):
        raise ValueError(
            "Legacy/current futures symbols overlap on dates: "
            f"{[str(x.date()) for x in duplicate_dates[:10]]}"
        )

    work["institution"] = institution

    print(
        f"{' + '.join(target_symbols)}: "
        f"{len(work):,} observations, "
        f"{work.index.min().date()} "
        f"→ {work.index.max().date()}"
    )

    if len(work) < 1_000:
        raise ValueError(
            f"Insufficient OI history for {institution}: {len(work):,} rows. "
            "Expected legacy and current FinLab symbols to produce at least 1,000 rows."
        )

    return work
