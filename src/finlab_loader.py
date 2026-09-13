
import os
from typing import Dict, Iterable

import pandas as pd


FUTURES_DATASET_CANDIDATES = [
    "futures_institutional_investors_trading_summary",
]

PRICE_DATASET_CANDIDATES = {
    "open": ["etl:adj_open"],
    "close": ["etl:adj_close"],
}


# 我們研究的是「臺股期貨」，不是 ETF、小台、MSCI 或其他商品
FUTURES_SYMBOL_MAP = {
    "外資及陸資": "臺股期貨_外資及陸資",
    "投信": "臺股期貨_投信",
    "自營商": "臺股期貨_自營商",
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

    for symbol in FUTURES_SYMBOL_MAP.values():

        n = (df["symbol"] == symbol).sum()

        print(
            f"{symbol}: {n:,} rows"
        )


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

    target_symbol = FUTURES_SYMBOL_MAP[institution]

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
        df["symbol"] == target_symbol,
        [
            "date",
            "多方未平倉口數",
            "空方未平倉口數",
        ],
    ].copy()

    if work.empty:
        raise ValueError(
            f"No data found for {target_symbol}"
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

    # 同一天若資料重複，只保留最後一筆
    work = work[
        ~work.index.duplicated(keep="last")
    ]

    work["institution"] = institution

    print(
        f"{target_symbol}: "
        f"{len(work):,} observations, "
        f"{work.index.min().date()} "
        f"→ {work.index.max().date()}"
    )

    return work
