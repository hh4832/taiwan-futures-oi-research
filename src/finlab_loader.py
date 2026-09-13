import os
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


LONG_OI_KEY = "futures_institutional_investors_trading_summary:多方未平倉口數"
SHORT_OI_KEY = "futures_institutional_investors_trading_summary:空方未平倉口數"
NET_OI_KEY = "futures_institutional_investors_trading_summary:多空未平倉口數淨額"

PRICE_DATASET_CANDIDATES = {
    "open": ["etl:adj_open"],
    "close": ["etl:adj_close"],
}

FUTURES_SYMBOL_MAP = {
    "外資及陸資": "臺股期貨_外資及陸資",
    "投信": "臺股期貨_投信",
    "自營商": "臺股期貨_自營商",
}


def _normalize_index(df: pd.DataFrame, source: str = "dataset") -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out.index = pd.to_datetime(out.index)
    duplicate_dates = out.index[out.index.duplicated(keep=False)].unique()
    if len(duplicate_dates):
        preview = [str(value.date()) for value in duplicate_dates[:10]]
        raise ValueError(f"Duplicate dates in {source}: {preview}")
    return out.sort_index()


def login_finlab():
    token = os.getenv("FINLAB_API_TOKEN")
    if not token:
        raise EnvironmentError("Missing FINLAB_API_TOKEN.")
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


def load_0050_price(
    symbol: str = "0050",
) -> tuple[pd.DataFrame, Dict[str, str]]:
    frames = {}
    selected = {}
    for side in ["open", "close"]:
        raw, key = get_first_available(PRICE_DATASET_CANDIDATES[side])
        raw = _normalize_index(pd.DataFrame(raw), key)
        if symbol not in raw.columns:
            raise KeyError(f"{symbol} not found in {key}")
        values = pd.to_numeric(raw[symbol], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"{symbol} in {key} contains no numeric observations")
        frames[side] = values.astype(float)
        selected[side] = key

    price = pd.concat(
        [
            frames["open"].rename("open_0050"),
            frames["close"].rename("close_0050"),
        ],
        axis=1,
    ).sort_index()
    return price, selected


def load_futures_raw() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load continuously updated, field-specific FinLab OI matrices."""
    from finlab import data

    keys = {"long": LONG_OI_KEY, "short": SHORT_OI_KEY, "net": NET_OI_KEY}
    frames = {}
    for side, key in keys.items():
        raw = data.get(key)
        if raw is None:
            raise RuntimeError(f"FinLab returned no data for {key}")
        frame = _normalize_index(pd.DataFrame(raw), key)
        if frame.empty:
            raise ValueError(f"FinLab returned an empty DataFrame for {key}")
        frames[side] = frame
    return frames, keys


def inspect_futures_schema(raw: Mapping[str, pd.DataFrame]) -> None:
    for side in ("long", "short", "net"):
        frame = pd.DataFrame(raw[side])
        print(f"Futures {side} shape: {frame.shape}")
        print(f"Futures {side} date range: {frame.index.min()} → {frame.index.max()}")
        missing = [column for column in FUTURES_SYMBOL_MAP.values() if column not in frame]
        if missing:
            raise KeyError(f"Missing target columns in {side} OI field: {missing}")


def standardize_futures_oi(
    raw: Mapping[str, pd.DataFrame],
    institution: str,
) -> pd.DataFrame:
    if institution not in FUTURES_SYMBOL_MAP:
        raise ValueError(f"Unknown institution: {institution}")
    if not {"long", "short"}.issubset(raw):
        raise ValueError("Field-specific OI input must contain long and short frames")

    target = FUTURES_SYMBOL_MAP[institution]
    selected = {}
    valid_dates = {}
    for side in ("long", "short"):
        frame = _normalize_index(pd.DataFrame(raw[side]), f"{side}_oi")
        if target not in frame.columns:
            raise KeyError(f"Target column {target} not found in {side} OI field")
        values = pd.to_numeric(frame[target], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"Numeric conversion removed all {side} OI for {institution}")
        selected[side] = values
        valid_dates[side] = values.dropna().index

    if not valid_dates["long"].equals(valid_dates["short"]):
        long_only = valid_dates["long"].difference(valid_dates["short"])
        short_only = valid_dates["short"].difference(valid_dates["long"])
        raise ValueError(
            f"Long/short date alignment failed for {institution}: "
            f"long_only={len(long_only)}, short_only={len(short_only)}"
        )

    work = pd.concat(
        [selected["long"].rename("long_oi"), selected["short"].rename("short_oi")],
        axis=1,
    ).sort_index()
    first_valid = valid_dates["long"].min()
    last_valid = valid_dates["long"].max()
    work = work.loc[first_valid:last_valid]
    work["institution"] = institution

    if "net" in raw:
        net_frame = _normalize_index(pd.DataFrame(raw["net"]), "net_oi_validation")
        if target not in net_frame.columns:
            raise KeyError(f"Target column {target} not found in net OI validation field")
        reported = pd.to_numeric(net_frame[target], errors="coerce").reindex(work.index)
        comparable = reported.notna()
        calculated = work["long_oi"] - work["short_oi"]
        mismatch = comparable & ~np.isclose(calculated, reported, equal_nan=False)
        if mismatch.any():
            raise ValueError(
                f"Long - short != reported net OI for {institution}: "
                f"{int(mismatch.sum())} mismatches"
            )

    print(
        f"{target}: {len(work):,} observations, "
        f"long {valid_dates['long'].min().date()} → {valid_dates['long'].max().date()}, "
        f"short {valid_dates['short'].min().date()} → {valid_dates['short'].max().date()}"
    )
    if len(work) < 1_000:
        raise ValueError(f"Insufficient OI history for {institution}: {len(work):,} rows")
    work.attrs.update(
        source_api_type="field_specific",
        target_column=target,
        first_long_date=valid_dates["long"].min(),
        last_long_date=valid_dates["long"].max(),
        first_short_date=valid_dates["short"].min(),
        last_short_date=valid_dates["short"].max(),
    )
    return work


def validate_latest_non_null_dates(
    oi_by_institution: Mapping[str, pd.DataFrame],
) -> None:
    """Fail when one target institution silently stops before the others."""
    individual = [frame for frame in oi_by_institution.values() if frame.attrs.get("target_column")]
    if not individual:
        return
    latest = max(frame.attrs["last_long_date"] for frame in individual)
    for frame in individual:
        institution = str(frame["institution"].iloc[0])
        for side in ("long", "short"):
            value = frame.attrs[f"last_{side}_date"]
            if value != latest:
                raise ValueError(
                    f"Latest non-null {side} date for {institution} is {value.date()}, "
                    f"but target-institution maximum is {latest.date()}"
                )
