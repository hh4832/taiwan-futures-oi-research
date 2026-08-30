
import numpy as np
import pandas as pd


def safe_divide(
    num: pd.Series,
    den: pd.Series
) -> pd.Series:

    out = (
        num.astype(float)
        / den.astype(float)
    )

    return out.replace(
        [np.inf, -np.inf],
        np.nan
    )


def rolling_zscore(
    s: pd.Series,
    window: int = 252,
    min_periods: int = 60
) -> pd.Series:

    mean = s.rolling(
        window,
        min_periods=min_periods
    ).mean()

    std = s.rolling(
        window,
        min_periods=min_periods
    ).std(ddof=1)

    return safe_divide(
        s - mean,
        std
    )


def expanding_percentile(
    s: pd.Series,
    min_periods: int = 60
) -> pd.Series:
    """
    Historical percentile without look-ahead.

    At date t, ranking uses observations
    available from the beginning through t only.
    """

    s = s.astype(float)

    # Pandas native expanding rank
    # avoids the extremely slow expanding.apply()
    out = (
        s.expanding(
            min_periods=min_periods
        )
        .rank(pct=True)
    )

    return out


def build_oi_features(
    df: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60
) -> pd.DataFrame:

    required = {
        "long_oi",
        "short_oi"
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    out = (
        df.copy()
        .sort_index()
    )

    # Absolute positioning
    out["net_oi"] = (
        out["long_oi"]
        - out["short_oi"]
    )

    out["gross_oi"] = (
        out["long_oi"]
        + out["short_oi"]
    )

    out["oi_ratio"] = safe_divide(
        out["net_oi"],
        out["gross_oi"]
    )

    # Daily changes
    out["delta_long_oi"] = (
        out["long_oi"].diff()
    )

    out["delta_short_oi"] = (
        out["short_oi"].diff()
    )

    out["delta_net_oi"] = (
        out["net_oi"].diff()
    )

    # Change in directional exposure
    # normalized by previous day's gross OI
    out["oi_change_ratio"] = safe_divide(
        out["delta_net_oi"],
        out["gross_oi"].shift(1)
    )

    feature_cols = [
        "net_oi",
        "oi_ratio",
        "oi_change_ratio",
        "delta_long_oi",
        "delta_short_oi",
    ]

    for col in feature_cols:

        out[f"{col}_z{window}"] = (
            rolling_zscore(
                out[col],
                window,
                min_periods
            )
        )

        out[f"{col}_pr"] = (
            expanding_percentile(
                out[col],
                min_periods
            )
        )

    return out
