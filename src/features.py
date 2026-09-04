import numpy as np
import pandas as pd


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / den.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Z-score current value against strictly prior observations."""
    history = s.astype(float).shift(1)
    mean = history.rolling(window, min_periods=min_periods).mean()
    std = history.rolling(window, min_periods=min_periods).std(ddof=1)
    return safe_divide(s.astype(float) - mean, std)


def rolling_percentile(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile of today's value within the strictly prior rolling history."""
    values = s.astype(float).to_numpy()
    result = np.full(len(values), np.nan, dtype=float)
    for i, current in enumerate(values):
        if np.isnan(current):
            continue
        history = values[max(0, i - window):i]
        history = history[~np.isnan(history)]
        if len(history) >= min_periods:
            result[i] = np.mean(history <= current)
    return pd.Series(result, index=s.index, name=s.name)


def build_oi_features(
    df: pd.DataFrame,
    accumulation_windows=(1, 3, 5, 10),
    rolling_windows=(60, 120, 252),
    min_periods_by_window=None,
) -> pd.DataFrame:
    required = {"long_oi", "short_oi"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if min_periods_by_window is None:
        min_periods_by_window = {60: 40, 120: 80, 252: 126}

    out = df.copy().sort_index()
    if out.index.has_duplicates:
        raise ValueError("OI index contains duplicate dates")
    out["net_oi"] = out["long_oi"] - out["short_oi"]
    out["gross_oi"] = out["long_oi"] + out["short_oi"]
    out["oi_ratio"] = safe_divide(out["net_oi"], out["gross_oi"])

    derived = {}
    for accumulation in accumulation_windows:
        denominator = out["gross_oi"].shift(accumulation)
        for side in ("net", "long", "short"):
            source = out[f"{side}_oi"]
            raw_name = f"delta_{side}_{accumulation}d"
            ratio_name = f"{side}_change_ratio_{accumulation}d"
            delta = source - source.shift(accumulation)
            ratio = safe_divide(delta, denominator)
            derived[raw_name] = delta
            derived[ratio_name] = ratio
            for rolling_window in rolling_windows:
                min_periods = min_periods_by_window[rolling_window]
                derived[f"{ratio_name}_z{rolling_window}"] = rolling_zscore(
                    ratio, rolling_window, min_periods
                )
                derived[f"{ratio_name}_pr{rolling_window}"] = rolling_percentile(
                    ratio, rolling_window, min_periods
                )
    return pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
