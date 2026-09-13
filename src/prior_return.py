import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .config import CONFIG, ResearchConfig
from .statistics import hac_lag


def _fit_hac(target: pd.Series, predictors: pd.DataFrame, maxlags: int):
    if isinstance(predictors, pd.Series):
        predictors = predictors.to_frame()
    work = pd.concat([target.rename("target"), predictors], axis=1).dropna()
    if len(work) < max(20, len(predictors.columns) + 3):
        return None, len(work)
    model = sm.OLS(
        work["target"].astype(float),
        sm.add_constant(work[predictors.columns].astype(float)),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return model, len(work)


def analyze_prior_return_conditioning(
    foreign: pd.DataFrame,
    config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test frozen Foreign net-percentile signals after prior-return adjustment."""
    rows = []
    for accumulation in config.accumulation_windows:
        score_col = f"net_change_ratio_{accumulation}d_pr"
        for rolling in config.rolling_windows:
            score = foreign[f"{score_col}{rolling}"].astype(float)
            specs = {
                "continuous_percentile": score,
                "PR_0_20_dummy": score.le(0.20).astype(float).where(score.notna()),
                "PR_80_100_dummy": score.ge(0.80).astype(float).where(score.notna()),
            }
            for lookback in config.prior_return_windows:
                prior_col = f"prior_{lookback}d_return"
                for signal_spec, signal in specs.items():
                    unadjusted, n0 = _fit_hac(
                        foreign["o1_c1"], signal.rename("oi"),
                        hac_lag(accumulation, 1),
                    )
                    adjusted, n = _fit_hac(
                        foreign["o1_c1"],
                        pd.concat(
                            [signal.rename("oi"), foreign[prior_col].rename("prior_return")],
                            axis=1,
                        ),
                        hac_lag(accumulation, 1),
                    )
                    beta0 = np.nan if unadjusted is None else float(unadjusted.params["oi"])
                    beta = np.nan if adjusted is None else float(adjusted.params["oi"])
                    attenuation = (
                        np.nan if not np.isfinite(beta0) or beta0 == 0
                        else (beta - beta0) / abs(beta0)
                    )
                    rows.append({
                        "analysis_role": "confirmatory_robustness",
                        "fdr_universe": "A_prior_return_conditioning",
                        "lookback": lookback,
                        "signal_spec": signal_spec,
                        "accumulation_window": accumulation,
                        "rolling_window": rolling,
                        "n": n,
                        "n_unadjusted": n0,
                        "beta_oi_unadjusted": beta0,
                        "beta_oi": beta,
                        "se_oi": np.nan if adjusted is None else float(adjusted.bse["oi"]),
                        "t_oi": np.nan if adjusted is None else float(adjusted.tvalues["oi"]),
                        "p_oi": np.nan if adjusted is None else float(adjusted.pvalues["oi"]),
                        "beta_prior_return": np.nan if adjusted is None else float(adjusted.params["prior_return"]),
                        "se_prior_return": np.nan if adjusted is None else float(adjusted.bse["prior_return"]),
                        "p_prior_return": np.nan if adjusted is None else float(adjusted.pvalues["prior_return"]),
                        "r_squared": np.nan if adjusted is None else float(adjusted.rsquared),
                        "effect_attenuation_ratio": attenuation,
                        "attenuation_formula": "(beta_adjusted-beta_unadjusted)/abs(beta_unadjusted)",
                    })
    results = pd.DataFrame(rows)
    results["q_value_global"] = np.nan
    valid = results["p_oi"].dropna()
    if len(valid):
        results.loc[valid.index, "q_value_global"] = multipletests(valid, method="fdr_bh")[1]
    expected = results.assign(
        expected=np.where(
            results["signal_spec"].eq("PR_0_20_dummy"),
            results["beta_oi"].lt(0),
            results["beta_oi"].gt(0),
        )
    )
    summary = (
        expected.groupby(["lookback", "signal_spec"], as_index=False)
        .agg(
            cells=("beta_oi", "count"),
            median_beta_unadjusted=("beta_oi_unadjusted", "median"),
            median_beta_adjusted=("beta_oi", "median"),
            median_attenuation_ratio=("effect_attenuation_ratio", "median"),
            expected_direction_cells=("expected", "sum"),
            global_fdr_cells=("q_value_global", lambda s: int((s < 0.05).sum())),
        )
    )
    return results, summary
