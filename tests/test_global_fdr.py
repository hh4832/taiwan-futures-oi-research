import numpy as np
import pandas as pd

from src.statistics import add_fdr_by_family, add_global_fdr


def _results():
    rows = []
    for horizon in (1, 2, 3, 5, 10, 20):
        for institution in ("外資及陸資", "投信"):
            for family in ("percentile", "zscore"):
                for p_value in (0.001, 0.04, 0.50):
                    rows.append(
                        {
                            "outcome_horizon": horizon,
                            "institution": institution,
                            "fdr_family": family,
                            "hac_p_value": p_value,
                        }
                    )
    rows.append(
        {"outcome_horizon": 1, "institution": "外資及陸資", "fdr_family": "percentile", "hac_p_value": np.nan}
    )
    return pd.DataFrame(rows)


def _corrected():
    family = add_fdr_by_family(_results(), family_cols=("institution", "fdr_family"))
    return add_global_fdr(family)


def test_global_fdr_uses_all_primary_hypotheses():
    out = _corrected()
    primary = out[out["outcome_horizon"].eq(1)]
    assert primary["q_value_global"].notna().sum() == primary["hac_p_value"].notna().sum()


def test_global_fdr_not_filtered_by_significance():
    out = _corrected()
    row = out[out["hac_p_value"].eq(0.50)].iloc[0]
    assert pd.notna(row["q_value_global"])


def test_family_and_global_fdr_are_distinct():
    out = _corrected()
    assert (out["q_value_family"] != out["q_value_global"]).fillna(False).any()


def test_secondary_global_fdr_is_horizon_specific():
    base = _results()
    changed = base.copy()
    changed.loc[changed["outcome_horizon"].eq(2), "hac_p_value"] = 0.99
    out1 = add_global_fdr(add_fdr_by_family(base, ("institution", "fdr_family")))
    out2 = add_global_fdr(add_fdr_by_family(changed, ("institution", "fdr_family")))
    for horizon in (3, 5, 10, 20):
        left = out1.loc[out1["outcome_horizon"].eq(horizon), "q_value_global"]
        right = out2.loc[out2["outcome_horizon"].eq(horizon), "q_value_global"]
        assert np.allclose(left, right, equal_nan=True)


def test_evidence_level_assignment_and_missing():
    frame = pd.DataFrame(
        {
            "outcome_horizon": [1] * 5,
            "hac_p_value": [0.001, 0.03, 0.04, 0.50, np.nan],
            "q_value_family": [0.01, 0.03, 0.20, 0.60, np.nan],
            "q_value_bh": [0.01, 0.03, 0.20, 0.60, np.nan],
            "significant_raw_05": [True, True, True, False, False],
            "significant_family_fdr_05": [True, True, False, False, False],
        }
    )
    out = add_global_fdr(frame)
    assert list(out["evidence_level"]) == ["Level A", "Level B", "Level C", "Level D", "Not evaluable"]
