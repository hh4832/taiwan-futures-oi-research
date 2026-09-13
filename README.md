# Taiwan Futures Institutional OI Research — v3

Research scaffold for testing whether Taiwan futures institutional open-interest positioning predicts future 0050 returns.

## Research logic

Hypotheses:
1. More bullish institutional positioning may be associated with higher future Taiwan-equity returns.
2. Change in positioning may contain more information than absolute positioning.
3. Predictive effects may be concentrated in historical extremes.

Freeze-candidate finite grid:
- ΔNet / ΔLong / ΔShort, normalized by gross OI at the start of the window
- 1 / 3 / 5 / 10-trading-day endpoint changes
- prior-only rolling percentile and Z-score over 60 / 120 / 252 days
- primary outcome: d1 open -> d1 close
- secondary persistence: d1 open -> d2 / d3 / d5 / d10 / d20 close

Institution comparison:
- 外資及陸資
- 投信
- 自營商
- 三大法人合計（前三者在共同交易日逐日加總）

外資是預先指定的主要法人；投信、自營商與三大法人合計屬完整保留的比較分析。
Each institution is tested on the same finite grid. Results retain family-level BH-FDR
and add horizon-specific global BH-FDR over every pre-specified hypothesis. The legacy
`q_value_bh` column remains an alias for `q_value_family`.

The grid is pre-specified. Do not select the isolated best-performing cell.

Primary percentile groups:
- PR 0-20
- PR 20-80
- PR 80-100

Important:
- d0 close -> d+h close is a statistical forward return, not automatically tradable.
- Outcome prices require `etl:adj_open` and `etl:adj_close`; missing adjusted fields stop the run without raw-price fallback.
- next-day open based returns are included to evaluate tradability after a d0 post-close signal.
- rolling/expanding transformations must not use future observations.
- exploratory multiple testing should use FDR correction.

## Local / Colab setup

```bash
pip install -r requirements.txt
```

Set FINLAB_API_TOKEN as an environment variable or Colab secret. Never commit it.

Run:

```python
from src.pipeline import run_research
daily, results, archive = run_research()
```

## Repository policy

- `main` = current reproducible research version.
- Commit each meaningful change.
- Use tags for frozen milestones.
- Use a feature branch for a large experimental change.
- Do not commit secrets or large raw market data.

## Colab workflow

1. Open `notebooks/01_run_research.ipynb` from GitHub in Colab.
2. Edit `REPO_URL` once to point to your repository.
3. Run cells from top to bottom.
4. During validation the notebook clones `research/futures-finite-grid-robustness`;
   after an authorized merge, change this setting to `main`.
5. Colab displays the exact Git commit hash used for the analysis.
6. A timestamped archive is uploaded beneath Google Drive folder ID
   `1JrDCBf__DZA5aIlp3jxUqATsuQohtpLG`.

This makes results traceable to a specific commit.

## Data sources and archive version

- Futures OI uses the field-specific FinLab long and short keys. Net OI is
  calculated as long minus short; the field-specific net key is validation-only.
- Only `臺股期貨_外資及陸資`, `臺股期貨_投信`, and `臺股期貨_自營商` are selected.
- 0050 outcomes require `etl:adj_open` and `etl:adj_close`; no raw fallback exists.
- Archives use `YYYYMMDD_HHMMSS_v2_data_refresh_global_fdr_<commit8>` in
  Asia/Taipei time and never overwrite an existing archive.

Evidence levels: Level A passes global FDR; Level B passes family FDR only;
Level C passes only raw HAC; Level D does not pass raw HAC; missing HAC p-values
are `Not evaluable`.

## v3: Foreign robustness and Dealer divergence

The v2 finite grid remains frozen. Version
`v3_foreign_robustness_dealer_divergence` adds four bounded modules without
introducing any new accumulation window, rolling window, percentile cutoff, or
Z-score analysis:

1. Prior 1/3/5/10-day adjusted-close return conditioning of the Foreign net
   percentile signal.
2. Compounded incremental outcomes: O1→C1, C1→C3, C3→C5, C5→C10, and
   C10→C20.
3. Proper H-offset non-overlap for C2/C3/C5/C10/C20, where sampling stride is
   the outcome horizon rather than the accumulation window.
4. Exploratory Foreign–Dealer divergence, defined as matching-cell Foreign
   percentile minus Dealer percentile, across the frozen 12-cell grid.

The divergence primary FDR universe contains exactly 24 hypotheses: 12 cells ×
the pre-specified Strong bearish and Strong bullish groups. It is separate from
the v2 global FDR and can at most become a prospective-validation candidate.
Foreign+Trust is calculated by combining underlying long/short OI first, then
computing its change ratio and causal percentile; institutional percentiles are
never added together.

The canonical Drive root remains folder
`1JrDCBf__DZA5aIlp3jxUqATsuQohtpLG`. New archives use
`YYYYMMDD_HHMMSS_v3_foreign_robustness_dealer_divergence_<commit8>` and do not
overwrite v2 archives.
