# Taiwan Futures Institutional OI Research

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

Primary institutions:
- 外資及陸資
- 投信
- 自營商

The grid is pre-specified. Do not select the isolated best-performing cell.

Primary percentile groups:
- PR 0-20
- PR 20-80
- PR 80-100

Important:
- d0 close -> d+h close is a statistical forward return, not automatically tradable.
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
4. The finite-grid notebook clones `research/futures-finite-grid-robustness`.
5. Colab displays the exact Git commit hash used for the analysis.
6. A timestamped archive is uploaded beneath Google Drive folder ID
   `1kRfLhTLdHevVuFkEzdZH9M5wEui7JuEg`.

This makes results traceable to a specific commit.
