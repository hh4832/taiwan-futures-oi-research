# Taiwan Futures Institutional OI Research

Research scaffold for testing whether Taiwan futures institutional open-interest positioning predicts future 0050 returns.

## Research logic

Hypotheses:
1. More bullish institutional positioning may be associated with higher future Taiwan-equity returns.
2. Change in positioning may contain more information than absolute positioning.
3. Predictive effects may be concentrated in historical extremes.

Primary predictors:
- OI Ratio = Net OI / (Long OI + Short OI)
- OI Change Ratio = ΔNet OI / prior-day gross OI

Primary institutions:
- 外資及陸資
- 投信
- 自營商

Primary horizons:
- 1 day
- 5 days

Primary percentile groups:
- PR 0-20
- PR 20-80
- PR 80-100

Important:
- d0 close -> d+h close is a statistical forward return, not automatically tradable.
- next-day open based returns are included to evaluate tradability after a d0 post-close signal.
- Outcome prices require `etl:adj_open` and `etl:adj_close`; missing adjusted fields stop the run without raw-price fallback.
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
daily, results = run_research()
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
4. The notebook clones the current `main` branch into the temporary Colab runtime.
5. Colab displays the exact Git commit hash used for the analysis.

This makes results traceable to a specific commit.
