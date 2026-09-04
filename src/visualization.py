from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_parameter_surface_figures(results: pd.DataFrame, figure_dir: str | Path) -> list[Path]:
    """Plot primary percentile surfaces without ranking or selecting a best cell."""
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary = results.loc[
        results["outcome"].eq("o1_c1")
        & results["standardization"].eq("percentile")
        & results["bin_set"].eq("coarse")
        & results["group"].isin(["PR_0_20", "PR_80_100"])
    ]
    paths = []
    for (institution, side, group), frame in primary.groupby(
        ["institution", "side", "group"]
    ):
        for metric in ("mean_return", "hac_coef"):
            surface = frame.pivot_table(
                index="rolling_window",
                columns="accumulation_window",
                values=metric,
                aggfunc="first",
            ).sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            image = ax.imshow(surface.to_numpy(), aspect="auto", cmap="RdBu_r")
            ax.set_xticks(range(len(surface.columns)), labels=surface.columns)
            ax.set_yticks(range(len(surface.index)), labels=surface.index)
            ax.set_xlabel("OI change window (trading days)")
            ax.set_ylabel("Rolling standardization window")
            ax.set_title(
                f"{institution} | {side} | {group} | O1-C1 | {metric}"
            )
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            institution_dir = output / institution
            institution_dir.mkdir(parents=True, exist_ok=True)
            path = institution_dir / (
                f"{institution}_{side}_{group}_o1_c1_{metric}_surface.png"
            )
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths
