"""Figure: dev_medium full-year `wrong` answers split into noise_period vs other, by period bin and noise class."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]


def make(per_star_csv=REPO / "benchmarks/baselines/dev_medium_fullyear_v5_tables/per_star.csv", out=None):
    """Stacked bars of wrong answers (count): noise_period (red) vs other (grey), by period bin and noise class."""
    s = pd.read_csv(per_star_csv)
    w = s[s.final_outcome == "wrong"].assign(kind=lambda d: d.noise_period_match.map({True: "noise_period", False: "other"}))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, order in ((axs[0], "period_bin", BINS), (axs[1], "noise_class", None)):
        ct = pd.crosstab(w[col], w.kind).reindex(columns=["noise_period", "other"], fill_value=0)
        ct = ct.reindex(order, fill_value=0) if order else ct
        ct.plot.bar(stacked=True, ax=ax, color=["#c62828", "#9e9e9e"], rot=20 if order else 15)
        ax.set_ylabel("wrong answers (n)"); ax.set_title(f"by {col.replace('_', ' ')}")
        if not order:
            ax.set_xticklabels([t.get_text().replace("contamination_", "contam.\n").replace("source_specific_", "source-\nspecific\n") for t in ax.get_xticklabels()], fontsize=8)
    fig.suptitle(f"dev_medium full-year: {len(w)} wrong answers, {int((w.kind == 'noise_period').sum())} match the noise source's P_ref")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    return fig


if __name__ == "__main__":
    make(out=Path(__file__).parent / "figures" / "wrong_noise_period_vs_other.png")
