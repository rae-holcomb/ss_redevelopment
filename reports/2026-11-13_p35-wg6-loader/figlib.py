"""Figure helpers for the P35 report. Only ``fig_examples`` reads the drive; the rest use committed CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
FIG = HERE / "figures"
ORDER = ["FL", "ECL", "ELL", "T2CEP", "RRLYR", "HB", "DSCT", "LPV", "CV"]
COL = {"FL": "#1b9e77", "ECL": "#d95f02", "ELL": "#e6ab02", "T2CEP": "#7570b3", "RRLYR": "#e7298a",
       "HB": "#66a61e", "DSCT": "#a6761d", "LPV": "#666666", "CV": "#1f78b4"}
SEASON_DAYS = 71.0


def manifest() -> pd.DataFrame:
    """The committed v0 manifest."""
    return pd.read_csv(ROOT / "datasets/wg6_roman_v0.csv", low_memory=False)


def fig_examples(man: pd.DataFrame) -> Path:
    """One loader-produced season-1 F146 light curve per class (first dev star of each; FL: a ~2 d rotator). Needs the drive."""
    from wg6 import load_wg6_roman, resampled_root
    root = resampled_root()
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5))
    for ax, cls in zip(axes.ravel(), ORDER):
        g = man[(man.cls == cls) & (man.eval_set == "dev") & (man.season == 1)]
        if cls == "FL":
            g = g[g.is_rotator & (g.wfmag == 17) & g.P_true.between(1.5, 2.5)]
        r = g.iloc[0]
        lc = load_wg6_roman(root / r.zip, r.member, "F146", season=1)
        mag = lc.meta["median_mag"]
        ax.plot(lc.time, lc.flux, ".", ms=1.2, color=COL[cls])
        P = "no catalogue period" if pd.isna(r.P_true) else f"P_true = {r.P_true:.4g} d"
        if cls in ("ECL", "ELL"):
            P += f" (photometric P/2 = {r.P_phot_expected:.4g} d)"
        ax.set_title(f"{cls}  {r.obj_id}\n{P};  F146 ~ {mag:.1f}", fontsize=8.5)
        ax.set_xlabel("days since season start"); ax.set_ylabel("rel. flux − 1")
    fig.tight_layout()
    p = FIG / "examples_per_class.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_period_hist(man: pd.DataFrame) -> Path:
    """Histogram of P_true per class (one entry per object) with the 71-d season span marked."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5))
    bins = np.logspace(-2, 3, 46)
    for ax, cls in zip(axes.ravel(), ORDER):
        g = man[man.cls == cls].drop_duplicates("obj_id")
        P = g.P_true.dropna()
        ax.hist(P, bins=bins, color=COL[cls])
        ax.axvline(SEASON_DAYS, color="k", ls="--", lw=1)
        ax.axvline(SEASON_DAYS / 2, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        n_long = int((P > SEASON_DAYS / 2).sum())
        ax.set_title(f"{cls}: {len(P)}/{len(g)} objects with P_true; {n_long} above R/2" if len(P) else f"{cls}: no period", fontsize=9)
        ax.set_xlabel("P_true [d]"); ax.set_ylabel("objects")
        if cls == "FL":
            ax.text(0.02, 0.9, "79 flat objects: no period", transform=ax.transAxes, fontsize=8)
    axes[0, 0].plot([], [], "k--", label="71-d season"); axes[0, 0].plot([], [], "k:", label="35.5 d = R/2")
    axes[0, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    fig.tight_layout()
    p = FIG / "p_true_hist.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_counts(man: pd.DataFrame) -> Path:
    """Star-season counts per class, per season, stacked dev/holdout (the table of record is data/counts.csv)."""
    c = man.groupby(["season", "cls", "eval_set"]).size().unstack("eval_set", fill_value=0)
    c.to_csv(HERE / "data/counts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, s in zip(axes, (1, 2, 3)):
        d = c.loc[s].reindex(ORDER)
        ax.bar(d.index, d["dev"], color="#4c78a8", label="dev")
        ax.bar(d.index, d["holdout"], bottom=d["dev"], color="#f58518", label="holdout")
        for i, (a, b) in enumerate(zip(d["dev"], d["holdout"])):
            ax.text(i, a + b, f"{a}/{b}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(f"season {s}"); ax.tick_params(axis="x", labelrotation=45)
    axes[0].set_ylabel("rows (files × magnitude copies)"); axes[0].legend()
    fig.tight_layout()
    p = FIG / "counts_class_season_split.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def fig_quickcheck() -> Path:
    """Pipeline answer vs the scored period on the quick-check stars (saved in data/quick_check.csv)."""
    q = pd.read_csv(HERE / "data/quick_check.csv")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    x = np.logspace(-1.2, 2.2, 10)
    ax.plot(x, x, "k-", lw=0.8); ax.plot(x, 2 * x, "k:", lw=0.7); ax.plot(x, x / 2, "k:", lw=0.7)
    for cls, g in q.groupby("cls"):
        ok = g.outcome.isin(["correct", "diffrot"])
        ax.scatter(g.P_score, g.P_found.fillna(0.06), s=40, color=COL[cls], marker="o", label=cls)
        ax.scatter(g.P_score[~ok], g.P_found.fillna(0.06)[~ok], s=130, facecolors="none", edgecolors="r")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, 120); ax.set_ylim(0.05, 120)
    ax.set_xlabel("scored period [d] (FL: rotation; EB: P_orb/2)"); ax.set_ylabel("pipeline final period [d]")
    ax.set_title("Quick check, 20 WG6 season-1 stars (red ring = not correct; no answer plotted at 0.06)", fontsize=8.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "quick_check_found_vs_true.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


if __name__ == "__main__":
    m = manifest()
    for f in (fig_period_hist, fig_counts):
        print(f(m))
    print(fig_quickcheck())
    if (len(sys.argv) > 1 and sys.argv[1] == "--drive"):
        print(fig_examples(m))
