"""Candidate recall by method and period bin on dev_medium (coordinator check-in, 2026-11-11).

Reads the existing T10 dev_medium feature tables (no pipeline runs) and answers:
  A. Stage 1: for each of the 7 guess_* methods, the fraction of stars where any of that
     method's raw guesses (P0) is a success (correct or diffrot), per period bin.
  B. Marginal gain: success-in-union of the 3 pooled methods, and how much each of the 4
     non-pooled methods would add to that union, per bin.
  C. Fitted pool: fraction of stars with a relevance >= 1 fitted candidate, credited to each
     contributing pooled method, plus the fraction where that method is the only source.

Usage:
    /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-11-11_candidate-recall-by-method/analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {"full-year": ROOT / "runs/t10_dev_medium_fullyear", "assigned": ROOT / "runs/t10_dev_medium_assigned"}
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
POOL = ["acf_fft", "lombscargle", "pairwise_histogram"]
EXTRA = ["acf_fft_short", "lombscargle_short", "wavelet", "acf_fft_highpass"]
METHODS = POOL + EXTRA
SUCCESS = {"correct", "diffrot"}


def stage1_success(stars: pd.DataFrame) -> pd.DataFrame:
    """Return a star x method boolean frame: True if that method's raw guesses include a success."""
    return pd.DataFrame({m: stars[f"stage1_{m}_outcome"].isin(SUCCESS) for m in METHODS}, index=stars.index)


def table_a(stars: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    """Per-bin, per-method Stage 1 success fraction, plus pool union and all-7 union."""
    df = ok.copy()
    df["pool_union"] = ok[POOL].any(axis=1)
    df["all7_union"] = ok.any(axis=1)
    df["bin"] = stars["eval_pbin"]
    t = df.groupby("bin").mean().reindex(BINS)
    t.loc["all"] = df.drop(columns="bin").mean()
    t["n"] = list(stars["eval_pbin"].value_counts().reindex(BINS)) + [len(stars)]
    return t


def table_b(stars: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    """Per-bin gain in Stage 1 success-union from adding each non-pooled method to the pool."""
    pool = ok[POOL].any(axis=1)
    rows = {}
    for m in EXTRA:
        gain = (~pool) & ok[m]
        rows[f"+{m}"] = gain.groupby(stars["eval_pbin"]).mean().reindex(BINS).tolist() + [gain.mean()]
    rows["+all four"] = ((~pool) & ok[EXTRA].any(axis=1)).groupby(stars["eval_pbin"]).mean().reindex(BINS).tolist() \
        + [((~pool) & ok[EXTRA].any(axis=1)).mean()]
    return pd.DataFrame(rows, index=BINS + ["all"])


def table_c(cands: pd.DataFrame, stars: pd.DataFrame) -> pd.DataFrame:
    """Per-bin fraction of stars with a relevance>=1 fitted candidate, by contributing pooled method."""
    rel = cands[(cands["relevance"] >= 1) & (~cands["fit_error"].astype(bool))]
    out = {}
    per_star_methods = rel.groupby("star_id")["contributing_methods"].apply(
        lambda s: set("+".join(s).split("+")))
    sb = stars.set_index("star_id")["eval_pbin"]
    for m in POOL:
        has = per_star_methods.apply(lambda ms: m in ms)
        only = per_star_methods.apply(lambda ms: ms == {m})
        has = has.reindex(sb.index, fill_value=False)
        only = only.reindex(sb.index, fill_value=False)
        out[f"{m}"] = has.groupby(sb).mean().reindex(BINS).tolist() + [has.mean()]
        out[f"{m} (only source)"] = only.groupby(sb).mean().reindex(BINS).tolist() + [only.mean()]
    anyrel = per_star_methods.reindex(sb.index).notna()
    out["any pooled method"] = anyrel.groupby(sb).mean().reindex(BINS).tolist() + [anyrel.mean()]
    return pd.DataFrame(out, index=BINS + ["all"])


def plot_a(tabs: dict) -> None:
    """Grouped bars: Stage 1 success fraction per method per bin, one panel per coverage mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    colors = plt.cm.tab10(np.arange(len(METHODS)))
    w = 0.11
    x = np.arange(len(BINS))
    for ax, (mode, t) in zip(axes, tabs.items()):
        for i, m in enumerate(METHODS):
            ax.bar(x + (i - 3) * w, t.loc[BINS, m], w, color=colors[i], label=m,
                   hatch=None if m in POOL else "//", edgecolor="white" if m in POOL else "k", lw=0.3)
        ax.plot(x, t.loc[BINS, "pool_union"], "k_", ms=40, mew=2.5, label="pool union (3)")
        ax.plot(x, t.loc[BINS, "all7_union"], "_", color="crimson", ms=40, mew=2.5, label="all-7 union")
        ax.set_xticks(x, BINS)
        ax.set_title(f"dev_medium, {mode}: raw guesses containing the true period")
        ax.set_xlabel("true-period bin")
    axes[0].set_ylabel("fraction of stars (success = correct or diffrot)")
    axes[0].set_ylim(0, 1.05)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=9, frameon=False,
               title="Solid bars: methods pooled into the comb fit today. Hatched: computed but not pooled.",
               title_fontsize=9)
    fig.savefig(OUT / "figures/fig_stage1_by_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_b(tabs: dict) -> None:
    """Bars: gain in Stage 1 union from adding each non-pooled method, per bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cols = list(tabs["full-year"].columns)
    x = np.arange(len(BINS))
    w = 0.16
    for ax, (mode, t) in zip(axes, tabs.items()):
        for i, c in enumerate(cols):
            ax.bar(x + (i - 2) * w, t.loc[BINS, c], w, label=c)
        ax.set_xticks(x, BINS)
        ax.set_title(f"{mode}: stars newly covered if added to the pool")
        ax.set_xlabel("true-period bin")
    axes[0].set_ylabel("added fraction of stars")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/fig_marginal_gain.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build all tables and figures for both coverage modes."""
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    ta, tb, tc = {}, {}, {}
    for mode, d in RUNS.items():
        stars = pd.read_csv(d / "stars.csv.gz")
        stars = stars[stars["is_rotator"]].reset_index(drop=True)
        cands = pd.read_csv(d / "candidates.csv.gz",
                            usecols=["star_id", "relevance", "contributing_methods", "fit_error"])
        ok = stage1_success(stars)
        ta[mode], tb[mode], tc[mode] = table_a(stars, ok), table_b(stars, ok), table_c(cands, stars)
        for name, t in (("stage1", ta[mode]), ("gain", tb[mode]), ("fitted_pool", tc[mode])):
            t.round(3).to_csv(OUT / f"data/{name}_{mode.replace('-', '')}.csv")
        print(f"\n=== {mode} ===\n-- A: stage 1 success by method --\n{ta[mode].round(2).to_string()}")
        print(f"-- B: gain over pool union --\n{tb[mode].round(3).to_string()}")
        print(f"-- C: fitted relevance>=1 by contributing method --\n{tc[mode].round(2).to_string()}")
    plot_a(ta)
    plot_b(tb)


if __name__ == "__main__":
    main()
