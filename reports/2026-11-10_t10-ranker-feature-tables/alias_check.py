"""alias_check.py -- T10's required check: do T9's alias features penalize diffrot periods?

For each alias-discrimination feature, compare its distribution on `diffrot`
candidates (fitted period inside the active-latitude band but outside P_eq +/- REL_TOL)
with `correct` candidates (within REL_TOL of P_eq), and place both against the
alias candidates (`half`, `double`).

"Penalize" means: diffrot candidates look shifted toward the alias side of the
feature, i.e. a model that learned "alias-like => bad" from correct-vs-alias
would also push diffrot periods down. We quantify that with

    shift = (median_diffrot - median_correct) / (median_alias - median_correct)

(0 = diffrot looks like correct, 1 = diffrot looks like an alias) and the AUC of
diffrot vs correct (0.5 = indistinguishable), computed overall and per period bin.

Usage (from the repo root):
    python reports/2026-11-10_t10-ranker-feature-tables/alias_check.py \
        --candidates reports/2026-10-14_t9-feature-search/data/candidates.csv.gz --tag t9_synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ALIAS_FEATURES = ["fold_half_corr", "fold_half_corr_at_2P", "acf_half_lag_contrast",
                  "acf_odd_even_contrast", "acf_height_at_P", "half_tooth_ratio",
                  "s_ph", "redchi_ratio_to_global_best"]


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a == b) via the Mann-Whitney U statistic; nan if either is empty."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def summarize(df: pd.DataFrame, features: list, group: str = "all") -> pd.DataFrame:
    """One row per feature: n, medians per outcome class, AUCs and the alias-shift fraction."""
    out = []
    for f in features:
        if f not in df:
            continue
        x = {k: df.loc[df.outcome.isin(v), f].astype(float).to_numpy()
             for k, v in dict(correct=["correct"], diffrot=["diffrot"], alias=["half", "double"]).items()}
        med = {k: np.nanmedian(v) if np.isfinite(v).any() else np.nan for k, v in x.items()}
        den = med["alias"] - med["correct"]
        out.append(dict(group=group, feature=f,
                        n_correct=int(np.isfinite(x["correct"]).sum()),
                        n_diffrot=int(np.isfinite(x["diffrot"]).sum()),
                        n_alias=int(np.isfinite(x["alias"]).sum()),
                        med_correct=med["correct"], med_diffrot=med["diffrot"], med_alias=med["alias"],
                        auc_correct_vs_alias=auc(x["correct"], x["alias"]),
                        auc_diffrot_vs_correct=auc(x["diffrot"], x["correct"]),
                        auc_diffrot_vs_alias=auc(x["diffrot"], x["alias"]),
                        alias_shift=(med["diffrot"] - med["correct"]) / den if abs(den) > 1e-12 else np.nan))
    return pd.DataFrame(out)


def main(argv=None):
    """Run the check on one candidate table and write <tag>_alias_check.csv and a figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin-col", default="period_bin")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.candidates, low_memory=False)
    if "outcome" not in df and "outcome_fit" in df:
        df["outcome"] = df["outcome_fit"]
    feats = [f for f in ALIAS_FEATURES if f in df]
    res = [summarize(df, feats)]
    for b, g in df.groupby(a.bin_col):
        res.append(summarize(g, feats, group=str(b)))
    res = pd.concat(res, ignore_index=True)
    (HERE / "data").mkdir(exist_ok=True)
    res.to_csv(HERE / "data" / f"{a.tag}_alias_check.csv", index=False)
    print(res[res.group == "all"].round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(feats)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.0))
    colors = dict(correct="#1f77b4", diffrot="#2ca02c", alias="#d62728")
    for ax, f in zip(axes.ravel(), feats):
        vals = {k: df.loc[df.outcome.isin(v), f].astype(float).dropna()
                for k, v in dict(correct=["correct"], diffrot=["diffrot"], alias=["half", "double"]).items()}
        allv = np.concatenate([v.to_numpy() for v in vals.values()])
        if len(allv) == 0:
            continue
        lo, hi = np.nanpercentile(allv, [1, 99])
        bins = np.linspace(lo, hi, 30)
        for k, v in vals.items():
            ax.hist(v.clip(lo, hi), bins=bins, density=True, histtype="step", lw=1.6, color=colors[k],
                    label=f"{k} (n={len(v)})")
        r = res[(res.group == "all") & (res.feature == f)].iloc[0]
        ax.set_title(f"{f}\nAUC d-vs-c {r.auc_diffrot_vs_correct:.2f}, shift {r.alias_shift:+.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"Alias features: diffrot vs correct vs alias candidates ({a.tag})", fontsize=10)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"fig_{a.tag}_alias_check.png", dpi=110)


if __name__ == "__main__":
    main()
