#!/usr/bin/env python3
"""
run_full_analysis.py

Run in command line with
python run_full_analysis.py --summary summary.csv --outdir analysis_out/

Standalone script for the guess_* candidate-generation detectability
analysis: given a summary.csv produced by batch_test_guesses.py (one row
per light curve x method, with hit/rank info and spot-model parameters),
this script:

  1. Verifies each method's structural search-range ceiling by checking
     that hit rate drops to exactly zero at the period predicted from
     that method's own max_period default.
  2. Characterizes complete candidate-generation failures (n_candidates
     == 0) and how strongly they're driven by spot ACTIVITY.
  3. Builds a period-controlled subsample (PERIOD <= --period-cutoff,
     default 50d, safely below every method's ceiling) to isolate genuine
     physical detectability drivers from the search-range artifact.
  4. Fits a random forest predicting "found by >= 1 method" from the
     spot-model parameters within that subsample, and reports feature
     importances + cross-validated AUC.
  5. Saves all 6 diagnostic plots and 3 derived CSVs to --outdir.

Usage
-----
    python run_full_analysis.py --summary summary.csv --outdir analysis_out/

Requires: pandas, numpy, matplotlib, scipy, scikit-learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Each method's default max-period search ceiling. Derived from the
# actual defaults in comb_fit.py:
#   guess_acf_fft:            max_period = (acf_lags[-1]-acf_lags[0]) / 2
#   guess_pairwise_histogram: max_lag    = acf_lags[-1]   (no extra /2)
#   guess_lombscargle:        max_period = baseline / 2
#   guess_wavelet:            max_period = baseline / 2
# where acf_lags's own range is truncated by compute_acf's max_lag_frac
# (default 1/3 of baseline), and baseline ~= 357.04d for the full SMARTS
# TESS survey duration. Update BASELINE_DAYS if you're running on a
# reduced-sector subset with a different effective baseline.
# ---------------------------------------------------------------------
BASELINE_DAYS = 357.04
ACF_MAX_LAG_FRAC = 1 / 3
ACF_MAX_LAG = BASELINE_DAYS * ACF_MAX_LAG_FRAC

METHOD_CUTOFFS = {
    "acf_fft": ACF_MAX_LAG / 2,
    "pairwise_histogram": ACF_MAX_LAG,
    "lombscargle": BASELINE_DAYS / 2,
    "wavelet": BASELINE_DAYS / 2,
}
METHOD_COLORS = {
    "acf_fft": "C0", "lombscargle": "C1",
    "pairwise_histogram": "C2", "wavelet": "C3",
}
PERIOD_BAND_EDGES = [0, 1, 10, 20, 50, np.inf]
PERIOD_BAND_LABELS = ["<1d", "1-10d", "10-20d", "20-50d", ">50d"]

PARAM_COLS = ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL",
              "MINLAT", "MAXLAT", "DIFFROT", "TSPOT", "BFLY"]


# ======================================================================
# 1. Structural search-range ceiling verification
# ======================================================================

def verify_structural_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Check that each method's hit rate drops to ~0 above its predicted
    search-range ceiling, confirming the ceiling is a hard structural
    limit rather than a gradual difficulty increase.

    Parameters
    ----------
    df : full per-(star, method) summary dataframe.

    Returns
    -------
    pd.DataFrame indexed by method, with columns for the predicted
    cutoff and the hit rate above/below it.
    """
    rows = []
    for m, cutoff in METHOD_CUTOFFS.items():
        sub = df[df["method"] == m]
        below = sub[sub["true_period"] <= cutoff]
        above = sub[sub["true_period"] > cutoff]
        rows.append(dict(
            method=m, predicted_cutoff_days=cutoff,
            hit_rate_below=below["found_true_period"].mean(),
            n_below=len(below),
            hit_rate_above=above["found_true_period"].mean() if len(above) else np.nan,
            n_above=len(above),
        ))
    return pd.DataFrame(rows).set_index("method")


def plot_hitrate_vs_period_structural(df: pd.DataFrame, bin_width: float = 5.0):
    """Plot 1: hit rate vs. true period per method, with each method's
    structural search-range ceiling overlaid as a dashed vertical line.
    """
    df = df.copy()
    bins = np.arange(0, df["true_period"].max() + bin_width, bin_width)
    df["period_bin"] = pd.cut(df["true_period"], bins)
    tab = df.pivot_table(
        index="period_bin", columns="method", values="found_true_period",
        aggfunc="mean", observed=True,
    )
    centers = [iv.mid for iv in tab.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in tab.columns:
        ax.plot(centers, tab[m], marker="o", ms=3, label=m, color=METHOD_COLORS.get(m))
    for m, cutoff in METHOD_CUTOFFS.items():
        if m in tab.columns:
            ax.axvline(cutoff, color=METHOD_COLORS[m], ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("true period (days)")
    ax.set_ylabel("hit rate (true period in top-10 candidates)")
    ax.set_title("Hit rate vs true period, by method\n"
                  "(dashed lines = predicted search-range ceiling per method)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 2. Zero-candidate (complete detection failure) characterization
# ======================================================================

def zero_candidate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each method returns zero candidates at all
    (as opposed to just missing the true period), and how that failure
    mode correlates with ACTIVITY, DIFFROT, TSPOT, MINLAT, MAXLAT, INCL.

    Parameters
    ----------
    df : full per-(star, method) summary dataframe, must include
        'n_candidates'.

    Returns
    -------
    pd.DataFrame, one row per (method, parameter), with point-biserial
    correlation between "zero candidates" and that parameter.
    """
    from scipy.stats import pointbiserialr

    rows = []
    for m in df["method"].unique():
        s = df[df["method"] == m]
        zero = s["n_candidates"] == 0
        if zero.sum() == 0:
            continue
        for p in ["ACTIVITY", "DIFFROT", "TSPOT", "MINLAT", "MAXLAT", "INCL"]:
            vals = s[p].abs() if p == "DIFFROT" else s[p]
            r, pval = pointbiserialr(zero.astype(int), vals)
            rows.append(dict(
                method=m, parameter=p, point_biserial_r=r, p_value=pval,
                zero_candidate_rate=zero.mean(),
                median_when_zero=vals[zero].median(),
                median_when_nonzero=vals[~zero].median(),
            ))
    return pd.DataFrame(rows)


def plot_zero_candidates_vs_activity(df: pd.DataFrame, star_ids, n_bins: int = 8):
    """Plot 4: rate of complete candidate-generation failure vs. ACTIVITY,
    for the methods that can fail this way (pairwise_histogram, wavelet).
    """
    sub = df[df["star_id"].isin(star_ids)]
    fig, ax = plt.subplots(figsize=(7, 5))
    edges = np.geomspace(max(sub["ACTIVITY"].min(), 0.01), sub["ACTIVITY"].max(), n_bins + 1)
    for m in ["pairwise_histogram", "wavelet"]:
        s = sub[sub["method"] == m].copy()
        if not len(s):
            continue
        s["zero"] = s["n_candidates"] == 0
        b = pd.cut(s["ACTIVITY"], edges)
        rate = s.groupby(b, observed=True)["zero"].mean()
        centers = [iv.mid for iv in rate.index]
        ax.plot(centers, rate.values, marker="o", color=METHOD_COLORS.get(m), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("ACTIVITY (solar-normalized spot activity level)")
    ax.set_ylabel("rate of ZERO candidates generated\n(complete detection failure)")
    ax.set_title("Complete candidate-generation failure vs spot activity level")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ======================================================================
# 3. Period-controlled subsample + intrinsic-difficulty parameter sweeps
# ======================================================================

def build_restricted_subsample(df: pd.DataFrame, period_cutoff: float = 50.0):
    """Build the period-controlled subsample used to isolate genuine
    physical difficulty from the search-range ceiling effect.

    Returns
    -------
    restricted : one row per star with true_period <= period_cutoff.
    merged : same, but with no period restriction (all stars).
    """
    piv = df.pivot_table(
        index="star_id", columns="method", values="found_true_period", aggfunc="first"
    ).astype(bool)
    n_found = piv.sum(axis=1)
    found_by_any = piv.any(axis=1)

    params = df.drop_duplicates("star_id").set_index("star_id")[PARAM_COLS]
    merged = params.join(n_found.rename("n_methods_found")).join(
        found_by_any.rename("found_by_any")
    )
    restricted = merged[merged["PERIOD"] <= period_cutoff].copy()
    return restricted, merged


def spearman_table(restricted: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each numeric spot-model parameter (using
    |DIFFROT| rather than signed DIFFROT) against 'found_by_any', within
    the period-controlled subsample.
    """
    r = restricted.copy()
    r["abs_DIFFROT"] = r["DIFFROT"].abs()
    cols = ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL",
            "MINLAT", "MAXLAT", "abs_DIFFROT", "TSPOT"]
    rows = []
    for c in cols:
        rho, pval = spearmanr(r[c], r["found_by_any"])
        rows.append(dict(parameter=c, spearman_rho=rho, p_value=pval))
    return pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)


def plot_absdiffrot_detectability(restricted: pd.DataFrame, n_bins: int = 7):
    """Plot 3: detection rate vs. |DIFFROT|, on the period-controlled subsample."""
    restricted = restricted.copy()
    restricted["abs_DIFFROT"] = restricted["DIFFROT"].abs()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    edges = np.linspace(0, restricted["abs_DIFFROT"].max(), n_bins + 1)
    b = pd.cut(restricted["abs_DIFFROT"], edges)
    rate = restricted.groupby(b, observed=True)["found_by_any"].mean()
    n = restricted.groupby(b, observed=True)["found_by_any"].count()
    centers = [iv.mid for iv in rate.index]
    ax.plot(centers, rate.values, marker="o", color="firebrick")
    for x, y, ni in zip(centers, rate.values, n.values):
        ax.annotate(f"n={ni}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("|DIFFROT|  (magnitude of differential-rotation shear)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(0, 1)
    ax.set_title("Detectability vs magnitude of differential rotation")
    fig.tight_layout()
    return fig, ax


def plot_param_sweeps(restricted: pd.DataFrame, n_bins: int = 8):
    """Plot 2: detection rate vs. each of 6 spot-model parameters, 2x3 grid."""
    params = ["ACTIVITY", "DIFFROT", "PERIOD", "MINLAT", "TSPOT", "INCL"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, param in zip(axes.flat, params):
        edges = np.linspace(restricted[param].min(), restricted[param].max(), n_bins + 1)
        b = pd.cut(restricted[param], edges)
        rate = restricted.groupby(b, observed=True)["found_by_any"].mean()
        centers = [iv.mid for iv in rate.index]
        ax.plot(centers, rate.values, marker="o", color="steelblue")
        ax.set_xlabel(param)
        ax.set_ylabel("detection rate")
        ax.set_ylim(0, 1)
        ax.set_title(param)
    fig.suptitle(
        "Detectability (found by >=1 method) vs spot-model parameters\n"
        "(period-controlled subsample: all methods have equal search range)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes


def plot_param_sweeps_by_period_band(merged: pd.DataFrame, params=None, n_bins: int = 6):
    """Plot 2b: like plot_param_sweeps, but split into several lines, one
    per true-period band (<1d, 1-10d, 10-20d, 20-50d, >50d), using the
    full (unrestricted) sample since period is now the splitting variable.
    """
    if params is None:
        params = ["ACTIVITY", "DIFFROT", "MINLAT", "MAXLAT", "TSPOT", "INCL"]
    merged = merged.copy()
    merged["period_band"] = pd.cut(
        merged["PERIOD"], bins=PERIOD_BAND_EDGES, labels=PERIOD_BAND_LABELS
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    band_colors = plt.cm.viridis(np.linspace(0, 0.9, len(PERIOD_BAND_LABELS)))

    for ax, param in zip(axes.flat, params):
        edges = np.linspace(merged[param].min(), merged[param].max(), n_bins + 1)
        for band, color in zip(PERIOD_BAND_LABELS, band_colors):
            sub = merged[merged["period_band"] == band]
            if len(sub) < 10:
                continue
            b = pd.cut(sub[param], edges)
            rate = sub.groupby(b, observed=True)["found_by_any"].mean()
            centers = [iv.mid for iv in rate.index]
            ax.plot(centers, rate.values, marker="o", ms=4, color=color, label=band)
        ax.set_xlabel(param)
        ax.set_ylabel("detection rate")
        ax.set_ylim(0, 1)
        ax.set_title(param)
    axes.flat[0].legend(title="true PERIOD", fontsize=8, title_fontsize=8, loc="upper left")
    fig.suptitle(
        "Detectability (found by >=1 method) vs spot-model parameters,\n"
        "split by true rotation period",
        fontsize=13,
    )
    fig.tight_layout()
    return fig, axes


# ======================================================================
# 4. Multivariate detectability model
# ======================================================================

def fit_detectability_model(restricted: pd.DataFrame) -> pd.Series:
    """Fit a random forest predicting 'found_by_any' from the spot-model
    parameters (using |DIFFROT|) and return feature importances, sorted
    descending. Prints 5-fold cross-validated ROC-AUC.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    restricted = restricted.copy()
    restricted["abs_DIFFROT"] = restricted["DIFFROT"].abs()
    feat_cols = ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL",
                 "MINLAT", "MAXLAT", "abs_DIFFROT", "TSPOT", "BFLY"]
    X = restricted[feat_cols].copy()
    X["BFLY"] = X["BFLY"].astype(int)
    y = restricted["found_by_any"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0, n_jobs=-1,
    )
    scores = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)


def plot_feature_importance(importances: pd.Series):
    """Plot 5: horizontal bar chart of random-forest feature importances."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp_sorted = importances.sort_values()
    ax.barh(imp_sorted.index, imp_sorted.values, color="slateblue")
    ax.set_xlabel("random forest feature importance")
    ax.set_title("What predicts detectability, among structurally-reachable LCs?\n"
                  "(predicting found-by-any-method)")
    fig.tight_layout()
    return fig, ax


# ======================================================================
# main
# ======================================================================

def main():
    """Run the full analysis end-to-end and save all outputs to --outdir."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, help="path to summary.csv from batch_test_guesses.py")
    p.add_argument("--outdir", required=True, help="directory to save plots/CSVs to")
    p.add_argument("--period-cutoff", type=float, default=50.0,
                    help="days; period-controlled subsample cutoff (default 50, "
                         "should stay below the smallest method ceiling, ~59.5d)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df["found_true_period"] = df["found_true_period"].astype(bool)
    print(f"Loaded {len(df)} rows, {df['star_id'].nunique()} stars, "
          f"{df['method'].nunique()} methods: {sorted(df['method'].unique())}\n")

    # --- 1. structural cutoffs ---
    cutoffs = verify_structural_cutoffs(df)
    print("=== Structural search-range cutoff verification ===")
    print(cutoffs.to_string())
    cutoffs.to_csv(outdir / "structural_cutoffs.csv")
    fig, _ = plot_hitrate_vs_period_structural(df)
    fig.savefig(outdir / "1_hitrate_vs_period_structural.png", dpi=150)
    plt.close(fig)

    # --- 2. zero-candidate failures ---
    zero_summary = zero_candidate_summary(df)
    print("\n=== Zero-candidate failure correlations ===")
    print(zero_summary.to_string(index=False))
    zero_summary.to_csv(outdir / "zero_candidate_correlations.csv", index=False)

    # --- 3. period-controlled subsample + parameter sweeps ---
    restricted, merged = build_restricted_subsample(df, period_cutoff=args.period_cutoff)
    print(f"\nPeriod-controlled subsample (PERIOD<={args.period_cutoff}d): "
          f"{len(restricted)}/{len(merged)} stars")
    restricted.to_csv(outdir / "restricted_subsample.csv")
    merged.to_csv(outdir / "merged_full.csv")

    corrs = spearman_table(restricted)
    print("\n=== Spearman correlation with found_by_any (period-controlled) ===")
    print(corrs.to_string(index=False))
    corrs.to_csv(outdir / "spearman_correlations.csv", index=False)

    fig, _ = plot_zero_candidates_vs_activity(df, restricted.index)
    fig.savefig(outdir / "4_zero_candidates_vs_activity.png", dpi=150)
    plt.close(fig)

    fig, _ = plot_absdiffrot_detectability(restricted)
    fig.savefig(outdir / "3_absdiffrot.png", dpi=150)
    plt.close(fig)

    fig, _ = plot_param_sweeps(restricted)
    fig.savefig(outdir / "2_param_sweeps_restricted.png", dpi=150)
    plt.close(fig)

    fig, _ = plot_param_sweeps_by_period_band(merged)
    fig.savefig(outdir / "2b_param_sweeps_by_period.png", dpi=150)
    plt.close(fig)

    # --- 4. multivariate model ---
    print("\n=== Random forest detectability model (period-controlled) ===")
    imp = fit_detectability_model(restricted)
    print(imp.to_string())
    imp.to_csv(outdir / "feature_importance.csv")

    fig, _ = plot_feature_importance(imp)
    fig.savefig(outdir / "5_feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"\nAll plots and tables saved to {outdir}/")


if __name__ == "__main__":
    main()
