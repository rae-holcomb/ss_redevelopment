#!/usr/bin/env python3
"""
analysis.py -- T16e: how coverage changes what candidate features mean.

Input: runs/t16e_features/candidates.pkl (build_feature_tables.py), or the slim
copy data/candidates.csv.gz that this script writes (the notebook reads that).

Outputs (data/):
  candidates.csv.gz      slim labelled candidate table (analysis columns only)
  outcome_counts.csv     candidate and final-answer outcome counts per coverage spec
  auc_by_coverage.csv    pooled candidate AUC (success vs not / vs alias / vs wrong)
                         and within-star AUC, per feature x coverage spec
  auc_top_candidate.csv  per-star AUC: does the feature predict whether the
                         arbitration's top-ranked candidate is a success?
  interaction.csv        logistic-regression check (existing vs + coverage
                         vs + interactions), grouped 5-fold CV, with bootstrap CIs
  feature_cost.csv       seconds per candidate for the coverage features and for
                         extract_candidate_features as a whole
figures/: fig_auc_heatmap.png, fig_auc_shift.png, fig_dist_existing.png,
          fig_dist_new.png, fig_top_candidate.png, fig_interaction.png,
          fig_tess_timescale.png

Usage (repo root):
    /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-10-15_t16e-coverage-features/analysis.py
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, FIGS = HERE / "data", HERE / "figures"
RUN_TABLE = ROOT / "runs" / "t16e_features" / "candidates.pkl"

SPECS = ["full", "3-5", "consecutive-2", "gapped-2", "single", "real-weighted"]
SUCCESS = ("correct", "diffrot")
ALIAS = ("half", "double")

NEW_FEATURES = ["n_sectors_observed", "n_contiguous_segments", "longest_contiguous_days",
                "largest_gap_days", "P_over_longest_segment", "n_cycles_observed_total",
                "tess_timescale_rel_dist"]
EXISTING_FEATURES = ["log_period", "log_redchi", "mean_height_snr", "max_height_snr",
                     "frac_positive_heights", "n_peaks_used", "n_teeth_available", "few_peaks",
                     "P_over_lc_baseline", "n_cycles_spanned", "mean_peak_trough_depth_snr",
                     "acf_height_at_P", "acf_half_lag_contrast", "acf_odd_even_contrast",
                     "fold_half_corr", "fold_half_corr_at_2P", "s_ph", "n_agreeing_methods",
                     "n_methods_recognize_as_peak", "redchi_ratio_to_global_best", "passed_gates",
                     "lc_baseline_days", "duty_cycle"]
STAR_LEVEL = {"n_sectors_observed", "n_contiguous_segments", "longest_contiguous_days",
              "largest_gap_days", "lc_baseline_days", "duty_cycle"}
META = ["star_id", "coverage_spec", "pattern", "coverage_class", "true_period", "period_lo",
        "period_hi", "snr", "fitted_P", "outcome", "cand_index", "selected", "pipeline_success",
        "final_outcome", "n_candidates", "feature_seconds", "coverage_feature_seconds"]

# validated default categorical palette (dataviz skill references/palette.md), fixed order
C_SUCCESS, C_ALIAS, C_WRONG = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load(prefer_run: bool = False) -> pd.DataFrame:
    """Labelled candidate table with helper columns.

    Parameters
    ----------
    prefer_run : read the full runs/ pickle (and refresh data/candidates.csv.gz)
        instead of the committed slim copy.

    Returns
    -------
    DataFrame, one row per fitted candidate (stub rows for stars with no
    candidates are dropped), plus success / alias / wrong (bool), star_key
    (spec + star_id) and base (star index shared across specs).
    """
    if prefer_run or not (DATA / "candidates.csv.gz").exists():
        full = pd.read_pickle(RUN_TABLE)
        keep = [c for c in META + EXISTING_FEATURES + NEW_FEATURES if c in full.columns]
        slim = full[list(dict.fromkeys(keep))]
        DATA.mkdir(exist_ok=True)
        slim.to_csv(DATA / "candidates.csv.gz", index=False)
    df = pd.read_csv(DATA / "candidates.csv.gz")
    df = df[df["fitted_P"].notna()].copy()
    df["success"] = df["outcome"].isin(SUCCESS)
    df["alias"] = df["outcome"].isin(ALIAS)
    df["wrong"] = df["outcome"] == "wrong"
    df["star_key"] = df["coverage_spec"] + "|" + df["star_id"]
    df["base"] = df["star_id"]
    for c in ("few_peaks", "passed_gates", "selected", "pipeline_success"):
        df[c] = df[c].astype(float)
    return df


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC-AUC (ties count 1/2); NaN scores dropped; NaN if one class is empty."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    s, y = score[ok], label[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_star_auc(df: pd.DataFrame, feat: str) -> float:
    """Mean over stars (with >= 1 success and >= 1 other candidate) of the per-star AUC."""
    vals = [auc(g[feat].to_numpy(), g["success"].to_numpy()) for _, g in df.groupby("star_key")
            if 0 < g["success"].sum() < len(g)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def outcome_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate outcome fractions and final-answer outcomes per coverage spec."""
    rows = []
    for spec in SPECS:
        d = df[df["coverage_spec"] == spec]
        stars = d.drop_duplicates("star_key")
        fin = stars["final_outcome"].value_counts(normalize=True)
        has_succ = d.groupby("star_key")["success"].any()
        rows.append(dict(
            coverage_spec=spec, n_stars=len(stars), n_candidates=len(d),
            cand_success=d["success"].mean(), cand_alias=d["alias"].mean(), cand_wrong=d["wrong"].mean(),
            stars_with_success_candidate=has_succ.mean(),
            final_success=fin.reindex(list(SUCCESS)).fillna(0).sum(),
            final_alias=fin.reindex(list(ALIAS)).fillna(0).sum(),
            final_wrong=fin.get("wrong", 0.0), final_none=fin.get("none", 0.0),
            single_sector_star_frac=(stars["n_sectors_observed"] == 1).mean(),
        ))
    return pd.DataFrame(rows)


def auc_by_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled and within-star AUCs per feature and coverage spec (long format)."""
    rows = []
    groups = [(s, df[df["coverage_spec"] == s]) for s in SPECS]
    groups.append(("single-sector stars (any spec)", df[df["n_sectors_observed"] == 1]))
    groups.append(("all", df))
    for spec, d in groups:
        for feat in EXISTING_FEATURES + NEW_FEATURES:
            x = d[feat].to_numpy(float)
            sa = d["success"] | d["alias"]
            sw = d["success"] | d["wrong"]
            rows.append(dict(
                coverage_spec=spec, feature=feat, new=feat in NEW_FEATURES,
                star_level=feat in STAR_LEVEL,
                auc_success_vs_rest=auc(x, d["success"]),
                auc_success_vs_alias=auc(x[sa], d["success"][sa]),
                auc_success_vs_wrong=auc(x[sw], d["success"][sw]),
                auc_within_star=np.nan if feat in STAR_LEVEL else within_star_auc(d, feat),
                n_cand=int(np.isfinite(x).sum()), n_success=int(d["success"].sum()),
            ))
    return pd.DataFrame(rows)


def top_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per star: the arbitration's top-ranked candidate (cand_index 0)."""
    top = df[df["cand_index"] == 0].copy()
    top["top_success"] = top["success"]
    return top


def auc_top_candidate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-star AUC: feature of the top candidate vs whether it is a success."""
    top = top_candidate_table(df)
    rows = []
    groups = [(s, top[top["coverage_spec"] == s]) for s in SPECS] + [("all", top)]
    for spec, d in groups:
        for feat in EXISTING_FEATURES + NEW_FEATURES:
            rows.append(dict(coverage_spec=spec, feature=feat, new=feat in NEW_FEATURES,
                             auc_top_success=auc(d[feat], d["top_success"]), n_stars=len(d),
                             frac_top_success=d["top_success"].mean()))
    return pd.DataFrame(rows)


def feature_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate cost of the coverage features and of the full extraction."""
    stars = df.drop_duplicates("star_key")
    out = []
    for spec, d in [*((s, stars[stars["coverage_spec"] == s]) for s in SPECS), ("all", stars)]:
        n = d["n_candidates"].sum()
        out.append(dict(coverage_spec=spec, n_stars=len(d), n_candidates=int(n),
                        coverage_us_per_candidate=1e6 * d["coverage_feature_seconds"].sum() / n,
                        coverage_us_per_star=1e6 * d["coverage_feature_seconds"].mean(),
                        all_features_ms_per_candidate=1e3 * d["feature_seconds"].sum() / n))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# interaction check: numpy logistic regression, grouped CV
# --------------------------------------------------------------------------

LOG_FEATS = {"P_over_lc_baseline", "P_over_longest_segment", "longest_contiguous_days",
             "largest_gap_days", "n_cycles_spanned", "lc_baseline_days", "redchi_ratio_to_global_best"}
MODEL_EXISTING = [f for f in EXISTING_FEATURES if f not in ("lc_baseline_days", "duty_cycle")]
MODEL_COVERAGE = NEW_FEATURES + ["lc_baseline_days", "duty_cycle"]
INTERACT_WITH = ["longest_contiguous_days", "n_cycles_observed_total"]


def _transform(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Model inputs: log10 for scale features, log1p for counts."""
    X = pd.DataFrame(index=df.index)
    for f in feats:
        x = df[f].astype(float)
        if f in LOG_FEATS:
            x = np.log10(x.clip(lower=1e-3))
        elif f == "n_cycles_observed_total":
            x = np.log1p(x.clip(lower=0))
        X[f] = x
    return X


def design(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Feature matrix for model kind 'existing', '+coverage' or '+coverage+interactions'."""
    X = _transform(df, MODEL_EXISTING)
    if kind == "existing":
        return X
    C = _transform(df, MODEL_COVERAGE)
    X = pd.concat([X, C], axis=1)
    if kind == "+coverage+interactions":
        for m in INTERACT_WITH:
            z = (C[m] - C[m].mean()) / (C[m].std() or 1.0)
            for f in MODEL_EXISTING:
                X[f"{f}*{m}"] = X[f] * z
    return X


def _prep(Xtr: np.ndarray, Xte: np.ndarray) -> tuple:
    """Median-impute (plus missing flags), clip at train 1/99 percentiles, standardize."""
    miss_cols = np.flatnonzero(np.isnan(Xtr).any(0) | np.isnan(Xte).any(0))
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    out = []
    lo, hi = np.nanpercentile(Xtr, 1, axis=0), np.nanpercentile(Xtr, 99, axis=0)
    for X in (Xtr, Xte):
        flags = np.isnan(X[:, miss_cols]).astype(float)
        Z = np.where(np.isnan(X), med, X)
        Z = np.clip(Z, np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf))
        out.append(np.hstack([Z, flags]))
    mu, sd = out[0].mean(0), out[0].std(0)
    sd[sd == 0] = 1.0
    return (out[0] - mu) / sd, (out[1] - mu) / sd


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1.0, n_iter: int = 50) -> np.ndarray:
    """L2-regularized logistic regression by Newton/IRLS. Returns [intercept, coefs...]."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    R = lam * np.eye(A.shape[1])
    R[0, 0] = 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + R @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def oof_scores(df: pd.DataFrame, kind: str, n_folds: int = 5, lam: float = 1.0) -> np.ndarray:
    """Out-of-fold logistic scores, folds grouped by base star (same star under every coverage)."""
    X = design(df, kind).to_numpy(float)
    X[~np.isfinite(X)] = np.nan  # e.g. infinite redchi ratios -> treated as missing
    y = df["success"].to_numpy(float)
    fold = df["base"].map(lambda s: zlib.crc32(s.encode()) % n_folds).to_numpy()
    s = np.full(len(df), np.nan)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        Xtr, Xte = _prep(X[tr], X[te])
        w = fit_logistic(Xtr, y[tr], lam)
        s[te] = w[0] + Xte @ w[1:]
    return s


def star_metrics(df: pd.DataFrame, score: np.ndarray) -> dict:
    """Candidate AUC, top-1 success among recoverable stars, and answer-confidence AUC.

    - cand_auc     : success vs not over all candidates;
    - top1_recov   : fraction of stars with >= 1 success candidate whose
                     highest-scoring candidate is a success (ranking);
    - top1_all     : same over all stars (= success rate if never declining);
    - conf_auc     : over all stars, AUC of the top score for "top pick is a
                     success" (how well the score knows when to decline).
    """
    d = df.assign(_s=score)
    idx = d.groupby("star_key")["_s"].idxmax()
    top = d.loc[idx]
    recov = d.groupby("star_key")["success"].any().reindex(top["star_key"]).to_numpy()
    return dict(cand_auc=auc(score, d["success"]),
                top1_recov=float(top["success"].to_numpy()[recov].mean()),
                top1_all=float(top["success"].mean()),
                conf_auc=auc(top["_s"], top["success"]))


def interaction_check(df: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Existing-only vs +coverage vs +coverage+interactions, on all stars and within single-sector stars.

    Two training sets: 'all' (every coverage spec pooled; evaluated on all
    stars and on the single-sector subset) and 'single-sector only' (trained
    and evaluated on n_sectors_observed == 1 stars). Paired bootstrap over base
    stars gives 90% intervals on (model - existing).
    """
    kinds = ["existing", "+coverage", "+coverage+interactions"]
    rng = np.random.default_rng(seed)
    rows = []
    single = df[df["n_sectors_observed"] == 1]
    setups = [("trained on all", df, [("all stars", None), ("single-sector stars", 1)]),
              ("trained on single-sector only", single, [("single-sector stars", 1)])]
    for train_name, d, evals in setups:
        d = d.reset_index(drop=True)
        scores = {k: oof_scores(d, k) for k in kinds}
        for eval_name, nsec in evals:
            m = np.ones(len(d), bool) if nsec is None else (d["n_sectors_observed"] == nsec).to_numpy()
            de = d[m]
            pipeline_top1 = de.drop_duplicates("star_key")["final_outcome"].isin(SUCCESS).mean()
            bases = de["base"].unique()
            point = {k: star_metrics(de, scores[k][m]) for k in kinds}
            boots = {k: [] for k in kinds}
            by_base = {b: np.flatnonzero(de["base"].to_numpy() == b) for b in bases}
            for _ in range(n_boot):
                pick = rng.choice(bases, len(bases))
                ii = np.concatenate([by_base[b] for b in pick])
                sub = de.iloc[ii].copy()
                # make resampled duplicates distinct stars
                sub["star_key"] = sub["star_key"] + "#" + np.repeat(np.arange(len(pick)),
                                                                    [len(by_base[b]) for b in pick]).astype(str)
                for k in kinds:
                    boots[k].append(star_metrics(sub, scores[k][m][ii]))
            for k in kinds:
                for metric in ("cand_auc", "top1_recov", "top1_all", "conf_auc"):
                    diff = np.array([b[metric] - e[metric] for b, e in zip(boots[k], boots["existing"])])
                    rows.append(dict(train=train_name, evaluate=eval_name, model=k, metric=metric,
                                     value=point[k][metric],
                                     delta_vs_existing=point[k][metric] - point["existing"][metric],
                                     delta_lo90=np.nanpercentile(diff, 5), delta_hi90=np.nanpercentile(diff, 95),
                                     n_stars=de["star_key"].nunique(), n_candidates=len(de),
                                     pipeline_final_success=pipeline_top1))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _style():
    """Shared matplotlib style: recessive axes and grid, text in ink colors."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "figure.facecolor": "white", "axes.facecolor": "white", "axes.axisbelow": True})
    return plt


def _diverging_cmap():
    """Orange (AUC < 0.5) - gray - blue (AUC > 0.5)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ob", ["#c24a1c", "#eb6834", "#f0efec", "#2a78d6", "#184f95"])


def fig_auc_heatmap(auc_df: pd.DataFrame, path: Path):
    """Heatmap of pooled success-vs-rest AUC, features x coverage spec."""
    plt = _style()
    cols = SPECS + ["single-sector stars (any spec)"]
    feats = EXISTING_FEATURES + NEW_FEATURES
    piv = auc_df.pivot(index="feature", columns="coverage_spec", values="auc_success_vs_rest").reindex(
        index=feats, columns=cols)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    im = ax.imshow(piv.to_numpy(), cmap=_diverging_cmap(), vmin=0.0, vmax=1.0, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)), [c.replace(" (any spec)", "\n(any spec)") for c in cols], rotation=30,
                  ha="right")
    ax.set_yticks(range(len(feats)), [f + ("  [new]" if f in NEW_FEATURES else "") for f in feats])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v - 0.5) > 0.3 else INK)
    ax.axhline(len(EXISTING_FEATURES) - 0.5, color=INK, lw=1)
    fig.colorbar(im, ax=ax, shrink=0.5, label="AUC, success vs not (candidates)")
    ax.set_title("Univariate AUC by coverage (synthetic n=200, seed 0)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_auc_shift(auc_df: pd.DataFrame, path: Path):
    """Dot plot: |AUC - 0.5| on full vs single coverage, per feature (existing + new)."""
    plt = _style()
    feats = [f for f in EXISTING_FEATURES + NEW_FEATURES]
    a = auc_df.set_index(["coverage_spec", "feature"])["auc_success_vs_rest"]
    full = np.array([a.get(("full", f), np.nan) for f in feats])
    sing = np.array([a.get(("single", f), np.nan) for f in feats])
    gap2 = np.array([a.get(("gapped-2", f), np.nan) for f in feats])
    order = np.argsort(np.nan_to_num(np.abs(full - 0.5), nan=-1))
    fig, ax = plt.subplots(figsize=(7.5, 8))
    y = np.arange(len(feats))
    for yi, i in zip(y, order):
        ax.plot([full[i], sing[i]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(full[order], y, s=36, color=C_SUCCESS, label="full", zorder=3, edgecolor="white", lw=1)
    ax.scatter(sing[order], y, s=36, color=C_ALIAS, label="single", zorder=3, edgecolor="white", lw=1, marker="s")
    ax.scatter(gap2[order], y, s=30, color=C_WRONG, label="gapped-2", zorder=3, edgecolor="white", lw=1,
               marker="D")
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.set_yticks(y, [feats[i] + (" [new]" if feats[i] in NEW_FEATURES else "") for i in order])
    ax.set_xlabel("AUC, success vs not (candidates)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Same feature, different coverage", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_distributions(df: pd.DataFrame, feats: list, path: Path, title: str, logx: set = frozenset()):
    """Rows = features, columns = coverage specs; step histograms (density) by outcome."""
    plt = _style()
    fig, axes = plt.subplots(len(feats), len(SPECS), figsize=(2.3 * len(SPECS), 1.9 * len(feats)),
                             squeeze=False)
    for i, f in enumerate(feats):
        x_all = df[f].astype(float)
        if f in logx:
            x_all = np.log10(x_all + 1) if f.startswith("n_") else np.log10(x_all.clip(lower=1e-3))
        fin = x_all[np.isfinite(x_all)]
        lo, hi = np.nanpercentile(fin, 1), np.nanpercentile(fin, 99)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 30)
        for j, spec in enumerate(SPECS):
            ax = axes[i, j]
            m = (df["coverage_spec"] == spec).to_numpy()
            for lab, col, c in (("success", "success", C_SUCCESS), ("alias", "alias", C_ALIAS),
                                ("wrong", "wrong", C_WRONG)):
                v = x_all[m & df[col].to_numpy()]
                v = v[np.isfinite(v)].clip(lo, hi)
                if len(v):
                    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=c,
                            label=f"{lab} ({len(v)})")
            if i == 0:
                ax.set_title(spec, loc="left")
            if j == 0:
                pre = ("log10 1+" if f.startswith("n_") else "log10 ") if f in logx else ""
                ax.set_ylabel(pre + f, fontsize=8)
            ax.set_yticks([])
            ax.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_top_candidate(df: pd.DataFrame, path: Path):
    """Top-ranked candidate success vs P_over_longest_segment and n_cycles_observed_total, per spec."""
    plt = _style()
    top = top_candidate_table(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins_p = [0, 0.03, 0.1, 0.2, 0.33, 0.5, 1.0, 1e9]
    labels_p = ["<.03", ".03-.1", ".1-.2", ".2-.33", ".33-.5", ".5-1", ">1"]
    bins_n = [-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, 29.5, 1e9]
    labels_n = ["0", "1", "2", "3-4", "5-9", "10-29", "30+"]
    cols = {"full": C_SUCCESS, "single": C_ALIAS, "gapped-2": C_WRONG, "consecutive-2": "#eda100",
            "3-5": "#e87ba4", "real-weighted": "#4a3aa7"}
    for ax, feat, bins, labels in ((axes[0], "P_over_longest_segment", bins_p, labels_p),
                                   (axes[1], "n_cycles_observed_total", bins_n, labels_n)):
        for spec in SPECS:
            d = top[top["coverage_spec"] == spec]
            b = pd.cut(d[feat], bins, labels=labels)
            r = d.groupby(b, observed=False)["top_success"].agg(["mean", "size"])
            r.loc[r["size"] < 5, "mean"] = np.nan
            ax.plot(range(len(labels)), r["mean"], "-o", ms=5, lw=2, color=cols[spec], label=spec)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(f"{feat} of the top-ranked candidate")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction whose top candidate\nis a success (bins with >= 5 stars)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Is the arbitration's top candidate right? By coverage and cycles observed", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tess_timescale(df: pd.DataFrame, path: Path):
    """Candidates by fitted_P near the TESS ladder: outcome mix vs tess_timescale_rel_dist."""
    plt = _style()
    d = df[(df["fitted_P"] > 3) & (df["fitted_P"] < 60)]
    bins = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
    labels = ["<2%", "2-5%", "5-10%", "10-20%", ">20%"]
    b = pd.cut(d["tess_timescale_rel_dist"], bins, labels=labels, include_lowest=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, sub, name in ((axes[0], d, "all candidates, all coverage specs"),
                          (axes[1], d[d["passed_gates"] == 1], "candidates that passed every gate")):
        bb = b.loc[sub.index]
        g = sub.groupby(bb, observed=False)
        frac = pd.DataFrame({"success": g["success"].mean(), "alias": g["alias"].mean(),
                             "wrong": g["wrong"].mean()})
        n = g.size()
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for col, c in (("success", C_SUCCESS), ("alias", C_ALIAS), ("wrong", C_WRONG)):
            ax.bar(x, frac[col].fillna(0), bottom=bottom, color=c, width=0.7, label=col,
                   edgecolor="white", lw=2)
            bottom += frac[col].fillna(0).to_numpy()
        for xi, nn in zip(x, n):
            ax.text(xi, 1.02, f"n={nn}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("tess_timescale_rel_dist (candidates with 3 < fitted_P < 60 d)")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("fraction of candidates")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_interaction(inter: pd.DataFrame, path: Path):
    """Model metrics with 90% bootstrap intervals on the delta vs existing-only."""
    plt = _style()
    panels = [("trained on all", "all stars"), ("trained on all", "single-sector stars"),
              ("trained on single-sector only", "single-sector stars")]
    metrics = [("cand_auc", "candidate AUC"), ("top1_recov", "top-1 success,\nrecoverable stars"),
               ("conf_auc", "answer-confidence AUC")]
    kinds = ["existing", "+coverage", "+coverage+interactions"]
    cols = [INK2, C_SUCCESS, C_ALIAS]
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 3.8), sharey=True)
    for ax, (tr, ev_) in zip(axes, panels):
        d = inter[(inter["train"] == tr) & (inter["evaluate"] == ev_)]
        for mi, (m, mlab) in enumerate(metrics):
            for ki, k in enumerate(kinds):
                r = d[(d["metric"] == m) & (d["model"] == k)].iloc[0]
                x = mi + (ki - 1) * 0.25
                ax.bar(x, r["value"], width=0.22, color=cols[ki], label=k if mi == 0 else None)
                if k != "existing":
                    base = r["value"] - r["delta_vs_existing"]
                    ax.errorbar(x, r["value"], yerr=[[r["delta_vs_existing"] - r["delta_lo90"]],
                                                     [r["delta_hi90"] - r["delta_vs_existing"]]],
                                color=INK, lw=1, capsize=2)
                    del base
        ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"{tr}\nevaluated on {ev_} (n={d['n_stars'].iloc[0]})", loc="left", fontsize=8)
    axes[0].legend(frameon=True, fontsize=7, loc="lower left")
    fig.suptitle("Logistic regression, grouped 5-fold CV (error bars: 90% paired-bootstrap interval of "
                 "the delta vs existing)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_figures(df, auc_df, inter):
    """Write every report figure from the computed tables."""
    FIGS.mkdir(exist_ok=True)
    fig_auc_heatmap(auc_df, FIGS / "fig_auc_heatmap.png")
    fig_auc_shift(auc_df, FIGS / "fig_auc_shift.png")
    fig_distributions(df, ["acf_height_at_P", "acf_half_lag_contrast", "mean_height_snr", "n_teeth_available",
                           "P_over_lc_baseline"], FIGS / "fig_dist_existing.png",
                      "Existing features by outcome (candidates), one column per coverage spec",
                      logx={"P_over_lc_baseline"})
    fig_distributions(df, ["P_over_longest_segment", "n_cycles_observed_total", "tess_timescale_rel_dist"],
                      FIGS / "fig_dist_new.png",
                      "New per-candidate coverage features by outcome, one column per coverage spec",
                      logx={"P_over_longest_segment", "n_cycles_observed_total"})
    fig_top_candidate(df, FIGS / "fig_top_candidate.png")
    fig_tess_timescale(df, FIGS / "fig_tess_timescale.png")
    fig_interaction(inter, FIGS / "fig_interaction.png")


def main():
    """Compute every table, write data/ and figures/."""
    df = load(prefer_run=RUN_TABLE.exists())
    DATA.mkdir(exist_ok=True)
    outcome_counts(df).to_csv(DATA / "outcome_counts.csv", index=False)
    auc_df = auc_by_coverage(df)
    auc_df.to_csv(DATA / "auc_by_coverage.csv", index=False)
    auc_top_candidate(df).to_csv(DATA / "auc_top_candidate.csv", index=False)
    feature_cost(df).to_csv(DATA / "feature_cost.csv", index=False)
    inter = interaction_check(df)
    inter.to_csv(DATA / "interaction.csv", index=False)
    make_figures(df, auc_df, inter)
    print("done", df.shape)


if __name__ == "__main__":
    sys.exit(main())
