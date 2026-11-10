"""Candidate ranker (T11, ranker v0): XGBoost learning-to-rank over T10 feature tables.

Each star is a ranking query; each candidate row of `candidates.csv.gz` (T10 schema,
`docs/FEATURE_TABLE_SCHEMA.md`) is a document with graded `relevance` (2 correct,
1 diffrot, 0 other). The ranker scores every candidate, reports the top-1 candidate's
period, and declines when the top score is below a threshold calibrated on
out-of-fold (OOF) *training* predictions.

Everything here is deterministic: fixed seeds, `tree_method="hist"`, explicit thread
counts. Splits always group by `noise_tic` (which also groups by star, since each star
has one noise source), so no light curve or noise source sits on both sides of a split.

Outcome scoring goes only through `evaluation.classify_period`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation  # noqa: E402

# Columns that are never model inputs (mirrors tools/build_feature_tables.NON_FEATURE_COLUMNS,
# duplicated so this module does not import the builder and its scorecard dependencies).
NON_FEATURE_COLUMNS = (
    "star_id", "noise_tic", "noise_classification", "noise_periodic", "noise_exclude_from_training",
    "noise_coherent_quiet", "is_rotator", "label_source", "true_period", "period_lo", "period_hi",
    "period_bin", "P_eval", "outcome", "relevance", "alias_class", "selected", "cand_index",
    "contributing_methods", "failed_gates", "fit_error_message", "t0", "fitted_t0",
)
PIPELINE_METHODS = ("acf_fft", "lombscargle", "pairwise_histogram")
HP_WINDOW_DAYS = {"acf_fft_hp2d": 2.0, "acf_fft_hp5d": 5.0, "acf_fft_hp10d": 10.0,
                  "acf_fft_hp20d": 20.0, "acf_fft_hp40d": 40.0}

DEFAULT_PARAMS = dict(objective="rank:ndcg", eval_metric="ndcg@1", tree_method="hist",
                      max_depth=4, eta=0.05, min_child_weight=1.0, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, seed=0, nthread=4,
                      lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8)
DEFAULT_ROUNDS = 300


# --------------------------------------------------------------------------- loading
def load_table(table_dir: str) -> tuple:
    """Read a T10 table directory into (candidates, stars) DataFrames.

    Candidates are sorted by (star_id, cand_index) so every star's rows are contiguous,
    which XGBoost's query grouping requires.
    """
    cands = pd.read_csv(os.path.join(table_dir, "candidates.csv.gz"), low_memory=False)
    stars = pd.read_csv(os.path.join(table_dir, "stars.csv.gz"), low_memory=False)
    cands = cands.sort_values(["star_id", "cand_index"], kind="mergesort").reset_index(drop=True)
    return cands, stars


def strict_quiet_mask(stars: pd.DataFrame) -> pd.Series:
    """True for stars on strict-quiet noise: classified quiet, not excluded, not coherent-quiet."""
    excl = stars["noise_exclude_from_training"].astype(str).str.lower().eq("true")
    coh = stars["noise_coherent_quiet"].astype(str).str.lower().eq("true")
    return stars["noise_classification"].eq("quiet") & ~excl & ~coh


def stratified_prefix(stars: pd.DataFrame, n: int, strat_col: str = "eval_pbin",
                      order_col: str = "train_order") -> list:
    """Star ids of a size-`n` subset with the same `strat_col` shares as the whole table.

    Within each stratum stars are taken in `order_col` order, so a smaller subset is
    always contained in a larger one (nested learning-curve points). `train_order`'s
    plain prefix is not stratified (its first 250 rows are all < 3 d), hence this helper.
    """
    if n >= len(stars):
        return stars["star_id"].tolist()
    shares = stars[strat_col].value_counts(normalize=True)
    quota = (shares * n).round().astype(int)
    diff = n - quota.sum()  # fix rounding so the total is exactly n
    for k in shares.index[: abs(diff)]:
        quota[k] += int(np.sign(diff))
    ids = []
    for k, q in quota.items():
        sub = stars[stars[strat_col] == k].sort_values(order_col, kind="mergesort")
        ids += sub["star_id"].head(q).tolist()
    return ids


# --------------------------------------------------------------------------- features
def _to_float(col: pd.Series) -> pd.Series:
    """Convert a feature column to float: booleans (possibly stored as strings) -> 0/1."""
    if col.dtype == object:
        low = col.astype(str).str.lower()
        mapped = low.map({"true": 1.0, "false": 0.0})
        mapped[col.isna()] = np.nan
        return mapped.astype(float)
    return col.astype(float)


def feature_matrix(cands: pd.DataFrame, columns: Optional[list] = None) -> tuple:
    """Numeric model-input matrix for a candidate table.

    `method` becomes one-hot `method_is_<m>`; `fft_highpass_best_window` becomes its
    window length in days; boolean columns become 0/1 with NaN kept. When `columns` is
    None, columns with at most one distinct non-NaN value are dropped (the constant gate
    thresholds and `*_enabled` flags); pass the training columns to get the same matrix
    for another table.

    Returns (X DataFrame of float32, column list).
    """
    base = [c for c in cands.columns if c not in NON_FEATURE_COLUMNS]
    out = {}
    for c in base:
        if c == "method":
            for m in PIPELINE_METHODS:
                out[f"method_is_{m}"] = (cands[c] == m).astype(float)
        elif c == "fft_highpass_best_window":
            out[c] = cands[c].map(HP_WINDOW_DAYS).astype(float)
        else:
            out[c] = _to_float(cands[c])
    X = pd.DataFrame(out, index=cands.index)
    if columns is None:
        columns = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    for c in columns:
        if c not in X:
            X[c] = np.nan
    # XGBoost rejects inf; keep its ordering information with a large finite sentinel.
    X = X[columns].replace([np.inf, -np.inf], [1e30, -1e30])
    return X.astype(np.float32), list(columns)


# --------------------------------------------------------------------------- labels
LABEL_VARIANTS = ("base", "fiterr0", "wideband_w05", "wideband_rel0")


def labels_and_weights(cands: pd.DataFrame, stars: pd.DataFrame, variant: str = "base") -> tuple:
    """Training labels (per row) and query weights (per star) for an ablation variant.

    Variants:
      base           relevance as built by T10 (unfit rows labelled at P0).
      fiterr0        ablation (a): fit-error rows get relevance 0, so the ranker is not
                     taught to prefer a candidate the pipeline could not fit.
      wideband_w05   ablation (b1): stars in band-width class `ge_0p5` get query weight 0.5
                     (XGBoost ranking weights are per query, not per row).
      wideband_rel0  ablation (b2): relevance 1 rows on `ge_0p5` stars become 0.

    Returns (y ndarray aligned with `cands`, weight Series indexed by star_id).
    """
    if variant not in LABEL_VARIANTS:
        raise ValueError(f"unknown label variant {variant!r}")
    y = cands["relevance"].astype(float).to_numpy().copy()
    w = pd.Series(1.0, index=stars["star_id"].values)
    wide = set(stars.loc[stars["band_width_class"] == "ge_0p5", "star_id"])
    if variant == "fiterr0":
        y[_to_float(cands["fit_error"]).to_numpy() == 1.0] = 0.0
    elif variant == "wideband_w05":
        w[w.index.isin(wide)] = 0.5
    elif variant == "wideband_rel0":
        y[cands["star_id"].isin(wide).to_numpy() & (y == 1.0)] = 0.0
    return y, w


# --------------------------------------------------------------------------- splits / training
def grouped_folds(stars: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Map star_id -> fold index, grouping by noise_tic (stars sharing a noise source stay together)."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = {}
    for k, (_, te) in enumerate(gkf.split(stars, groups=stars["noise_tic"].values)):
        for sid in stars["star_id"].values[te]:
            fold[sid] = k
    return fold


def _dmatrix(X, y, star_ids, qweights=None):
    """DMatrix with query ids from star_ids (rows must be grouped by star)."""
    import xgboost as xgb
    codes, uniq = pd.factorize(pd.Series(star_ids), sort=False)
    d = xgb.DMatrix(X, label=y, qid=codes, missing=np.nan)
    if qweights is not None:
        d.set_weight(np.asarray(qweights.loc[uniq].values, dtype=float))
    return d


def train_ranker(X, y, star_ids, qweights=None, params: Optional[dict] = None,
                 n_rounds: int = DEFAULT_ROUNDS):
    """Train one XGBoost ranking Booster; rows must be contiguous per star."""
    import xgboost as xgb
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    return xgb.train(p, _dmatrix(X, y, star_ids, qweights), num_boost_round=int(n_rounds))


def predict(models, X) -> np.ndarray:
    """Mean score of one or more Boosters (the CV fold ensemble) on matrix X."""
    import xgboost as xgb
    if not isinstance(models, (list, tuple)):
        models = [models]
    d = xgb.DMatrix(X, missing=np.nan)
    return np.mean([m.predict(d) for m in models], axis=0)


@dataclass
class CVResult:
    """Out-of-fold scores (aligned with the training candidates) plus the fold models."""
    oof: np.ndarray
    models: list = field(default_factory=list)
    fold_of_star: dict = field(default_factory=dict)


def cross_validate(cands, stars, X, y, qweights=None, params=None, n_rounds=DEFAULT_ROUNDS,
                   n_splits: int = 5, seed: int = 0) -> CVResult:
    """Grouped K-fold CV: train on K-1 folds, score the held-out fold; returns OOF + models."""
    fold = grouped_folds(stars, n_splits, seed)
    row_fold = cands["star_id"].map(fold).to_numpy()
    oof = np.full(len(cands), np.nan)
    models = []
    for k in range(n_splits):
        tr, te = row_fold != k, row_fold == k
        m = train_ranker(X[tr], y[tr], cands["star_id"].values[tr], qweights, params, n_rounds)
        oof[te] = predict(m, X[te])
        models.append(m)
    return CVResult(oof=oof, models=models, fold_of_star=fold)


# --------------------------------------------------------------------------- top-1 and decline
def top1(cands: pd.DataFrame, scores: np.ndarray, stars: pd.DataFrame) -> pd.DataFrame:
    """Per-star top-scoring candidate, its outcome, and the score margin to the runner-up.

    The reported period is `P_eval` (fitted P if fitted, else P0). Its outcome is
    recomputed with `evaluation.classify_period` against the star's truth. Stars with no
    candidate rows appear with top_score NaN and outcome `none`.
    """
    df = cands[["star_id", "P_eval", "relevance", "fit_error", "beyond_reach", "method",
                "selected", "cand_index"]].copy()
    df["score"] = scores
    df = df.sort_values(["star_id", "score", "cand_index"], ascending=[True, False, True],
                        kind="mergesort")
    df["rank_in_star"] = df.groupby("star_id").cumcount() + 1
    second = df[df.rank_in_star == 2].set_index("star_id")["score"]
    sel_rank = df[df.selected.astype(str).str.lower() == "true"].set_index("star_id")["rank_in_star"]
    top = df[df.rank_in_star == 1].set_index("star_id")
    out = stars[["star_id", "true_period", "period_lo", "period_hi", "period_bin",
                 "final_period", "final_outcome", "max_relevance", "band_width_class",
                 "searchable"]].set_index("star_id").copy()
    out["top_score"] = top["score"]
    out["margin"] = out["top_score"] - second.reindex(out.index).fillna(-np.inf).clip(lower=-1e9)
    out["top_period"] = top["P_eval"]
    out["top_relevance"] = top["relevance"]
    kind = pd.Series(np.where(_to_float(top["fit_error"]) == 1, "fit_error",
                              np.where(_to_float(top["beyond_reach"]) == 1, "beyond_reach", "fitted")),
                     index=top.index)
    out["top_kind"] = kind.reindex(out.index)
    out["top_method"] = top["method"]
    out["hand_choice_rank"] = sel_rank.reindex(out.index)
    out["top_outcome"] = [
        evaluation.classify_period(p, t, evaluation.REL_TOL, (lo, hi))
        for p, t, lo, hi in zip(out.top_period, out.true_period, out.period_lo, out.period_hi)]
    return out.reset_index()


def _success(outcomes) -> np.ndarray:
    """Boolean array: outcome is in evaluation.SUCCESS_OUTCOMES."""
    return np.isin(np.asarray(outcomes), evaluation.SUCCESS_OUTCOMES)


def coverage_accuracy_curve(top: pd.DataFrame) -> pd.DataFrame:
    """Coverage (fraction answered) vs precision (success among answered) as the threshold drops.

    One row per distinct top score, highest first; `threshold` is the smallest top score
    still answered at that point.
    """
    t = top.dropna(subset=["top_score"]).sort_values("top_score", ascending=False)
    s = _success(t.top_outcome).astype(float)
    n_all = len(top)
    k = np.arange(1, len(t) + 1)
    curve = pd.DataFrame(dict(threshold=t.top_score.values, coverage=k / n_all,
                              precision=np.cumsum(s) / k, success_rate=np.cumsum(s) / n_all))
    return curve.groupby("threshold", sort=False).tail(1).reset_index(drop=True)


def threshold_for_precision(top: pd.DataFrame, target: float) -> float:
    """Lowest top-score threshold whose answered set still has precision >= target.

    Uses the precision of the *cumulative* answered set (sorted by score), so the chosen
    point is the largest coverage meeting the target. Returns +inf if no point does.
    """
    c = coverage_accuracy_curve(top)
    ok = c[c.precision >= target]
    return float(ok.threshold.min()) if len(ok) else float("inf")


def threshold_for_coverage(top: pd.DataFrame, coverage: float) -> float:
    """Top-score threshold that answers (about) the given fraction of stars."""
    c = coverage_accuracy_curve(top)
    ok = c[c.coverage <= coverage + 1e-12]
    return float(ok.threshold.min()) if len(ok) else float("inf")


def decide(top: pd.DataFrame, threshold: float) -> pd.Series:
    """Ranker outcome after the decline threshold: top_outcome, or `none` below threshold."""
    ans = top.top_score >= threshold
    return pd.Series(np.where(ans, top.top_outcome, evaluation.NONE), index=top.index)


def outcome_summary(outcomes, bins=None) -> pd.DataFrame:
    """Success / alias / wrong / declined fractions, overall and per period bin.

    `outcomes` are evaluation outcome strings; `bins` (same length) groups them.
    """
    o = pd.Series(np.asarray(outcomes))
    b = pd.Series(np.asarray(bins) if bins is not None else np.full(len(o), "all"))
    rows = []
    for key in ["all"] + [x for x in evaluation.PERIOD_BIN_LABELS if (b == x).any()]:
        sub = o if key == "all" else o[b.values == key]
        n = len(sub)
        rows.append(dict(bin=key, n=n,
                         success=float(np.isin(sub, evaluation.SUCCESS_OUTCOMES).mean()) if n else np.nan,
                         alias=float(np.isin(sub, evaluation.ALIAS_OUTCOMES).mean()) if n else np.nan,
                         wrong=float((sub == evaluation.WRONG).mean()) if n else np.nan,
                         declined=float((sub == evaluation.NONE).mean()) if n else np.nan))
    return pd.DataFrame(rows)


def oracle_outcome(cands: pd.DataFrame, stars: pd.DataFrame) -> pd.Series:
    """Best outcome anywhere in each star's candidate list (`evaluation.best_outcome`)."""
    g = cands.groupby("star_id")["P_eval"].apply(list)
    res = {}
    for _, r in stars.iterrows():
        res[r.star_id] = evaluation.best_outcome(g.get(r.star_id, []), r.true_period,
                                                 evaluation.REL_TOL, (r.period_lo, r.period_hi))[0]
    return pd.Series(res)


# --------------------------------------------------------------------------- diagnostics
SPECTRUM_PREFIX_TO_METHOD = (  # longest prefix first
    ("fft_highpass_", "acf_fft_highpass"), ("fft_short_", "acf_fft_short"), ("fft_", "acf_fft"),
    ("ls_short_", "lombscargle_short"), ("ls_", "lombscargle"), ("wavelet_", "wavelet"),
    ("hist_", "pairwise_histogram"),
)
STAR_LEVEL = {"r_var", "flicker_std", "flux_std", "flux_skewness", "flux_kurtosis", "duty_cycle",
              "von_neumann_eta", "fliper_band_1", "fliper_band_2", "fliper_band_3"}
COVERAGE = {"n_sectors_observed", "n_contiguous_segments", "longest_contiguous_days",
            "largest_gap_days", "P_over_longest_segment", "n_cycles_observed_total",
            "tess_timescale_rel_dist", "P_over_reach", "acf_std", "baseline_days",
            "lc_baseline_days", "n_acf_points", "beyond_reach"}
AGREEMENT = {"n_agreeing_candidates", "n_agreeing_methods", "best_competing_redchi",
             "redchi_ratio_to_best_competing", "redchi_ratio_to_global_best",
             "is_harmonic_of_global_best", "harmonic_factor_of_global_best",
             "is_global_best_redchi", "n_methods_recognize_as_peak", "combined_method_strength"}
ALIAS = {"s_ph", "acf_height_at_P", "acf_half_lag_contrast", "acf_odd_even_contrast",
         "fold_half_corr", "fold_half_corr_at_2P"}
PROVENANCE = {"period", "log_period", "period_frac_of_baseline", "rank", "strength",
              "coverage_at_generation", "n_duplicate_guesses", "n_contributing_methods"}


def feature_family(name: str) -> str:
    """Feature family (FEATURE_DOCUMENTATION sections) used to group importances."""
    if name.startswith("gate_") or name in {"passed_base_gates", "n_failed_gates", "acf_pair_frac_at_P",
                                            "passes_min_peaks", "passes_redchi", "passes_height_snr",
                                            "passed_gates", "fit_error"}:
        return "gates"
    if name.startswith("method_is_"):
        return "candidate_method"
    if feature_method(name) is not None:
        return "method_spectrum"
    for fam, s in (("star_level", STAR_LEVEL), ("coverage_reach", COVERAGE), ("agreement", AGREEMENT),
                   ("alias_sph", ALIAS), ("provenance", PROVENANCE)):
        if name in s:
            return fam
    return "comb_fit"


def feature_method(name: str) -> Optional[str]:
    """Stage 1 `guess_*` method a method-spectrum (or method one-hot) feature belongs to, else None."""
    if name.startswith("method_is_"):
        return name[len("method_is_"):]
    for pre, m in SPECTRUM_PREFIX_TO_METHOD:
        if name.startswith(pre):
            return m
    return None


def shap_contributions(models, X: pd.DataFrame) -> pd.DataFrame:
    """Mean SHAP contribution per row and feature over the fold ensemble (`pred_contribs=True`).

    Returns a DataFrame with the feature columns plus `bias`.
    """
    import xgboost as xgb
    d = xgb.DMatrix(X, missing=np.nan)
    if not isinstance(models, (list, tuple)):
        models = [models]
    c = np.mean([m.predict(d, pred_contribs=True) for m in models], axis=0)
    return pd.DataFrame(c, columns=list(X.columns) + ["bias"], index=X.index)


def gain_importance(models, columns: list) -> pd.Series:
    """Total gain per feature summed over the fold models (0 for unused features)."""
    tot = pd.Series(0.0, index=columns)
    for m in (models if isinstance(models, (list, tuple)) else [models]):
        for k, v in m.get_score(importance_type="total_gain").items():
            tot[k] = tot.get(k, 0.0) + v
    return tot.sort_values(ascending=False)
