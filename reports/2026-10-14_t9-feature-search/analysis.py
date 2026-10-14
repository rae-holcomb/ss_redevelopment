#!/usr/bin/env python3
"""
analysis.py -- T9 pass 3: univariate separation, redundancy, decline and
arbitration analysis on the labelled candidate table from passes 1-2.

Reads runs/t9_feature_table/candidates.pkl and writes small tables into the
report's data/ folder (the notebook makes every figure from those):

    candidates.csv.gz         the full labelled candidate table (no JSON cols)
    auc_overall.csv           pooled ROC-AUC per feature x contrast
    auc_by_bin.csv            same, per true-period bin
    auc_within_star.csv       within-star pairwise concordance (what a ranker sees)
    redundant_pairs.csv       |Spearman rho| > 0.95 feature pairs
    near_constant.csv         features with <= 1 distinct value or >= 99% one value
    decline_auc.csv           star-level separation for decline prediction
    stars.csv                 one row per star (truth, pipeline outcome, recall flags)
    arbitration_rules.csv     top-1 success of simple single-feature selection rules
    arbitration_gbm.csv       star-grouped CV top-1 success of a pointwise GBM, with
                              and without the new features (indicative only)
    loss_breakdown.csv        issue-12 loss: final outcome x "a success candidate passed gates"
    decline_curve.csv         answered-fraction vs precision for a thresholded single-feature rule
    decline_current.csv       the current pipeline's operating point

Usage (repo root):
    /opt/anaconda3/envs/ss_local/bin/python reports/2026-10-14_t9-feature-search/analysis.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
import evaluation as ev  # noqa: E402

DATA = HERE / "data"
SUCCESS = (ev.CORRECT, ev.DIFFROT)
NEW_FEATURES = ["acf_height_at_P", "acf_half_lag_contrast", "acf_odd_even_contrast", "fold_half_corr",
                "fold_half_corr_at_2P"]
PROTOTYPE_ONLY = ["rvar_p2p_snr", "star_g_acf"]

# Columns that are identifiers, truth, bookkeeping, or pipeline outputs
# that would not exist as model inputs. cand_index is the hand-tuned
# arbitration's own ordering -- kept out of the feature list and used only
# as the "current pipeline" reference.
NON_FEATURES = {
    "star_id", "outcome", "cand_index", "selected", "fitted_t0", "peak_ns_json", "peak_heights_json",
    "true_period", "period_bin", "period_lo", "period_hi", "snr", "shear", "inclination_deg",
    "n_spots", "spot_lifetime_days", "red_noise_sigma", "ramp_amplitude", "pipeline_success",
    "final_outcome", "n_pooled", "star_seconds", "label", "method", "fft_highpass_best_window",
}

CONTRASTS = {
    "succ_vs_alias": (ev.HALF, ev.DOUBLE),
    "succ_vs_half": (ev.HALF,),
    "succ_vs_double": (ev.DOUBLE,),
    "succ_vs_wrong": (ev.WRONG,),
}


def feature_columns(df: pd.DataFrame) -> list:
    """Numeric/bool candidate feature columns (excludes ids, truth, bookkeeping)."""
    cols = []
    for c in df.columns:
        if c in NON_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def auc_table(df: pd.DataFrame, feats: list, min_n: int = 5) -> pd.DataFrame:
    """Pooled ROC-AUC of each feature for success (positive) vs each contrast class.

    AUC > 0.5 means success candidates tend to have HIGHER values. Rows with
    NaN in the feature are dropped (coverage reported). `sep` = |AUC - 0.5|.
    """
    rows = []
    is_succ = df["outcome"].isin(SUCCESS)
    for name, neg in CONTRASTS.items():
        sub = df[is_succ | df["outcome"].isin(neg)]
        y = sub["outcome"].isin(SUCCESS).to_numpy()
        for f in feats:
            x = sub[f].astype(float).to_numpy()
            ok = np.isfinite(x)
            n_pos, n_neg = int(y[ok].sum()), int((~y[ok]).sum())
            auc = np.nan
            if n_pos >= min_n and n_neg >= min_n and np.unique(x[ok]).size > 1:
                auc = roc_auc_score(y[ok], x[ok])
            rows.append(dict(feature=f, contrast=name, auc=auc, sep=abs(auc - 0.5) if np.isfinite(auc) else np.nan,
                             n_pos=n_pos, n_neg=n_neg, coverage=float(ok.mean()) if len(ok) else np.nan))
    return pd.DataFrame(rows)


def within_star_concordance(df: pd.DataFrame, feats: list, neg: tuple) -> pd.DataFrame:
    """Within-star pairwise concordance: over every (success, negative) candidate
    pair from the SAME star, the fraction where the success candidate has the
    higher feature value (ties count 1/2). This is the quantity a pairwise
    ranker optimizes; star-level features are exactly 0.5 by construction.
    """
    rows = []
    groups = [g for _, g in df.groupby("star_id")
              if g["outcome"].isin(SUCCESS).any() and g["outcome"].isin(neg).any()]
    for f in feats:
        num, den = 0.0, 0
        for g in groups:
            xs = g.loc[g["outcome"].isin(SUCCESS), f].astype(float).to_numpy()
            xn = g.loc[g["outcome"].isin(neg), f].astype(float).to_numpy()
            xs, xn = xs[np.isfinite(xs)], xn[np.isfinite(xn)]
            if len(xs) == 0 or len(xn) == 0:
                continue
            d = xs[:, None] - xn[None, :]
            num += float((d > 0).sum() + 0.5 * (d == 0).sum())
            den += d.size
        rows.append(dict(feature=f, concordance=num / den if den else np.nan, n_pairs=den,
                         n_stars=len(groups)))
    out = pd.DataFrame(rows)
    out["sep"] = (out["concordance"] - 0.5).abs()
    return out


def redundancy(df: pd.DataFrame, feats: list, thresh: float = 0.95) -> pd.DataFrame:
    """Feature pairs with |Spearman rho| > thresh (pairwise-complete)."""
    usable = [f for f in feats if df[f].astype(float).nunique() > 1]
    rho = df[usable].astype(float).corr(method="spearman", min_periods=30)
    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            r = rho.loc[a, b]
            if np.isfinite(r) and abs(r) > thresh:
                rows.append(dict(feature_a=a, feature_b=b, spearman=r))
    return pd.DataFrame(rows).sort_values("spearman", key=np.abs, ascending=False)


def near_constant(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Features with <=1 distinct finite value, or one value covering >= 99% of finite rows."""
    rows = []
    for f in feats:
        x = df[f].astype(float)
        fin = x[np.isfinite(x)]
        nun = fin.nunique()
        top = fin.value_counts(normalize=True).iloc[0] if len(fin) else 1.0
        if nun <= 1 or top >= 0.99:
            rows.append(dict(feature=f, n_unique=int(nun), top_value_frac=float(top),
                             finite_frac=float(np.isfinite(x).mean())))
    return pd.DataFrame(rows)


def star_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per star: truth, pipeline outcome, and whether any fitted
    candidate was a success (Stage-2 'recall')."""
    g = df.groupby("star_id", sort=False)
    st = g.agg(true_period=("true_period", "first"), period_bin=("period_bin", "first"),
               snr=("snr", "first"), shear=("shear", "first"), final_outcome=("final_outcome", "first"),
               pipeline_success=("pipeline_success", "first"), n_cands=("outcome", lambda s: s.notna().sum()))
    st["any_success_cand"] = g["outcome"].apply(lambda s: s.isin(SUCCESS).any())
    st["any_alias_cand"] = g["outcome"].apply(lambda s: s.isin((ev.HALF, ev.DOUBLE)).any())
    st["final_success"] = st["final_outcome"].isin(SUCCESS)
    return st.reset_index()


def decline_analysis(df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Star-level AUCs for two decline questions.

    'answer_ok': among stars where the pipeline returned an answer, does the
    feature (taken from the SELECTED candidate's row, which also carries the
    star-level features) separate success from wrong/alias answers?
    'recoverable': over all stars with >= 1 fitted candidate, does the feature
    (from the top-ranked candidate row, cand_index 0) separate stars with at
    least one success candidate from stars with none (= should decline)?
    """
    feats = feature_columns(df)
    rows = []
    sel = df[df["selected"]].set_index("star_id")
    y = st.set_index("star_id").loc[sel.index, "final_success"].to_numpy()
    top = df[df["cand_index"] == 0].set_index("star_id")
    y2 = st.set_index("star_id").loc[top.index, "any_success_cand"].to_numpy()
    for f in feats:
        for q, frame, yy in (("answer_ok", sel, y), ("recoverable", top, y2)):
            x = frame[f].astype(float).to_numpy()
            ok = np.isfinite(x)
            auc = np.nan
            if ok.sum() > 10 and 0 < yy[ok].sum() < ok.sum() and np.unique(x[ok]).size > 1:
                auc = roc_auc_score(yy[ok], x[ok])
            rows.append(dict(question=q, feature=f, auc=auc, sep=abs(auc - 0.5) if np.isfinite(auc) else np.nan,
                             n_pos=int(yy[ok].sum()), n_neg=int((~yy[ok].astype(bool)).sum())))
    return pd.DataFrame(rows)


def arbitration_rules(df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Top-1 success of 'pick the candidate maximizing feature X' rules,
    over stars with >= 1 fitted candidate. Rules restricted to gate-passing
    candidates fall back to all candidates if none pass. Compared with the
    current arbitration (lowest redchi among gate passers) answering on every
    star, i.e. ignoring its decline decision, so the comparison is fair."""
    rules = {
        "current: min redchi among passed (no decline)": ("cand_index", -1, False),
        "min redchi (all)": ("redchi", -1, False),
        "max n_agreeing_methods, then min redchi": ("n_agreeing_methods", +1, False),
        "max combined_method_strength": ("combined_method_strength", +1, False),
        "max mean_peak_trough_depth_snr (passed)": ("mean_peak_trough_depth_snr", +1, True),
        "max mean_height_snr (passed)": ("mean_height_snr", +1, True),
        "max coverage_at_fit (passed)": ("coverage_at_fit", +1, True),
        "max acf_half_lag_contrast (passed)": ("acf_half_lag_contrast", +1, True),
        "max acf_half_lag_contrast (all, ignore gates)": ("acf_half_lag_contrast", +1, False),
        "max mean_peak_trough_depth_snr (all, ignore gates)": ("mean_peak_trough_depth_snr", +1, False),
        "max acf_height_at_P (passed)": ("acf_height_at_P", +1, True),
        "max ls_nearest_peak_frac_power (passed)": ("ls_nearest_peak_frac_power", +1, True),
    }
    rows = []
    stars = [g for _, g in df.groupby("star_id") if g["outcome"].notna().any()]
    for name, (col, sign, passed_only) in rules.items():
        res = []
        for g in stars:
            pool = g[g["passed_gates"]] if passed_only and g["passed_gates"].any() else g
            x = sign * pool[col].astype(float)
            if col == "n_agreeing_methods":  # tie-break on redchi
                x = x - 1e-6 * pool["redchi"].rank()
            if x.notna().sum() == 0:
                res.append((g["period_bin"].iloc[0], False))
                continue
            pick = pool.loc[x.idxmax()]
            res.append((g["period_bin"].iloc[0], pick["outcome"] in SUCCESS))
        r = pd.DataFrame(res, columns=["period_bin", "success"])
        row = dict(rule=name, overall=r["success"].mean(), n=len(r))
        for b in ev.PERIOD_BIN_LABELS:
            row[b] = r.loc[r["period_bin"] == b, "success"].mean()
        rows.append(row)
    oracle = st[st["n_cands"] > 0]
    row = dict(rule="oracle (any success candidate)", overall=oracle["any_success_cand"].mean(), n=len(oracle))
    for b in ev.PERIOD_BIN_LABELS:
        row[b] = oracle.loc[oracle["period_bin"] == b, "any_success_cand"].mean()
    rows.append(row)
    return pd.DataFrame(rows)


def loss_breakdown(df: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """For stars with >= 1 success candidate, cross-tabulate the pipeline's
    final outcome against whether any success candidate passed the gates
    (open issue 12: is the loss in the gates or in the redchi ranking?)."""
    sp = df.groupby("star_id").apply(lambda d: bool(d.loc[d["outcome"].isin(SUCCESS), "passed_gates"].any()))
    x = st[st["any_success_cand"]].join(sp.rename("success_cand_passed_gates"), on="star_id")
    return pd.crosstab(x["final_outcome"], x["success_cand_passed_gates"]).reset_index()


def decline_curve(df: pd.DataFrame, col: str = "acf_half_lag_contrast",
                  thresholds=np.linspace(-0.1, 0.6, 71)) -> pd.DataFrame:
    """Decline trade-off for the rule 'pick max `col` over ALL candidates,
    answer only if that max >= threshold'. Per threshold: fraction of stars
    answered, success rate among answered (precision), and success over all
    stars. Compare with the current pipeline's single operating point."""
    stars = [g for _, g in df.groupby("star_id") if g["outcome"].notna().any()]
    picks = []
    for g in stars:
        x = g[col].astype(float)
        if x.notna().any():
            r = g.loc[x.idxmax()]
            picks.append((float(r[col]), r["outcome"] in SUCCESS))
        else:
            picks.append((-np.inf, False))
    v = np.array([p[0] for p in picks])
    ok = np.array([p[1] for p in picks])
    n = len(stars)
    rows = []
    for t in thresholds:
        a = v >= t
        rows.append(dict(rule=f"max {col}", threshold=t, answered=a.mean(),
                         precision=ok[a].mean() if a.any() else np.nan, success_all=(a & ok).sum() / n))
    return pd.DataFrame(rows)


def arbitration_gbm(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """Indicative only (n=200 stars): star-grouped K-fold, pointwise
    HistGradientBoostingClassifier on success-vs-not, pick the highest-scoring
    candidate per test star; top-1 success with and without the new features.
    Not the T11 ranker -- just a sanity check that the new features add
    information beyond the existing set."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    d = df[df["outcome"].notna()].reset_index(drop=True)
    y = d["outcome"].isin(SUCCESS).to_numpy()
    base = [f for f in feature_columns(d) if f not in NEW_FEATURES + PROTOTYPE_ONLY]
    sets = {"existing": base, "existing+new": base + NEW_FEATURES}
    rows = []
    for name, feats in sets.items():
        X = d[feats].astype(float).to_numpy()
        picks = np.zeros(len(d), dtype=bool)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=d["star_id"]):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                 random_state=seed)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            te_df = d.iloc[te].assign(_p=p)
            for _, g in te_df.groupby("star_id"):
                picks[g["_p"].idxmax()] = True
        chosen = d[picks]
        row = dict(feature_set=name, n_features=len(feats), overall=chosen["outcome"].isin(SUCCESS).mean(),
                   n=len(chosen))
        for b in ev.PERIOD_BIN_LABELS:
            cb = chosen[chosen["period_bin"] == b]
            row[b] = cb["outcome"].isin(SUCCESS).mean() if len(cb) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Run every analysis and write the data/ tables."""
    DATA.mkdir(exist_ok=True)
    df = pd.read_pickle(ROOT / "runs/t9_feature_table/candidates.pkl")
    cand = df[df["outcome"].notna()].copy()
    feats = feature_columns(cand)
    print(f"{cand['star_id'].nunique()} stars with candidates, {len(cand)} candidates, {len(feats)} features")
    print(cand["outcome"].value_counts().to_string())

    df.drop(columns=["peak_ns_json", "peak_heights_json"]).to_csv(DATA / "candidates.csv.gz", index=False)

    overall = auc_table(cand, feats)
    overall.to_csv(DATA / "auc_overall.csv", index=False)
    by_bin = pd.concat([auc_table(g, feats).assign(period_bin=b) for b, g in cand.groupby("period_bin")])
    by_bin.to_csv(DATA / "auc_by_bin.csv", index=False)
    ws = pd.concat([within_star_concordance(cand, feats, neg).assign(contrast=name)
                    for name, neg in CONTRASTS.items()])
    ws.to_csv(DATA / "auc_within_star.csv", index=False)
    redundancy(cand, feats).to_csv(DATA / "redundant_pairs.csv", index=False)
    near_constant(cand, feats).to_csv(DATA / "near_constant.csv", index=False)

    st = star_table(df)
    st.to_csv(DATA / "stars.csv", index=False)
    decline_analysis(cand, st).to_csv(DATA / "decline_auc.csv", index=False)
    arbitration_rules(cand, st).to_csv(DATA / "arbitration_rules.csv", index=False)
    arbitration_gbm(cand).to_csv(DATA / "arbitration_gbm.csv", index=False)
    loss_breakdown(cand, st).to_csv(DATA / "loss_breakdown.csv", index=False)
    decline_curve(cand).to_csv(DATA / "decline_curve.csv", index=False)
    sel = st["final_outcome"] != ev.NONE
    pd.DataFrame([dict(rule="current pipeline", answered=sel.mean(),
                       precision=st.loc[sel, "final_success"].mean(),
                       success_all=st["final_success"].mean())]).to_csv(DATA / "decline_current.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
