#!/usr/bin/env python3
"""T27: ranker v0 on non-rotator negatives, and v0 retrained / recalibrated with negatives.

Models and decline rules (all calibration on training OOF only; dev_medium is scored once):

- ``v0``        T11's saved fold models (``runs/t11_ranker/models/n1000_base_ndcg``), unchanged.
                Its rotator OOF top-1 comes from T11's ``n1000_base_ndcg_oof_top1.csv``; the
                train-pool negatives were never seen by v0, so its scores on them are held-out.
- ``v0+neg_d0`` / ``v0+neg_all``  v0's config (tune.json best ``ndcg``, seed 0, 5-fold CV
                grouped by ``noise_tic``) retrained on train_1k + the train-pool negatives
                (draw 0 only, or all three assigned draws). Negatives have relevance 0 on
                every row.
- ``+head``     (exploratory) a star-level ``binary:logistic`` decline model on the ranker's
                top-1 row (its features + top score + margin), label = "top-1 is a success"
                (0 for every negative), trained on OOF ranker scores with the same folds.

Thresholds per model (``prior`` r = negatives per rotator in the precision; negatives are
re-weighted so the result does not depend on how many draws are in the table):

- ``hand_prec_r0``  v0's rule: match hand arbitration's training precision, rotators only;
- ``hand_prec_r1``  same rule with negatives counted at r = 1 (hand's precision drops too);
- ``prec_0.89_r1``  v0's numeric target (0.89) with negatives at r = 1.

Outputs (``--outdir``, default ``runs/t27_ranker``): ``results.json``, per-model OOF / dev
top-1 CSVs, ``summary.csv`` (one row per model x threshold x dev table).

Usage::

    python tools/train_ranker_negatives.py --neg-train <drive>/extracted/t27_negatives/train \\
        --neg-dev <drive>/extracted/t27_negatives/dev_medium
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import ranking_model as rm  # noqa: E402

WT = os.path.abspath(os.path.join(HERE, ".."))
MAIN = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment"
DEFAULT_TRAIN = "/Volumes/Data_Vault/smarts_data/extracted/t10_tables/train_1k"
DEFAULT_DEV = {"fullyear": f"{MAIN}/runs/t10_dev_medium_fullyear",
               "assigned": f"{MAIN}/runs/t10_dev_medium_assigned"}
DEFAULT_V0 = os.path.join(WT, "runs", "t11_ranker")
V0_KEY = "n1000_base_ndcg"
HEAD_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=3,
                   eta=0.05, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8, seed=0, nthread=4)
HEAD_ROUNDS = 300


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV atomically."""
    tmp = path + ".tmp.csv"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def is_neg(stars: pd.DataFrame) -> np.ndarray:
    """True for non-rotator rows of a star table."""
    return ~stars["is_rotator"].astype(str).str.lower().eq("true").to_numpy()


def top1_any(cands: pd.DataFrame, scores: np.ndarray, stars: pd.DataFrame) -> pd.DataFrame:
    """``ranking_model.top1`` for a table that may contain non-rotators.

    Rotators go through ``rm.top1`` unchanged. For non-rotators the top-1 outcome is
    ``wrong`` (any period) and every star appears (``none`` if it has no candidate rows).
    Adds ``is_rotator``, ``draw_kind`` and ``coverage`` columns.
    """
    neg = is_neg(stars)
    parts = []
    s = pd.Series(scores, index=cands.index)
    if (~neg).any():
        rs = stars[~neg]
        m = cands.star_id.isin(set(rs.star_id)).to_numpy()
        t = rm.top1(cands[m], s[m].to_numpy(), rs)
        t["is_rotator"] = True
        parts.append(t)
    if neg.any():
        ns = stars[neg].copy()
        m = cands.star_id.isin(set(ns.star_id)).to_numpy()
        fake = ns.assign(true_period=1.0, period_lo=np.nan, period_hi=np.nan)
        t = rm.top1(cands[m], s[m].to_numpy(), fake)
        t["true_period"] = np.nan
        t["top_outcome"] = np.where(t.top_period.notna(), ev.WRONG, ev.NONE)
        t["is_rotator"] = False
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    meta = stars.set_index("star_id")
    dk = meta["draw_id"].astype(str).eq("full").map({True: "fullyear", False: "assigned"}) \
        if "draw_id" in meta else pd.Series("rotator", index=meta.index)
    out["draw_kind"] = out.star_id.map(dk).fillna("rotator")
    cov_col = "assigned_coverage_class" if "assigned_coverage_class" in meta else "coverage_class"
    out["coverage"] = out.star_id.map(meta.get(cov_col, meta["coverage_class"]))
    if "assigned_coverage_class" in meta:
        rot_cov = meta["coverage_class"]
        out["coverage"] = out["coverage"].fillna(out.star_id.map(rot_cov))
    out["noise_tic"] = out.star_id.map(meta["noise_tic"])
    return out


def weighted_curve(top: pd.DataFrame, r: float, score_col: str = "top_score") -> pd.DataFrame:
    """Coverage-precision curve with negatives weighted so they count r per rotator.

    precision(thr) = rotator successes / (rotators answered + r * n_rot * FP_rate(thr)).
    """
    t = top.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    rot = t.is_rotator.to_numpy(bool)
    n_rot = int(top.is_rotator.sum())
    n_neg = int((~top.is_rotator.astype(bool)).sum())
    w = np.where(rot, 1.0, (r * n_rot / n_neg) if n_neg else 0.0)
    succ = rm._success(t.top_outcome) & rot
    cw = np.cumsum(w)
    c = pd.DataFrame(dict(threshold=t[score_col].values, precision=np.cumsum(succ * 1.0) / cw,
                          rot_coverage=np.cumsum(rot) / max(n_rot, 1)))
    return c.groupby("threshold", sort=False).tail(1).reset_index(drop=True)


def threshold_for_precision_w(top: pd.DataFrame, target: float, r: float, score_col="top_score") -> float:
    """Lowest threshold whose answered set has weighted precision >= target (+inf if none)."""
    c = weighted_curve(top, r, score_col)
    ok = c[c.precision >= target]
    return float(ok.threshold.min()) if len(ok) else float("inf")


def hand_precision(stars: pd.DataFrame, r: float) -> float:
    """Hand arbitration's precision with negatives weighted r per rotator."""
    neg = is_neg(stars)
    ans = stars.final_outcome.ne(ev.NONE).to_numpy()
    succ = np.isin(stars.final_outcome, ev.SUCCESS_OUTCOMES) & ~neg
    n_rot, n_neg = (~neg).sum(), neg.sum()
    fp = ans[neg].mean() if n_neg else 0.0
    return float(succ.sum() / (ans[~neg].sum() + r * n_rot * fp))


def evaluate(top: pd.DataFrame, thr: float, score_col: str = "top_score") -> dict:
    """Rotator success / wrong / alias / declined and negative FP rate at one threshold."""
    ans = top[score_col].to_numpy(float) >= thr
    out = np.where(ans, top.top_outcome, ev.NONE)
    rot = top.is_rotator.to_numpy(bool)
    o = out[rot]
    res = dict(n_rot=int(rot.sum()), success=float(np.isin(o, ev.SUCCESS_OUTCOMES).mean()),
               alias=float(np.isin(o, ev.ALIAS_OUTCOMES).mean()), wrong=float((o == ev.WRONG).mean()),
               declined=float((o == ev.NONE).mean()))
    neg = ~rot
    res["n_neg"] = int(neg.sum())
    res["neg_fp_rate"] = float(ans[neg].mean()) if neg.any() else np.nan
    a_rot = (o != ev.NONE).sum()
    s_rot = np.isin(o, ev.SUCCESS_OUTCOMES).sum()
    res["precision_r0"] = float(s_rot / a_rot) if a_rot else np.nan
    fp = res["neg_fp_rate"] if neg.any() else 0.0
    res["precision_r1"] = float(s_rot / (a_rot + rot.sum() * fp)) if (a_rot + rot.sum() * fp) else np.nan
    return res


def hand_eval(stars: pd.DataFrame) -> dict:
    """Same metrics for hand arbitration (the pipeline's final answer)."""
    t = stars.assign(top_score=np.where(stars.final_outcome.ne(ev.NONE), 1.0, -1.0),
                     top_outcome=stars.final_outcome, is_rotator=~is_neg(stars))
    return evaluate(t, 0.0)


def fp_breakdown(top: pd.DataFrame, thr: float, score_col: str = "top_score") -> dict:
    """Negative FP rate by draw kind, coverage class and the top-1 period's bin."""
    n = top[~top.is_rotator.astype(bool)].copy()
    n["ans"] = n[score_col].to_numpy(float) >= thr
    n["pbin"] = [ev.period_bin(p) if np.isfinite(p) else "none" for p in n.top_period.astype(float)]
    out = {"by_draw_kind": n.groupby("draw_kind").ans.mean().to_dict(),
           "by_coverage": n.groupby("coverage").ans.mean().to_dict(),
           "answered_by_top_period_bin": (n[n.ans].pbin.value_counts() / len(n)).to_dict()}
    return out


def load_neg(path: str, kinds=("assigned", "fullyear"), draws=None) -> tuple:
    """Negatives table restricted to draw kinds / draw ids; candidates sorted for XGBoost."""
    c, s = rm.load_table(path)
    s["draw_id"] = s.draw_id.astype(str)
    dk = np.where(s.draw_id == "full", "fullyear", "assigned")
    keep = np.isin(dk, kinds)
    if draws is not None:
        keep &= s.draw_id.isin([str(d) for d in draws]).to_numpy() | (dk == "fullyear")
    s = s[keep].reset_index(drop=True)
    c = c[c.star_id.isin(set(s.star_id))].reset_index(drop=True)
    return c, s


def concat_tables(a: tuple, b: tuple) -> tuple:
    """Concatenate two (candidates, stars) tables; candidates re-sorted by (star_id, cand_index)."""
    c = pd.concat([a[0], b[0]], ignore_index=True)
    c = c.sort_values(["star_id", "cand_index"], kind="mergesort").reset_index(drop=True)
    return c, pd.concat([a[1], b[1]], ignore_index=True)


def head_features(top: pd.DataFrame, cands: pd.DataFrame, scores: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Star-level decline features: the top-1 row's model features + top score + margin + n rows."""
    df = pd.DataFrame(dict(star_id=cands.star_id.values, cand_index=cands.cand_index.values, score=scores))
    df["row"] = np.arange(len(df))
    df = df.sort_values(["star_id", "score", "cand_index"], ascending=[True, False, True], kind="mergesort")
    first = df.groupby("star_id").head(1).set_index("star_id")
    n_rows = df.groupby("star_id").size()
    H = X.iloc[first.row.values].reset_index(drop=True)
    H.insert(0, "star_id", first.index.values)
    H["h_top_score"] = first.score.values
    H["h_n_rows"] = n_rows.loc[first.index].values
    H = top[["star_id", "margin"]].merge(H, on="star_id", how="left").rename(columns={"margin": "h_margin"})
    return H


def train_head(Htr: pd.DataFrame, ytr: np.ndarray, rounds: int = HEAD_ROUNDS):
    """One binary:logistic Booster on head features (star_id dropped)."""
    import xgboost as xgb
    X = Htr.drop(columns="star_id").replace([np.inf, -np.inf], [1e30, -1e30]).astype(np.float32)
    return xgb.train(HEAD_PARAMS, xgb.DMatrix(X, label=ytr, missing=np.nan), num_boost_round=rounds)


def predict_head(models, H: pd.DataFrame) -> np.ndarray:
    """Mean head probability over fold models."""
    import xgboost as xgb
    X = H.drop(columns="star_id").replace([np.inf, -np.inf], [1e30, -1e30]).astype(np.float32)
    d = xgb.DMatrix(X, missing=np.nan)
    return np.mean([m.predict(d) for m in models], axis=0)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--neg-train", required=True)
    ap.add_argument("--neg-dev", required=True)
    ap.add_argument("--dev-fullyear", default=DEFAULT_DEV["fullyear"])
    ap.add_argument("--dev-assigned", default=DEFAULT_DEV["assigned"])
    ap.add_argument("--v0-dir", default=DEFAULT_V0)
    ap.add_argument("--outdir", default=os.path.join(WT, "runs", "t27_ranker"))
    ap.add_argument("--n-threads", type=int, default=4)
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    import xgboost as xgb

    mdir = os.path.join(args.v0_dir, "models", V0_KEY)
    cols = json.load(open(os.path.join(mdir, "columns.json")))
    v0_models = []
    for k in range(5):
        b = xgb.Booster()
        b.load_model(os.path.join(mdir, f"fold{k}.json"))
        v0_models.append(b)
    v0_res = json.load(open(os.path.join(args.v0_dir, "results", f"{V0_KEY}.json")))
    best = dict(json.load(open(os.path.join(args.v0_dir, "tune.json")))["best"]["ndcg"])
    n_rounds = best.pop("n_rounds")
    best.pop("oof_top1_success", None)
    params = dict(rm.DEFAULT_PARAMS, objective="rank:ndcg", **best, nthread=args.n_threads, seed=0)

    train = rm.load_table(args.train)
    neg_tr_all = load_neg(args.neg_train, kinds=("assigned",))
    neg_tr_d0 = load_neg(args.neg_train, kinds=("assigned",), draws=[0])
    neg_tr_full = load_neg(args.neg_train, kinds=("fullyear",))
    dev = {"fullyear": concat_tables(rm.load_table(args.dev_fullyear),
                                     load_neg(args.neg_dev, kinds=("fullyear",))),
           "assigned": concat_tables(rm.load_table(args.dev_assigned),
                                     load_neg(args.neg_dev, kinds=("assigned",)))}
    results = dict(config=dict(params=params, n_rounds=n_rounds, train=args.train, neg_train=args.neg_train,
                               neg_dev=args.neg_dev, v0=mdir, columns=len(cols)),
                   n=dict(train_rot=len(train[1]), neg_train_assigned=len(neg_tr_all[1]),
                          neg_train_d0=len(neg_tr_d0[1]), neg_train_fullyear=len(neg_tr_full[1]),
                          **{f"dev_{k}_rot": int((~is_neg(v[1])).sum()) for k, v in dev.items()},
                          **{f"dev_{k}_neg": int(is_neg(v[1]).sum()) for k, v in dev.items()}),
                   hand={}, models={})
    for k, (c, s) in dev.items():
        results["hand"][f"dev_{k}"] = hand_eval(s)
    tr_with_neg = concat_tables(train, neg_tr_all)
    results["hand"]["train_r0"] = hand_precision(train[1], 0.0)
    results["hand"]["train_r1"] = hand_precision(tr_with_neg[1], 1.0)
    results["hand"]["train_neg_fp_rate"] = float(neg_tr_all[1].final_outcome.ne(ev.NONE).mean())
    rows = []

    def score_model(name, models, oof_top, extra_train_negs=None, head=None):
        """Thresholds on OOF (+ held-out train negatives), then dev scoring; fills results/rows."""
        thr = dict(v0_original=float(v0_res["thresholds"]["match_hand_precision"]),
                   hand_prec_r0=threshold_for_precision_w(oof_top[oof_top.is_rotator], results["hand"]["train_r0"], 0.0),
                   hand_prec_r1=threshold_for_precision_w(oof_top, results["hand"]["train_r1"], 1.0),
                   **{"prec_0.89_r1": threshold_for_precision_w(oof_top, 0.89, 1.0)})
        res = dict(thresholds=thr, oof={k: evaluate(oof_top, v) for k, v in thr.items()},
                   oof_fp=fp_breakdown(oof_top, thr["hand_prec_r1"]), dev={})
        _atomic_csv(oof_top, os.path.join(args.outdir, f"{name}_oof_top1.csv"))
        for dname, (dc, ds) in dev.items():
            Xd, _ = rm.feature_matrix(dc, cols)
            sc_ = rm.predict(models, Xd)
            dtop = top1_any(dc, sc_, ds)
            if head is not None:
                Hd = head_features(dtop, dc, sc_, Xd)
                dtop = dtop.merge(pd.DataFrame(dict(star_id=Hd.star_id, p_head=predict_head(head, Hd))),
                                  on="star_id", how="left")
                dtop["top_score"] = dtop["p_head"]
            _atomic_csv(dtop, os.path.join(args.outdir, f"{name}_{dname}_top1.csv"))
            res["dev"][dname] = {}
            for tname, t in thr.items():
                if name.endswith("+head") and tname == "v0_original":
                    continue
                e = evaluate(dtop, t)
                e["fp"] = fp_breakdown(dtop, t)
                res["dev"][dname][tname] = e
                rows.append(dict(model=name, threshold=tname, thr=t, dev=dname,
                                 **{k: v for k, v in e.items() if k != "fp"}))
        results["models"][name] = res
        print(f"done {name}", flush=True)

    # ---- v0, unchanged: rotator OOF from T11 + its (held-out) scores on train negatives
    v0_oof = pd.read_csv(os.path.join(args.v0_dir, "results", f"{V0_KEY}_oof_top1.csv"))
    v0_oof["is_rotator"] = True
    Xn, _ = rm.feature_matrix(neg_tr_all[0], cols)
    v0_neg_top = top1_any(neg_tr_all[0], rm.predict(v0_models, Xn), neg_tr_all[1])
    Xf, _ = rm.feature_matrix(neg_tr_full[0], cols)
    v0_negfull_top = top1_any(neg_tr_full[0], rm.predict(v0_models, Xf), neg_tr_full[1])
    _atomic_csv(pd.concat([v0_neg_top, v0_negfull_top]), os.path.join(args.outdir, "v0_train_neg_top1.csv"))
    results["v0_on_train_negatives"] = {
        "assigned": {k: float((v0_neg_top.top_score >= v).mean())
                     for k, v in v0_res["thresholds"].items()},
        "fullyear": {k: float((v0_negfull_top.top_score >= v).mean())
                     for k, v in v0_res["thresholds"].items()},
        "fp_at_v0_threshold": fp_breakdown(pd.concat([v0_neg_top, v0_negfull_top]),
                                           v0_res["thresholds"]["match_hand_precision"])}
    v0_cal = pd.concat([v0_oof, v0_neg_top], ignore_index=True)
    v0_cal["is_rotator"] = v0_cal["is_rotator"].astype(bool)
    score_model("v0", v0_models, v0_cal)

    # ---- retrained with negatives (same config, grouped by noise_tic)
    fits = {}
    for name, negs in (("v0+neg_d0", neg_tr_d0), ("v0+neg_all", neg_tr_all)):
        c, s = concat_tables(train, negs)
        X, _ = rm.feature_matrix(c, cols)
        y = c.relevance.astype(float).to_numpy()
        w = pd.Series(1.0, index=s.star_id.values)
        cv = rm.cross_validate(c, s, X, y, w, params, n_rounds, n_splits=5, seed=0)
        top = top1_any(c, cv.oof, s)
        score_model(name, cv.models, top)
        fits[name] = (c, s, X, cv, top)

    # ranking unchanged? compare retrained OOF rotator top-1 with v0's
    for name, (c, s, X, cv, top) in fits.items():
        a = top[top.is_rotator].set_index("star_id")
        b = v0_oof.set_index("star_id").loc[a.index]
        results["models"][name]["rotator_top1_same_as_v0"] = float(
            np.mean(np.isclose(a.top_period.astype(float), b.top_period.astype(float), rtol=1e-9, equal_nan=True)))
        results["models"][name]["oof_rotator_top1_success"] = float(rm._success(a.top_outcome).mean())
    results["v0_oof_rotator_top1_success"] = float(rm._success(v0_oof.top_outcome).mean())

    # ---- exploratory decline head on v0+neg_all (OOF ranker scores, same folds)
    c, s, X, cv, top = fits["v0+neg_all"]
    H = head_features(top, c, cv.oof, X)
    lab = pd.Series(rm._success(top.top_outcome) & top.is_rotator.to_numpy(bool), index=top.star_id)
    yH = lab.loc[H.star_id].to_numpy(float)
    fold = H.star_id.map(cv.fold_of_star).to_numpy()
    oofH = np.full(len(H), np.nan)
    heads = []
    for k in range(5):
        m = train_head(H[fold != k], yH[fold != k])
        oofH[fold == k] = predict_head([m], H[fold == k])
        heads.append(m)
    htop = top.merge(pd.DataFrame(dict(star_id=H.star_id, p_head=oofH)), on="star_id", how="left")
    htop["top_score"] = htop.p_head
    score_model("v0+neg_all+head", cv.models, htop, head=heads)

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
