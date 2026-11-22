#!/usr/bin/env python3
"""Ranker v1 (T11): rank:ndcg ranker + star-level decline model + per-bin decline thresholds.

Pipeline (everything calibrated on out-of-fold (OOF) training predictions; dev tables are
only scored):

1. **Data.** Rotators: ``train_3k`` (assigned coverage). Negatives: the T27 train-pool
   noise stacks, all three assigned draws per source, minus sources with a persistent weak
   period (``ranking_model.persistent_weak_period``). Their full-year draws are never
   trained on (no full-year rotators in training) but are scored out-of-fold for the FP rate.
2. **Folds.** 5-fold ``GroupKFold`` on ``noise_tic`` over rotators *and* negatives (728 train
   negatives share a noise source with a train rotator), so no noise light curve sits on both
   sides of any split.
3. **Ranker.** ``rank:ndcg`` with T11's tuned config (depth 3, eta 0.1, 150 rounds), trained on
   rotators only (all-zero-relevance queries give NDCG no gradient). Every candidate row,
   including ``beyond_reach`` and fit-error rows, is a document; ``P_over_reach`` is a feature.
4. **Decline model.** ``binary:logistic`` on each star's top-1 row (its features + top score,
   margin, score spread, row count), label = "top-1 is a success" (0 for every negative),
   trained on OOF ranker scores with the same folds. Its probability is ``score_v1``.
5. **Thresholds.** Precision counts negatives at r per rotator (default r = 1, Rae's working
   mix). Rules: ``rank_global`` (v0-style: ranker score, one threshold), ``head_global``,
   ``head_perbin`` (per reported-period bin, same target), ``head_perbin_hand`` (each bin
   matched to hand arbitration's training precision in that bin), and a target grid
   ``head_perbin_p{0.80..0.98}`` for the two-set tuning in the report.

Outputs (``--outdir``): ``models/``, ``oof_top1.csv`` (training stars incl. negatives),
``dev_top1_<set>_<mode>.csv``, ``thresholds.json``, ``summary.csv`` (rule x set x mode x bin),
``curves/``, ``config.json``.

Usage::

    nice -n 15 python tools/train_ranker_v1.py --outdir <main>/runs/ranker_v1/nohalf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import ranking_model as rm  # noqa: E402

MAIN = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment"
TT = f"{MAIN}/runs/p31_l0x_train_tables"
DEV = {("dev_medium", "fullyear"): f"{MAIN}/runs/p31_l0x_dev_medium_fullyear",
       ("dev_medium", "assigned"): f"{MAIN}/runs/p31_l0x_dev_medium_assigned",
       ("dev_medium_b", "fullyear"): f"{MAIN}/runs/p31_l0x_dev_medium_b_fullyear",
       ("dev_medium_b", "assigned"): f"{MAIN}/runs/p31_l0x_dev_medium_b_assigned"}
RANK_PARAMS = dict(objective="rank:ndcg", max_depth=3, eta=0.1, min_child_weight=1.0)
RANK_ROUNDS = 150
PREC_GRID = [round(x, 2) for x in np.arange(0.80, 0.985, 0.01)]
BINS = list(ev.PERIOD_BIN_LABELS)


def load(table_dir: str) -> tuple:
    """(candidates, stars) of a feature table."""
    return rm.load_table(table_dir)


def subset(t: tuple, mask) -> tuple:
    """Restrict a (cands, stars) table to the stars where `mask` (aligned with stars) is True."""
    c, s = t
    s = s[np.asarray(mask, bool)].reset_index(drop=True)
    c = c[c["star_id"].isin(set(s["star_id"]))].reset_index(drop=True)
    return c, s


def concat(*ts) -> tuple:
    """Concatenate (cands, stars) tables; candidates re-sorted so each star's rows are contiguous."""
    c = pd.concat([t[0] for t in ts], ignore_index=True)
    c = c.sort_values(["star_id", "cand_index"], kind="mergesort").reset_index(drop=True)
    return c, pd.concat([t[1] for t in ts], ignore_index=True)


def summarize(top: pd.DataFrame, answered: np.ndarray) -> list:
    """Rows of rotator outcome fractions (overall + per true-period bin) plus the negatives' FP rate."""
    out = rm.decided_outcomes(top, answered)
    rot = top["is_rotator"].to_numpy(bool)
    tab = rm.outcome_summary(out[rot], top.loc[rot, "period_bin"].to_numpy())
    neg = ~rot
    tab["neg_n"] = int(neg.sum())
    tab["neg_fp_rate"] = float(answered[neg].mean()) if neg.any() else np.nan
    return tab.to_dict("records")


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=f"{MAIN}/runs/ranker_v1")
    ap.add_argument("--train", default=f"{TT}/train_3k")
    ap.add_argument("--neg-train", default=f"{TT}/neg_train")
    ap.add_argument("--neg-dev", default=f"{TT}/neg_dev_medium")
    ap.add_argument("--neg-flags", default=f"{TT}/neg_source_flags.csv",
                    help="flags rows (tic, P_ref, tsplit_min_snr) for the negative sources")
    ap.add_argument("--keep-persistent", action="store_true", help="ablation: keep persistent-period sources")
    ap.add_argument("--r", type=float, default=1.0, help="negatives per rotator in the precision")
    ap.add_argument("--n-threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "curves"), exist_ok=True)

    # ---------------------------------------------------------------- data
    rot = load(args.train)
    neg = load(args.neg_train)
    flags = pd.read_csv(args.neg_flags).set_index("tic")
    pers_tics = set(flags.index[rm.persistent_weak_period(flags["P_ref"], flags["tsplit_min_snr"])])
    ns = neg[1]
    full = ns["draw_id"].astype(str).eq("full").to_numpy()
    pers = ns["noise_tic"].isin(pers_tics).to_numpy()
    keep = ~pers | args.keep_persistent
    neg_tr = subset(neg, ~full & keep)                    # trained on
    neg_full = subset(neg, full)                          # FP only (OOF)
    neg_pers = subset(neg, ~full & pers)                  # dropped; scored OOF for the record
    train = concat(rot, neg_tr)
    cols = rm.feature_matrix(rot[0])[1]
    ndraw_src = neg_tr[1]["noise_tic"].nunique()
    info = dict(n_rot=len(rot[1]), n_neg_train=len(neg_tr[1]), n_neg_sources=int(ndraw_src),
                n_persistent_sources=int(neg[1].loc[pers, "noise_tic"].nunique()),
                n_neg_full=len(neg_full[1]), n_features=len(cols),
                neg_per_rot_train=len(neg_tr[1]) / len(rot[1]), r=args.r)
    print(info, flush=True)

    # ---------------------------------------------------------------- folds (by noise_tic over everything)
    all_tics = pd.concat([rot[1]["noise_tic"], neg[1]["noise_tic"]])
    fold_tic = rm.grouped_folds_by_tic(all_tics, 5, args.seed)

    def row_fold(t):
        """Fold of every candidate row of a (cands, stars) table, via its star's noise_tic."""
        return t[0]["star_id"].map(t[1].set_index("star_id")["noise_tic"]).map(fold_tic).to_numpy()

    params = dict(rm.DEFAULT_PARAMS, **RANK_PARAMS, nthread=args.n_threads, seed=args.seed)
    Xr = rm.feature_matrix(rot[0], cols)[0]
    yr = rot[0]["relevance"].astype(float).to_numpy()
    rot_row_fold = row_fold(rot)
    oof_scores = {}
    tabs = {"rot": rot, "neg_tr": neg_tr, "neg_full": neg_full, "neg_pers": neg_pers}
    Xs = {k: rm.feature_matrix(v[0], cols)[0] for k, v in tabs.items()}
    for k in tabs:
        oof_scores[k] = np.full(len(tabs[k][0]), np.nan)
    rankers = []
    for k in range(5):
        tr = rot_row_fold != k
        m = rm.train_ranker(Xr[tr], yr[tr], rot[0]["star_id"].values[tr], None, params, RANK_ROUNDS)
        rankers.append(m)
        m.save_model(os.path.join(args.outdir, "models", f"ranker_fold{k}.json"))
        for name, t in tabs.items():
            rf = row_fold(t)
            te = rf == k
            if te.any():
                oof_scores[name][te] = rm.predict(m, Xs[name][te])
    print(f"ranker CV done {time.perf_counter() - t_start:.0f}s", flush=True)

    # ---------------------------------------------------------------- decline head (OOF ranker scores)
    def tops_and_H(name):
        c, s = tabs[name]
        top = rm.top1_any(c, oof_scores[name], s)
        H = rm.head_features(c, oof_scores[name], Xs[name])
        return top, H

    T = {k: tops_and_H(k) for k in tabs}
    top_tr = pd.concat([T["rot"][0], T["neg_tr"][0]], ignore_index=True)
    H_tr = pd.concat([T["rot"][1], T["neg_tr"][1]], ignore_index=True)
    lab = pd.Series(rm._success(top_tr["top_outcome"]) & top_tr["is_rotator"].to_numpy(bool),
                    index=top_tr["star_id"])
    y_h = lab.loc[H_tr["star_id"]].to_numpy(float)
    tic_of = pd.concat([t[1].set_index("star_id")["noise_tic"] for t in tabs.values()])
    hfold = H_tr["star_id"].map(tic_of).map(fold_tic).to_numpy()
    heads = []
    p_oof = {k: pd.Series(np.nan, index=T[k][1]["star_id"]) for k in tabs}
    for k in range(5):
        m = rm.train_head(H_tr[hfold != k], y_h[hfold != k], params=dict(nthread=args.n_threads, seed=args.seed))
        heads.append(m)
        m.save_model(os.path.join(args.outdir, "models", f"head_fold{k}.json"))
        for name in tabs:
            H = T[name][1]
            f = H["star_id"].map(tic_of).map(fold_tic).to_numpy()
            if (f == k).any():
                p_oof[name][f == k] = rm.predict_head(m, H[f == k])
    for name in tabs:
        T[name][0]["score_v1"] = T[name][0]["star_id"].map(p_oof[name]).to_numpy()
        T[name][0]["set"] = name
    oof = pd.concat([T[k][0] for k in tabs], ignore_index=True)
    oof.to_csv(os.path.join(args.outdir, "oof_top1.csv"), index=False)
    cal = oof[oof["set"].isin(["rot", "neg_tr"])].reset_index(drop=True)
    print(f"head CV done {time.perf_counter() - t_start:.0f}s", flush=True)

    # ---------------------------------------------------------------- thresholds on OOF
    cal_stars = pd.concat([rot[1], neg_tr[1]], ignore_index=True)
    hand_p = rm.hand_precision(cal_stars, args.r)
    hand_pb = rm.hand_precision(cal_stars, args.r, bins=BINS)
    hand_pb = {b: (v if np.isfinite(v) else hand_p) for b, v in hand_pb.items()}
    rules = {
        "rank_global": ("top_score", rm.threshold_for_precision_w(cal, hand_p, args.r, "top_score")),
        "head_global": ("score_v1", rm.threshold_for_precision_w(cal, hand_p, args.r, "score_v1")),
        "head_perbin": ("score_v1", rm.perbin_thresholds(cal, hand_p, args.r, "score_v1")),
        "head_perbin_hand": ("score_v1", rm.perbin_thresholds(cal, hand_pb, args.r, "score_v1")),
        "head_none": ("score_v1", -np.inf),  # no decline: top-1 always answered
    }
    for p in PREC_GRID:
        rules[f"head_perbin_p{p:.2f}"] = ("score_v1", rm.perbin_thresholds(cal, p, args.r, "score_v1"))
        rules[f"head_global_p{p:.2f}"] = ("score_v1", rm.threshold_for_precision_w(cal, p, args.r, "score_v1"))
    thr_json = dict(hand_train_precision=hand_p, hand_train_precision_by_bin=hand_pb,
                    rules={k: v[1] for k, v in rules.items()}, score_cols={k: v[0] for k, v in rules.items()})
    with open(os.path.join(args.outdir, "thresholds.json"), "w") as fh:
        json.dump(thr_json, fh, indent=1, default=float)
    for sc_col in ("score_v1", "top_score"):
        rm.weighted_curve(cal, args.r, sc_col).to_csv(
            os.path.join(args.outdir, "curves", f"oof_{sc_col}.csv"), index=False)

    rows = []

    def record(top, setname, mode, hand_outcomes=None):
        for rule, (col, thr) in rules.items():
            ans = rm.apply_thresholds(top, thr, col)
            for r_ in summarize(top, ans):
                rows.append(dict(rule=rule, set=setname, mode=mode, **r_))
        if hand_outcomes is not None:
            h = top.assign(top_outcome=hand_outcomes)
            ans = h["top_outcome"].ne(ev.NONE).to_numpy()
            for r_ in summarize(h, ans):
                rows.append(dict(rule="hand", set=setname, mode=mode, **r_))

    # OOF (training) summaries: rotators + kept negatives; plus full-year negatives' FP
    cal_hand = cal["star_id"].map(cal_stars.set_index("star_id")["final_outcome"]).to_numpy()
    record(cal, "train_oof", "assigned", cal_hand)
    nf = T["neg_full"][0]
    record(nf, "train_oof_negfull", "fullyear",
           nf["star_id"].map(neg_full[1].set_index("star_id")["final_outcome"]).to_numpy())
    npers = T["neg_pers"][0]
    if len(npers):
        record(npers, "train_oof_negpers", "assigned",
               npers["star_id"].map(neg_pers[1].set_index("star_id")["final_outcome"]).to_numpy())

    # ---------------------------------------------------------------- dev scoring (fold ensembles)
    negdev = load(args.neg_dev)
    nd_full = negdev[1]["draw_id"].astype(str).eq("full").to_numpy()
    nd_pers = negdev[1]["noise_tic"].isin(pers_tics).to_numpy()
    for (setname, mode), path in DEV.items():
        d = load(path)
        if setname == "dev_medium":
            d = concat(d, subset(negdev, nd_full if mode == "fullyear" else ~nd_full))
        X = rm.feature_matrix(d[0], cols)[0]
        sc_ = rm.predict(rankers, X)
        top = rm.top1_any(d[0], sc_, d[1])
        H = rm.head_features(d[0], sc_, X)
        top["score_v1"] = top["star_id"].map(pd.Series(rm.predict_head(heads, H), index=H["star_id"])).to_numpy()
        top["persistent_source"] = top["star_id"].map(
            d[1].set_index("star_id")["noise_tic"]).isin(pers_tics).to_numpy() & ~top["is_rotator"].to_numpy(bool)
        top.to_csv(os.path.join(args.outdir, f"dev_top1_{setname}_{mode}.csv"), index=False)
        hand = top["star_id"].map(d[1].set_index("star_id")["final_outcome"]).to_numpy()
        record(top, setname, mode, hand)
        print(f"scored {setname} {mode} {time.perf_counter() - t_start:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    info["seconds"] = time.perf_counter() - t_start
    info["columns"] = cols
    with open(os.path.join(args.outdir, "config.json"), "w") as fh:
        json.dump(info, fh, indent=1, default=float)
    s = pd.DataFrame(rows)
    show = s[(s["bin"] == "all") & s["rule"].isin(["hand", "rank_global", "head_global", "head_perbin",
                                                  "head_perbin_hand", "head_none"])]
    print(show[["rule", "set", "mode", "n", "success", "wrong", "alias", "declined", "neg_fp_rate"]]
          .to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
