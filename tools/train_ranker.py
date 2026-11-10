"""T11 ranker v0: tune, train and evaluate the XGBoost candidate ranker on T10 feature tables.

All model selection (objective, hyperparameters) and the decline threshold use grouped
5-fold CV *within the training table* (groups = noise_tic). The dev_medium tables are
only scored at the end, with the fold-model ensemble and the OOF-calibrated thresholds.

Stages, each cached under --outdir and skipped with --resume:
  tune   grid search over objective x hyperparameters on the training table (--tune-size stars)
  fit    for every (size, label variant, objective) key: CV -> OOF top-1 -> thresholds ->
         dev_medium top-1 (full-year and assigned) -> results/<key>.json + top1 CSVs

Adding the 3,000-star learning-curve point once Rae's run finishes is one command:
  python tools/train_ranker.py --train /Volumes/Data_Vault/smarts_data/extracted/t10_tables/train_3k \
      --sizes 3000 --variants base --outdir runs/t11_ranker --resume
(tuned hyperparameters are reused from runs/t11_ranker/tune.json).
"""
from __future__ import annotations

import argparse
import itertools
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

import evaluation  # noqa: E402
import ranking_model as rm  # noqa: E402

MAIN = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment"
DEFAULT_TRAIN = "/Volumes/Data_Vault/smarts_data/extracted/t10_tables/train_1k"
DEFAULT_DEV = {"fullyear": f"{MAIN}/runs/t10_dev_medium_fullyear",
               "assigned": f"{MAIN}/runs/t10_dev_medium_assigned"}
OBJECTIVES = {"ndcg": "rank:ndcg", "pairwise": "rank:pairwise"}
GRID = dict(max_depth=[3, 5], eta=[0.03, 0.1], n_rounds=[150, 400], min_child_weight=[1.0, 5.0])
SUBSETS = ("all", "strict_quiet", "searchable", "strict_quiet_searchable")


def _atomic_write(path: str, text: str) -> None:
    """Write text to path via a temp file + rename (no partial files if interrupted)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV atomically."""
    tmp = path + ".tmp.csv"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def check_inputs(train_dir: str, dev_dirs: dict) -> None:
    """Fail early with a clear message if the drive is unplugged or a table is missing."""
    if train_dir.startswith("/Volumes/") and not os.path.isdir(os.path.dirname(os.path.dirname(train_dir))):
        sys.exit("Plug in the Data_Vault drive (training table is on it): " + train_dir)
    for d in [train_dir, *dev_dirs.values()]:
        if not os.path.exists(os.path.join(d, "candidates.csv.gz")):
            sys.exit(f"missing feature table: {d}/candidates.csv.gz")


def subset_masks(stars: pd.DataFrame) -> dict:
    """Boolean masks (aligned with stars) for the reporting subsets."""
    sq = rm.strict_quiet_mask(stars).to_numpy()
    se = stars["searchable"].astype(str).str.lower().eq("true").to_numpy()
    return {"all": np.ones(len(stars), bool), "strict_quiet": sq, "searchable": se,
            "strict_quiet_searchable": sq & se}


def restrict(cands, stars, ids):
    """Rows of the candidate and star tables for the given star ids (order preserved)."""
    ids = set(ids)
    return (cands[cands.star_id.isin(ids)].reset_index(drop=True),
            stars[stars.star_id.isin(ids)].reset_index(drop=True))


def hand_point(stars: pd.DataFrame) -> dict:
    """Hand arbitration's coverage and precision (success among answered) on a star table."""
    ans = stars.final_outcome != evaluation.NONE
    succ = np.isin(stars.final_outcome, evaluation.SUCCESS_OUTCOMES)
    return dict(coverage=float(ans.mean()), precision=float(succ[ans].mean()) if ans.any() else np.nan,
                success=float(succ.mean()))


def run_cv(cands, stars, cols, variant, params, n_rounds, n_threads, seed=0):
    """Grouped CV for one configuration; returns (CVResult, X, OOF top-1 table)."""
    X, _ = rm.feature_matrix(cands, cols)
    y, w = rm.labels_and_weights(cands, stars, variant)
    p = dict(params, nthread=n_threads, seed=seed)
    cv = rm.cross_validate(cands, stars, X, y, w, p, n_rounds, n_splits=5, seed=seed)
    return cv, X, rm.top1(cands, cv.oof, stars)


def stage_tune(cands, stars, cols, outdir, n_threads, resume):
    """Grid search over objective x hyperparameters by OOF top-1 success; writes tune.csv/json."""
    path = os.path.join(outdir, "tune.json")
    if resume and os.path.exists(path):
        return json.load(open(path))
    rows = []
    keys = list(GRID)
    for obj, vals in itertools.product(OBJECTIVES, itertools.product(*GRID.values())):
        hp = dict(zip(keys, vals))
        n_rounds = hp.pop("n_rounds")
        params = dict(rm.DEFAULT_PARAMS, objective=OBJECTIVES[obj], **hp)
        t = time.time()
        _, _, top = run_cv(cands, stars, cols, "base", params, n_rounds, n_threads)
        s = rm._success(top.top_outcome)
        thr = rm.threshold_for_precision(top, hand_point(stars)["precision"])
        cov = float((top.top_score >= thr).mean())
        rows.append(dict(objective=obj, n_rounds=n_rounds, **hp, oof_top1_success=float(s.mean()),
                         oof_cov_at_hand_precision=cov, seconds=time.time() - t))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    _atomic_csv(df, os.path.join(outdir, "tune.csv"))
    best = {}
    for obj, g in df.groupby("objective"):
        # primary: OOF top-1 success; tie-break: coverage at hand precision
        b = g.sort_values(["oof_top1_success", "oof_cov_at_hand_precision"], ascending=False).iloc[0]
        best[obj] = dict(n_rounds=int(b.n_rounds), max_depth=int(b.max_depth), eta=float(b.eta),
                         min_child_weight=float(b.min_child_weight),
                         oof_top1_success=float(b.oof_top1_success))
    out = dict(tune_n_stars=int(len(stars)), grid=GRID, best=best)
    _atomic_write(path, json.dumps(out, indent=2))
    return out


def summarize_dev(top: pd.DataFrame, dstars: pd.DataFrame, thresholds: dict) -> dict:
    """Outcome summaries for each threshold x subset on one dev table."""
    masks = subset_masks(dstars.set_index("star_id").loc[top.star_id].reset_index())
    res = {}
    for tname, thr in thresholds.items():
        dec = rm.decide(top, thr).to_numpy()
        for sname, m in masks.items():
            res[f"{tname}|{sname}"] = rm.outcome_summary(dec[m], top.period_bin.values[m]).to_dict("records")
    return res


def stage_fit(key, cands, stars, cols, dev, outdir, variant, obj, best, n_threads, resume,
              save_models=False, train_dir=""):
    """CV-train one key, calibrate thresholds on OOF, score dev tables; writes results/<key>.json."""
    rdir = os.path.join(outdir, "results")
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, f"{key}.json")
    if resume and os.path.exists(path):
        print("skip (resume)", key)
        return
    hp = dict(best[obj])
    n_rounds = hp.pop("n_rounds")
    hp.pop("oof_top1_success", None)
    params = dict(rm.DEFAULT_PARAMS, objective=OBJECTIVES[obj], **hp)
    t = time.time()
    cv, X, top = run_cv(cands, stars, cols, variant, params, n_rounds, n_threads)
    hand = hand_point(stars)
    thresholds = {"no_decline": -np.inf,
                  "match_hand_precision": rm.threshold_for_precision(top, hand["precision"]),
                  "match_hand_coverage": rm.threshold_for_coverage(top, hand["coverage"]),
                  "precision_0.80": rm.threshold_for_precision(top, 0.80)}
    res = dict(key=key, train_dir=train_dir, n_train_stars=int(len(stars)), n_train_rows=int(len(cands)),
               variant=variant, objective=obj, params=params, n_rounds=n_rounds,
               thresholds={k: float(v) for k, v in thresholds.items()}, hand_train=hand,
               oof={f"{t_}|all": rm.outcome_summary(rm.decide(top, v), top.period_bin).to_dict("records")
                    for t_, v in thresholds.items()},
               dev={})
    _atomic_csv(top, os.path.join(rdir, f"{key}_oof_top1.csv"))
    for dname, (dc, ds) in dev.items():
        Xd, _ = rm.feature_matrix(dc, cols)
        dtop = rm.top1(dc, rm.predict(cv.models, Xd), ds)
        _atomic_csv(dtop, os.path.join(rdir, f"{key}_{dname}_top1.csv"))
        res["dev"][dname] = summarize_dev(dtop, ds, thresholds)
    if save_models:
        mdir = os.path.join(outdir, "models", key)
        os.makedirs(mdir, exist_ok=True)
        for k, m in enumerate(cv.models):
            m.save_model(os.path.join(mdir, f"fold{k}.json"))
        _atomic_write(os.path.join(mdir, "columns.json"), json.dumps(cols))
        _atomic_csv(pd.DataFrame({"star_id": cands.star_id, "cand_index": cands.cand_index,
                                  "oof": cv.oof}), os.path.join(mdir, "oof_scores.csv"))
    res["seconds"] = time.time() - t
    _atomic_write(path, json.dumps(res, indent=1, default=float))
    print(f"done {key}: {res['seconds']:.0f}s", flush=True)


def baselines(dev, outdir):
    """Hand arbitration and oracle (best candidate in list) summaries on each dev table."""
    path = os.path.join(outdir, "baselines.json")
    out = {}
    for dname, (dc, ds) in dev.items():
        masks = subset_masks(ds)
        orc = rm.oracle_outcome(dc, ds).reindex(ds.star_id).to_numpy()
        out[dname] = {}
        for sname, m in masks.items():
            out[dname][f"hand|{sname}"] = rm.outcome_summary(ds.final_outcome.values[m], ds.period_bin.values[m]).to_dict("records")
            out[dname][f"oracle|{sname}"] = rm.outcome_summary(orc[m], ds.period_bin.values[m]).to_dict("records")
    _atomic_write(path, json.dumps(out, indent=1, default=float))


def main(argv=None):
    """CLI entry point: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=DEFAULT_TRAIN, help="T10 training table directory")
    ap.add_argument("--dev-fullyear", default=DEFAULT_DEV["fullyear"])
    ap.add_argument("--dev-assigned", default=DEFAULT_DEV["assigned"])
    ap.add_argument("--outdir", default="runs/t11_ranker")
    ap.add_argument("--sizes", default="250,500,1000", help="learning-curve sizes (stars)")
    ap.add_argument("--variants", default="base,fiterr0,wideband_w05,wideband_rel0",
                    help="label variants, run at the largest size only (base runs at every size)")
    ap.add_argument("--objectives", default="ndcg,pairwise", help="objectives compared at the largest size")
    ap.add_argument("--tune-size", type=int, default=0, help="stars used for tuning (0 = all)")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    dev_dirs = {"fullyear": a.dev_fullyear, "assigned": a.dev_assigned}
    check_inputs(a.train, dev_dirs)
    os.makedirs(a.outdir, exist_ok=True)
    cands, stars = rm.load_table(a.train)
    dev = {k: rm.load_table(v) for k, v in dev_dirs.items()}
    _, cols = rm.feature_matrix(cands)
    cpath = os.path.join(a.outdir, "columns.json")
    if os.path.exists(cpath):
        cols = json.load(open(cpath))  # keep one feature list across learning-curve points
    else:
        _atomic_write(cpath, json.dumps(cols))
    if not (a.resume and os.path.exists(os.path.join(a.outdir, "baselines.json"))):
        baselines(dev, a.outdir)

    tc, ts = cands, stars
    if a.tune_size:
        tc, ts = restrict(cands, stars, rm.stratified_prefix(stars, a.tune_size))
    best = stage_tune(tc, ts, cols, a.outdir, a.n_threads, a.resume)["best"]
    print("best", best, flush=True)
    # the objective with the higher tuned OOF success is the headline ranker
    head_obj = max(best, key=lambda o: best[o]["oof_top1_success"])
    sizes = sorted(int(s) for s in a.sizes.split(","))
    largest = max(sizes)
    for n in sizes:
        ids = rm.stratified_prefix(stars, n)
        c, s = restrict(cands, stars, ids)
        n_eff = len(s)
        variants = a.variants.split(",") if n == largest else ["base"]
        objs = a.objectives.split(",") if n == largest else [head_obj]
        for v in variants:
            for o in (objs if v == "base" else [head_obj]):
                stage_fit(f"n{n_eff}_{v}_{o}", c, s, cols, dev, a.outdir, v, o, best, a.n_threads,
                          a.resume, save_models=(v == "base"), train_dir=a.train)
    _atomic_write(os.path.join(a.outdir, "headline.json"), json.dumps({"objective": head_obj}))


if __name__ == "__main__":
    main()
