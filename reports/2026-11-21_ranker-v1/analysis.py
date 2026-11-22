#!/usr/bin/env python3
"""Ranker v1 report: derive every table the figures need from a `tools/train_ranker_v1.py` run.

Writes small CSV/JSON tables to `data/` (the notebook and figures.py read only those):
- `summary_all.csv`        rule x set x mode x bin outcome fractions (copied from the run);
- `curves.csv`             precision vs answer-rate curves (OOF + dev), negatives at r per rotator;
- `cutoffs.csv`            operating points of the named rules on those curves;
- `tuning.csv`             two-set protocol: per-bin / global precision-target grid, picks;
- `failures.csv`           per dev table: stars hand gets and the ranker (PICK rule) doesn't, by kind;
- `importance_*.csv`       gain + grouped permutation importance (ranker top-1, head AUC) on OOF;
- `ranker_metrics_<set>_<mode>.json`  scorecard-shaped `final` block for check_goals.py.

Usage (repo root)::

    nice -n 15 python reports/2026-11-21_ranker-v1/analysis.py --run <main>/runs/ranker_v1/nohalf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "claude_files"), os.path.join(ROOT, "tools")]
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import ranking_model as rm  # noqa: E402
import train_ranker_v1 as tr  # noqa: E402

DATA = os.path.join(HERE, "data")
PICK = "head_perbin_p0.86"
MATCHED = "head_perbin"
SETS = [("dev_medium", "fullyear"), ("dev_medium", "assigned"), ("dev_medium_b", "fullyear"),
        ("dev_medium_b", "assigned")]
BASELINES = {("dev_medium", "fullyear"): "dev_medium_clean_fullyear_v9",
             ("dev_medium", "assigned"): "dev_medium_clean_assigned_v9",
             ("dev_medium_b", "fullyear"): "dev_medium_b_clean_fullyear_v9",
             ("dev_medium_b", "assigned"): "dev_medium_b_clean_assigned_v9"}


def curves(run: str, thr: dict, r: float) -> tuple:
    """Precision-vs-answer-rate curves (global head-score sweep) + named cutoffs, OOF and dev."""
    rows, cuts = [], []
    tops = {("train_oof", "assigned"): pd.read_csv(os.path.join(run, "oof_top1.csv"))}
    tops[("train_oof", "assigned")] = tops[("train_oof", "assigned")].query("set in ['rot','neg_tr']")
    for s, m in SETS:
        tops[(s, m)] = pd.read_csv(os.path.join(run, f"dev_top1_{s}_{m}.csv"))
    for (s, m), t in tops.items():
        for col in ("score_v1", "top_score"):
            c = rm.weighted_curve(t, r, col)
            # thin to <= 400 points for plotting
            idx = np.unique(np.linspace(0, len(c) - 1, min(len(c), 400)).astype(int))
            rows.append(c.iloc[idx].assign(set=s, mode=m, score=col))
        w = rm.neg_weights(t, r)
        rot = t["is_rotator"].to_numpy(bool)
        succ = rm._success(t["top_outcome"]) & rot

        def point(ans, name):
            cuts.append(dict(set=s, mode=m, rule=name, answer_rate=ans[rot].mean(),
                             success=succ[ans].sum() / rot.sum(),
                             precision=succ[ans].sum() / (w * ans).sum() if (w * ans).sum() else np.nan,
                             fp_rate=ans[~rot].mean() if (~rot).any() else np.nan))
        for name in ("rank_global", "head_global", MATCHED, "head_perbin_hand", PICK, "head_global_p0.88"):
            point(rm.apply_thresholds(t, thr["rules"][name], thr["score_cols"][name]), name)
        point_hand = _hand_point(t, s, m, run)
        if point_hand is not None:
            cuts.append(point_hand)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cuts)


def _hand_point(t, s, m, run):
    """Hand arbitration's point on the same axes, from summary.csv (rotators) + FP rate."""
    summ = pd.read_csv(os.path.join(run, "summary.csv"))
    h = summ[(summ.rule == "hand") & (summ.set == s) & (summ["mode"] == m) & (summ.bin == "all")]
    if not len(h):
        return None
    h = h.iloc[0]
    ans = 1 - h.declined
    fp = h.neg_fp_rate if np.isfinite(h.neg_fp_rate) else 0.0
    r = 1.0 if np.isfinite(h.neg_fp_rate) else 0.0
    return dict(set=s, mode=m, rule="hand", answer_rate=ans, success=h.success,
                precision=h.success / (ans + r * fp), fp_rate=h.neg_fp_rate)


def tuning(summ: pd.DataFrame) -> pd.DataFrame:
    """Two-set protocol over the precision-target grids: pick on one set, judge on the other."""
    a = summ[summ.bin == "all"]
    hand = a[a.rule == "hand"].set_index(["set", "mode"])
    out = []
    for fam in ("head_perbin_p", "head_global_p"):
        g = a[a.rule.str.startswith(fam)].copy()
        g["p"] = g.rule.str[len(fam):].astype(float)
        for tune, judge in (("dev_medium", "dev_medium_b"), ("dev_medium_b", "dev_medium")):
            ok = []
            for p, gp in g.groupby("p"):
                w = gp.set_index(["set", "mode"])["wrong"]
                if all(w[(tune, m)] <= hand.loc[(tune, m), "wrong"] + 1e-12 for m in ("fullyear", "assigned")):
                    ok.append((gp.set_index(["set", "mode"]).loc[[(tune, "fullyear"), (tune, "assigned")],
                                                                 "success"].sum(), p))
            pick = max(ok)[1] if ok else np.nan
            for m in ("fullyear", "assigned"):
                row = g[(g.p == pick) & (g.set == judge) & (g["mode"] == m)]
                h = hand.loc[(judge, m)]
                out.append(dict(family=fam.rstrip("_p"), tuned_on=tune, judged_on=judge, mode=m, pick=pick,
                                success=float(row.success.iloc[0]) if len(row) else np.nan,
                                wrong=float(row.wrong.iloc[0]) if len(row) else np.nan,
                                hand_success=h.success, hand_wrong=h.wrong,
                                passes=bool(len(row) and row.wrong.iloc[0] <= h.wrong + 1e-12)))
    return pd.DataFrame(out)


def failures(run: str, thr: dict) -> pd.DataFrame:
    """Stars hand arbitration gets right and the PICK rule doesn't, classified by why."""
    rows = []
    for s, m in SETS:
        t = pd.read_csv(os.path.join(run, f"dev_top1_{s}_{m}.csv"))
        t = t[t.is_rotator.astype(bool)].copy()
        ans = rm.apply_thresholds(t, thr["rules"][PICK], thr["score_cols"][PICK])
        out = rm.decided_outcomes(t, ans)
        hand_ok = rm._success(t["final_outcome"])
        rk_ok = rm._success(out)
        top_ok = rm._success(t["top_outcome"])
        kind = np.select([~ans & top_ok, ~ans & ~top_ok, ans & np.isin(out, ev.ALIAS_OUTCOMES)],
                         ["declined, top-1 right", "declined, top-1 wrong", "answered alias"],
                         "answered wrong")
        for lose, label in ((hand_ok & ~rk_ok, "ranker loses"), (~hand_ok & rk_ok, "ranker wins")):
            sub = t[lose].assign(kind=np.where(label == "ranker loses", kind[lose], "win"), side=label,
                                 set=s, mode=m)
            rows.append(sub[["star_id", "set", "mode", "side", "kind", "period_bin", "true_period",
                             "top_period", "final_period", "top_bin", "score_v1", "top_score",
                             "hand_choice_rank", "top_kind", "top_method", "top_outcome", "final_outcome"]])
    return pd.concat(rows, ignore_index=True)


def ranker_metrics(run: str, thr: dict, rule: str) -> None:
    """Scorecard-shaped metrics.json (final.overall / by_bin / by_coverage_class) for check_goals."""
    for s, m in SETS:
        t = pd.read_csv(os.path.join(run, f"dev_top1_{s}_{m}.csv"))
        t = t[t.is_rotator.astype(bool)].reset_index(drop=True)
        out = rm.decided_outcomes(t, rm.apply_thresholds(t, thr["rules"][rule], thr["score_cols"][rule]))
        base = json.load(open(os.path.join(ROOT, "benchmarks", "baselines", BASELINES[(s, m)] + ".json")))
        cov = t["star_id"].map(pd.read_csv(os.path.join(tr.DEV[(s, m)], "stars.csv.gz"))
                               .set_index("star_id")["coverage_class"])
        final = dict(overall=ev.outcome_fractions(out),
                     by_bin={b: ev.outcome_fractions(out[t.period_bin == b]) for b in ev.PERIOD_BIN_LABELS},
                     by_coverage_class={c: ev.outcome_fractions(out[cov == c]) for c in sorted(cov.dropna().unique())})
        import subprocess
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        doc = dict(final=final, config=dict(base["config"], git_commit=commit, system=f"ranker v1 ({rule})", run=run))
        with open(os.path.join(DATA, f"ranker_metrics_{s}_{m}.json"), "w") as fh:
            json.dump(doc, fh, indent=1, default=float)


def importance(run: str, n_rep: int = 3) -> None:
    """Gain + grouped (feature-family) permutation importance on OOF, ranker and head, same folds."""
    import xgboost as xgb
    cfg = json.load(open(os.path.join(run, "config.json")))
    cols = cfg["columns"]
    rot = tr.load(f"{tr.TT}/train_3k")
    neg = tr.load(f"{tr.TT}/neg_train")
    flags = pd.read_csv(f"{tr.TT}/neg_source_flags.csv").set_index("tic")
    pers = set(flags.index[rm.persistent_weak_period(flags.P_ref, flags.tsplit_min_snr)])
    ns = neg[1]
    neg_tr = tr.subset(neg, ~ns.draw_id.astype(str).eq("full").to_numpy() & ~ns.noise_tic.isin(pers).to_numpy())
    fold_tic = rm.grouped_folds_by_tic(pd.concat([rot[1].noise_tic, neg[1].noise_tic]), 5, 0)
    rankers = [xgb.Booster(model_file=os.path.join(run, "models", f"ranker_fold{k}.json")) for k in range(5)]
    heads = [xgb.Booster(model_file=os.path.join(run, "models", f"head_fold{k}.json")) for k in range(5)]
    data = tr.concat(rot, neg_tr)
    c, s = data
    X = rm.feature_matrix(c, cols)[0]
    tic = c.star_id.map(s.set_index("star_id").noise_tic)
    rf = tic.map(fold_tic).to_numpy()
    fams = pd.Series({f: rm.feature_family(f) for f in cols})
    groups = {f"family:{k}": list(v.index) for k, v in fams.groupby(fams)}
    for meth in ("wavelet", "lombscargle", "lombscargle_short", "acf_fft", "acf_fft_short",
                 "acf_fft_highpass", "pairwise_histogram"):
        g = [f for f in cols if rm.feature_method(f) == meth]
        if g:
            groups[f"method:{meth}"] = g
    for f in ("P_over_reach", "beyond_reach"):
        if f in cols:
            groups[f"feature:{f}"] = [f]

    def evaluate(Xp):
        sc_ = np.full(len(c), np.nan)
        for k in range(5):
            sc_[rf == k] = rm.predict(rankers[k], Xp[rf == k])
        top = rm.top1_any(c, sc_, s)
        H = rm.head_features(c, sc_, Xp)
        hf = H.star_id.map(s.set_index("star_id").noise_tic).map(fold_tic).to_numpy()
        p = np.full(len(H), np.nan)
        for k in range(5):
            p[hf == k] = rm.predict_head(heads[k], H[hf == k])
        tt = top.set_index("star_id").loc[H.star_id]
        y = rm._success(tt.top_outcome) & tt.is_rotator.to_numpy(bool)
        from sklearn.metrics import roc_auc_score
        rot_m = tt.is_rotator.to_numpy(bool)
        return float(y[rot_m].mean()), float(roc_auc_score(y, p))

    base_top1, base_auc = evaluate(X)
    rng = np.random.default_rng(0)
    rows = []
    for name, g in groups.items():
        d1, d2 = [], []
        for _ in range(n_rep):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))  # rows; the family's columns move together
            Xp[g] = X[g].to_numpy()[idx]
            a, b = evaluate(Xp)
            d1.append(base_top1 - a)
            d2.append(base_auc - b)
        rows.append(dict(group=name, n_features=len(g), drop_top1_success=np.mean(d1),
                         drop_head_auc=np.mean(d2)))
        print(name, rows[-1], flush=True)
    pd.DataFrame(rows).assign(base_top1=base_top1, base_auc=base_auc).sort_values(
        "drop_head_auc", ascending=False).to_csv(os.path.join(DATA, "importance_permutation.csv"), index=False)
    gr = rm.gain_importance(rankers, cols).rename("ranker_gain")
    hcols = cols + ["h_top_score", "h_margin", "h_score_std", "h_n_rows"]
    gh = rm.gain_importance(heads, hcols).rename("head_gain")
    g = pd.concat([gr, gh], axis=1).fillna(0.0)
    g["family"] = [rm.feature_family(f) if not f.startswith("h_") else "ranker_score" for f in g.index]
    g.rename_axis("feature").reset_index().to_csv(os.path.join(DATA, "importance_gain.csv"), index=False)


def _table_tasks(table) -> list:
    """Scorecard load tasks for a feature table's stars, rebuilt from its config.json."""
    import build_feature_tables as bft
    cfg = json.load(open(table / "config.json"))
    ns = argparse.Namespace(manifest=os.path.join(ROOT, cfg["manifest"]), data_root=cfg["data_root"],
                            ignore_manifest_sectors=cfg["ignore_manifest_sectors"], only=cfg.get("only"),
                            limit=None, nonrotator=cfg.get("nonrotator", False),
                            only_set=cfg.get("only_set"), flags=None, coherent_list=None)
    if ns.data_root and not os.path.isabs(ns.data_root):
        ns.data_root = os.path.join(ROOT, ns.data_root)
    return bft.build_tasks(ns)


def examples(run: str, n_per_kind: int = 2) -> None:
    """Per-candidate scores + light curves for representative full-year losses (dev_medium)."""
    import xgboost as xgb
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import scorecard as sc
    cfg = json.load(open(os.path.join(run, "config.json")))
    f = pd.read_csv(os.path.join(DATA, "failures.csv"))
    f = f[(f.side == "ranker loses") & (f["mode"] == "fullyear")]
    pick = []
    for kind in ["declined, top-1 right", "answered alias", "answered wrong", "declined, top-1 wrong"]:
        sub = f[f.kind == kind].sort_values(["set", "star_id"])
        pick += [(r.set, r.star_id, kind) for r in sub.drop_duplicates("period_bin").head(n_per_kind if kind.startswith("declined, top-1 right") else 1).itertuples()]
    pick = pick[:5]
    rankers = [xgb.Booster(model_file=os.path.join(run, "models", f"ranker_fold{k}.json")) for k in range(5)]
    rows, lcs = [], {}
    from pathlib import Path
    for s, sid, kind in pick:
        tab = tr.DEV[(s, "fullyear")]
        c, st = tr.load(tab)
        c = c[c.star_id == sid].reset_index(drop=True)
        c["score"] = rm.predict(rankers, rm.feature_matrix(c, cfg["columns"])[0])
        rows.append(c[["star_id", "cand_index", "P_eval", "relevance", "selected", "beyond_reach", "fit_error",
                       "score", "method"]].assign(set=s, kind=kind,
                                                  true_period=float(st.set_index("star_id").loc[sid, "true_period"])))
        task = [t for t in _table_tasks(Path(tab)) if t["star_id"] == sid][0]
        _, t_, f_, *_ = sc._load(task)
        lcs[sid] = (np.asarray(t_), np.asarray(f_))
    pd.concat(rows).to_csv(os.path.join(DATA, "examples_candidates.csv"), index=False)
    np.savez_compressed(os.path.join(DATA, "examples_lc.npz"), **{f"{k}__t": v[0] for k, v in lcs.items()},
                        **{f"{k}__f": v[1] for k, v in lcs.items()})


def ablation(base: str, alt: str) -> None:
    """Ablation tables: whole-set and per-bin summaries, two-set tuning and OOF curves for two runs."""
    rows, tun, cvs = [], [], []
    for name, run in (("base", base), ("alt", alt)):
        cfg = json.load(open(os.path.join(run, "config.json")))
        summ = pd.read_csv(os.path.join(run, "summary.csv"))
        summ.insert(0, "run", name)
        summ["half"] = cfg.get("half", False)
        rows.append(summ)
        t = tuning(summ[summ.columns[1:-1]])
        t.insert(0, "run", name)
        t["half"] = cfg.get("half", False)
        tun.append(t)
        cv = pd.read_csv(os.path.join(run, "curves", "oof_score_v1.csv"))
        cv.insert(0, "run", name)
        cv["half"] = cfg.get("half", False)
        cvs.append(cv)
    pd.concat(rows).to_csv(os.path.join(DATA, "ablation_summary.csv"), index=False)
    pd.concat(tun).to_csv(os.path.join(DATA, "ablation_tuning.csv"), index=False)
    pd.concat(cvs).to_csv(os.path.join(DATA, "ablation_curves.csv"), index=False)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--ablation", default=None,
                    help="second run dir (e.g. runs/ranker_v1/half): write data/ablation_* comparing it to --run")
    args = ap.parse_args(argv)
    os.makedirs(DATA, exist_ok=True)
    if args.ablation:
        ablation(args.run, args.ablation)
        return
    thr = json.load(open(os.path.join(args.run, "thresholds.json")))
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    summ = pd.read_csv(os.path.join(args.run, "summary.csv"))
    summ.to_csv(os.path.join(DATA, "summary_all.csv"), index=False)
    with open(os.path.join(DATA, "run_info.json"), "w") as fh:
        json.dump(dict(run=args.run, thresholds=thr, config={k: v for k, v in cfg.items() if k != "columns"},
                       pick=PICK, matched=MATCHED), fh, indent=1, default=float)
    cv, cuts = curves(args.run, thr, cfg["r"])
    cv.to_csv(os.path.join(DATA, "curves.csv"), index=False)
    cuts.to_csv(os.path.join(DATA, "cutoffs.csv"), index=False)
    tuning(summ).to_csv(os.path.join(DATA, "tuning.csv"), index=False)
    failures(args.run, thr).to_csv(os.path.join(DATA, "failures.csv"), index=False)
    ranker_metrics(args.run, thr, PICK)
    examples(args.run)
    if not args.skip_importance:
        importance(args.run)


if __name__ == "__main__":
    main()
