"""T11 ranker v0 report analysis: tables in data/ (compute) and figures in figures/ (from data/).

  python reports/2026-11-10_t11-ranker-v0/analysis.py --run runs/t11_ranker --compute
  python reports/2026-11-10_t11-ranker-v0/analysis.py --run runs/t11_ranker           # figures only

The compute step reads the training table (drive) for permutation importance and the dev_medium
FITS (data/smarts_dev) for the example-star plots; the figure step needs only data/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "claude_files"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import determinism  # noqa: E402

determinism.pin_blas_threads()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation  # noqa: E402
import ranking_model as rm  # noqa: E402
import train_ranker as tr  # noqa: E402

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
BINS = list(evaluation.PERIOD_BIN_LABELS)
THR_MAIN = "match_hand_precision"


def _res(run):
    """All results/<key>.json files of a run, keyed by key."""
    return {os.path.basename(p)[:-5]: json.load(open(p))
            for p in glob.glob(os.path.join(run, "results", "*.json"))}


def _head_key(run, res):
    """Key of the headline model: largest base run of the headline objective."""
    obj = json.load(open(os.path.join(run, "headline.json")))["objective"]
    ks = [k for k, v in res.items() if v["variant"] == "base" and v["objective"] == obj]
    return max(ks, key=lambda k: res[k]["n_train_stars"]), obj


def _flat(records, **tags):
    """outcome_summary records -> DataFrame rows with extra tag columns."""
    return pd.DataFrame([dict(r, **tags) for r in records])


# ------------------------------------------------------------------ compute
def compute(run):
    """Build every data/ table the README and figures use."""
    os.makedirs(DATA, exist_ok=True)
    res = _res(run)
    base = json.load(open(os.path.join(run, "baselines.json")))
    head, obj = _head_key(run, res)
    json.dump(dict(head_key=head, objective=obj, thresholds=res[head]["thresholds"],
                   hand_train=res[head]["hand_train"], params=res[head]["params"],
                   n_rounds=res[head]["n_rounds"], tune=json.load(open(os.path.join(run, "tune.json")))),
              open(os.path.join(DATA, "headline_model.json"), "w"), indent=1, default=float)
    pd.read_csv(os.path.join(run, "tune.csv")).to_csv(os.path.join(DATA, "tune.csv"), index=False)

    # headline comparison
    rows = []
    for d in ("fullyear", "assigned"):
        for s in tr.SUBSETS:
            rows.append(_flat(base[d][f"hand|{s}"], dev=d, subset=s, method="hand arbitration"))
            rows.append(_flat(base[d][f"oracle|{s}"], dev=d, subset=s, method="oracle (best in list)"))
            for t in res[head]["thresholds"]:
                rows.append(_flat(res[head]["dev"][d][f"{t}|{s}"], dev=d, subset=s, method=f"ranker {t}"))
    pd.concat(rows).to_csv(os.path.join(DATA, "comparison.csv"), index=False)

    # learning curve, ablations, objective comparison: overall rows only
    rows = []
    for k, v in res.items():
        for d in ("fullyear", "assigned"):
            for s in ("all", "strict_quiet"):
                for t in (THR_MAIN, "no_decline"):
                    r = [x for x in v["dev"][d][f"{t}|{s}"] if x["bin"] == "all"][0]
                    rows.append(dict(key=k, n_train=v["n_train_stars"], variant=v["variant"],
                                     objective=v["objective"], dev=d, subset=s, threshold=t, **r))
        o = [x for x in v["oof"][f"{THR_MAIN}|all"] if x["bin"] == "all"][0]
        o2 = [x for x in v["oof"]["no_decline|all"] if x["bin"] == "all"][0]
        rows.append(dict(key=k, n_train=v["n_train_stars"], variant=v["variant"], objective=v["objective"],
                         dev="train_oof", subset="all", threshold=THR_MAIN, **o))
        rows.append(dict(key=k, n_train=v["n_train_stars"], variant=v["variant"], objective=v["objective"],
                         dev="train_oof", subset="all", threshold="no_decline", **o2))
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(DATA, "runs_overall.csv"), index=False)

    # ablation detail: correct vs diffrot at top, wide-band vs rest, fit-error picks (dev fullyear)
    rows = []
    for k, v in res.items():
        if v["n_train_stars"] != res[head]["n_train_stars"]:
            continue
        for d in ("fullyear", "assigned"):
            t = pd.read_csv(os.path.join(run, "results", f"{k}_{d}_top1.csv"))
            dec = rm.decide(t, v["thresholds"][THR_MAIN])
            wide = t.band_width_class == "ge_0p5"
            for name, m in (("all", np.ones(len(t), bool)), ("ge_0p5", wide), ("not_ge_0p5", ~wide)):
                rows.append(dict(key=k, variant=v["variant"], objective=v["objective"], dev=d, stars=name,
                                 n=int(m.sum()),
                                 success=float(np.isin(dec[m], evaluation.SUCCESS_OUTCOMES).mean()),
                                 correct=float((dec[m] == "correct").mean()),
                                 diffrot=float((dec[m] == "diffrot").mean()),
                                 declined=float((dec[m] == "none").mean()),
                                 top1_success_no_decline=float(np.isin(t.top_outcome[m], evaluation.SUCCESS_OUTCOMES).mean()),
                                 top_is_fit_error=float((t.top_kind[m] == "fit_error").mean()),
                                 answered_fit_error=float(((t.top_kind == "fit_error") & (dec != "none"))[m].mean())))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "ablation_detail.csv"), index=False)

    # threshold-free variant comparison: success when every variant answers the same fraction
    # of dev stars (hand arbitration's dev coverage). A reading on dev, never used for selection.
    rows = []
    for k, v in res.items():
        for d in ("fullyear", "assigned"):
            t = pd.read_csv(os.path.join(run, "results", f"{k}_{d}_top1.csv"))
            hc = tr.hand_point(rm.load_table(tr.DEFAULT_DEV[d])[1])["coverage"]
            c = rm.coverage_accuracy_curve(t)
            p = c[c.coverage <= hc + 1e-12].iloc[-1]
            rows.append(dict(key=k, n_train=v["n_train_stars"], variant=v["variant"], objective=v["objective"],
                             dev=d, coverage=float(p.coverage), success=float(p.success_rate),
                             precision=float(p.precision)))
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fixed_coverage.csv"), index=False)

    # coverage-accuracy curves (OOF and dev) for the headline model
    curves = []
    oof = pd.read_csv(os.path.join(run, "results", f"{head}_oof_top1.csv"))
    curves.append(rm.coverage_accuracy_curve(oof).assign(set="train OOF", subset="all"))
    dev_tops = {}
    for d in ("fullyear", "assigned"):
        t = pd.read_csv(os.path.join(run, "results", f"{head}_{d}_top1.csv"))
        dev_tops[d] = t
        ds = rm.load_table(tr.DEFAULT_DEV[d])[1].set_index("star_id").loc[t.star_id].reset_index()
        t["strict_quiet"] = rm.strict_quiet_mask(ds).values
        for s, m in (("all", np.ones(len(t), bool)), ("strict_quiet", t.strict_quiet.values)):
            curves.append(rm.coverage_accuracy_curve(t[m]).assign(set=f"dev {d}", subset=s))
    pd.concat(curves).to_csv(os.path.join(DATA, "coverage_curves.csv"), index=False)
    hp = {d: tr.hand_point(rm.load_table(tr.DEFAULT_DEV[d])[1]) for d in ("fullyear", "assigned")}
    hp["train"] = res[head]["hand_train"]
    json.dump(hp, open(os.path.join(DATA, "hand_points.json"), "w"), indent=1)

    # failure modes (both dev modes) + beyond-reach + diffrot bias
    thr = res[head]["thresholds"][THR_MAIN]
    fm = []
    for d, t in dev_tops.items():
        t = t.copy()
        t["ranker"] = rm.decide(t, thr)
        hs = np.isin(t.final_outcome, evaluation.SUCCESS_OUTCOMES)
        rs = np.isin(t.ranker, evaluation.SUCCESS_OUTCOMES)
        t["case"] = np.select([hs & ~rs, rs & ~hs, hs & rs], ["hand wins", "ranker wins", "both succeed"], "neither")
        why = np.where(t.ranker == "none",
                       np.where(np.isin(t.top_outcome, evaluation.SUCCESS_OUTCOMES), "declined (top was right)",
                                "declined (top was wrong)"),
                       np.where(np.isin(t.ranker, evaluation.ALIAS_OUTCOMES), "answered alias", "answered wrong"))
        t["why"] = np.where(rs, "", why)
        t["dev"] = d
        fm.append(t)
    fm = pd.concat(fm)
    fm.to_csv(os.path.join(DATA, "per_star_dev.csv.gz"), index=False)
    oof["ranker"] = rm.decide(oof, thr)
    oof["dev"] = "train_oof"
    bias = []
    for d, t in list(fm.groupby("dev")) + [("train_oof", oof)]:
        for cls, m in (("has correct cand (max rel 2)", t.max_relevance == 2),
                       ("diffrot-only (max rel 1)", t.max_relevance == 1),
                       ("top is correct", t.top_relevance == 2), ("top is diffrot", t.top_relevance == 1)):
            sub = t[m]
            bias.append(dict(set=d, stars=cls, n=len(sub), answered=float((sub.ranker != "none").mean()),
                             success=float(np.isin(sub.ranker, evaluation.SUCCESS_OUTCOMES).mean()),
                             hand_success=float(np.isin(sub.final_outcome, evaluation.SUCCESS_OUTCOMES).mean()),
                             median_top_score=float(sub.top_score.median())))
    pd.DataFrame(bias).to_csv(os.path.join(DATA, "diffrot_bias.csv"), index=False)
    br = fm.groupby(["dev", "top_kind"]).apply(lambda g: pd.Series(dict(
        n=len(g), answered=(g.ranker != "none").mean(),
        top_success=np.isin(g.top_outcome, evaluation.SUCCESS_OUTCOMES).mean(),
        success_if_answered=np.isin(g.ranker[g.ranker != "none"], evaluation.SUCCESS_OUTCOMES).mean()
        if (g.ranker != "none").any() else np.nan)), include_groups=False).reset_index()
    br.to_csv(os.path.join(DATA, "top_kind.csv"), index=False)

    # importance: gain, SHAP (dev full-year rows), grouped permutation on OOF (train)
    mdir = os.path.join(run, "models", head)
    import xgboost as xgb
    models = []
    for k in range(5):
        b = xgb.Booster()
        b.load_model(os.path.join(mdir, f"fold{k}.json"))
        models.append(b)
    cols = json.load(open(os.path.join(mdir, "columns.json")))
    gain = rm.gain_importance(models, cols)
    dc, ds = rm.load_table(tr.DEFAULT_DEV["fullyear"])
    Xd, _ = rm.feature_matrix(dc, cols)
    sh = rm.shap_contributions(models, Xd)
    imp = pd.DataFrame(dict(feature=cols, gain=gain.reindex(cols).values,
                            mean_abs_shap=sh[cols].abs().mean().values))
    imp["family"] = imp.feature.map(rm.feature_family)
    imp["method"] = imp.feature.map(rm.feature_method).fillna("")
    # SHAP on relevant (rel>=1) rows split by relevance, for the half-lag check
    for lvl in (1, 2):
        imp[f"mean_shap_rel{lvl}"] = sh.loc[dc.relevance.values == lvl, cols].mean().values
    imp["mean_shap_rel0"] = sh.loc[dc.relevance.values == 0, cols].mean().values
    imp.to_csv(os.path.join(DATA, "importance.csv"), index=False)

    tc, ts = rm.load_table(res[head]["train_dir"])
    ids = rm.stratified_prefix(ts, res[head]["n_train_stars"])
    tc, ts = tr.restrict(tc, ts, ids)
    Xt, _ = rm.feature_matrix(tc, cols)
    fold = rm.grouped_folds(ts, 5, 0)
    rf = tc.star_id.map(fold).to_numpy()

    def oof_success(X):
        sc = np.full(len(tc), np.nan)
        for k in range(5):
            m = rf == k
            sc[m] = rm.predict(models[k], X[m])
        return float(rm._success(rm.top1(tc, sc, ts).top_outcome).mean())

    ref = oof_success(Xt)
    rng = np.random.default_rng(0)
    perm = []
    groups = {**{f"family:{f}": list(imp.feature[imp.family == f]) for f in imp.family.unique()},
              **{f"method:{m}": list(imp.feature[imp.method == m]) for m in imp.method.unique() if m}}
    for g, fs in groups.items():
        drops = []
        for rep in range(3):
            Xp = Xt.copy()
            idx = rng.permutation(len(Xp))
            Xp[fs] = Xt[fs].values[idx]  # permute the family's columns jointly across rows
            drops.append(ref - oof_success(Xp))
        perm.append(dict(group=g, n_features=len(fs), drop_mean=float(np.mean(drops)),
                         drop_std=float(np.std(drops))))
    pd.DataFrame(perm).assign(ref_oof_top1_success=ref).to_csv(os.path.join(DATA, "perm_importance.csv"), index=False)

    corr = []
    for f in imp.family.unique():
        fs = list(imp.feature[imp.family == f])
        if len(fs) < 2:
            continue
        C = Xt[fs].rank().corr().abs().values
        iu = np.triu_indices(len(fs), 1)
        hi = [(fs[i], fs[j], C[i, j]) for i, j in zip(*iu) if C[i, j] > 0.95]
        corr.append(dict(family=f, n_features=len(fs), n_pairs_rho_gt_0p95=len(hi),
                         examples="; ".join(f"{a}~{b} ({c:.2f})" for a, b, c in hi[:4])))
    pd.DataFrame(corr).to_csv(os.path.join(DATA, "family_correlation.csv"), index=False)

    # example stars (full-year): 4 hand wins in different bins, 2 ranker wins
    ex = fm[fm.dev == "fullyear"]
    pick = []
    for case, n in (("hand wins", 4), ("ranker wins", 2)):
        sub = ex[ex.case == case].sort_values("star_id")
        for b in BINS:
            if len([p for p in pick if p[1] == case]) >= n:
                break
            s = sub[sub.period_bin == b]
            if len(s):
                pick.append((s.star_id.iloc[0], case))
    exrows = []
    from preprocessing import load_smarts_fits
    for sid, case in pick:
        r = ex[ex.star_id == sid].iloc[0]
        lc = load_smarts_fits(os.path.join(ROOT, "data", "smarts_dev", sid + ".fits"))
        np.savez_compressed(os.path.join(DATA, f"lc_{sid}.npz"), time=lc.time, flux=lc.flux)
        exrows.append(dict(star_id=sid, case=case, true_period=r.true_period, period_lo=r.period_lo,
                           period_hi=r.period_hi, hand_period=r.final_period, hand_outcome=r.final_outcome,
                           ranker_period=r.top_period, ranker_outcome=r.ranker, top_outcome=r.top_outcome,
                           top_kind=r.top_kind, why=r.why, hand_choice_rank=r.hand_choice_rank))
    pd.DataFrame(exrows).to_csv(os.path.join(DATA, "examples.csv"), index=False)


# ------------------------------------------------------------------ figures
def _save(fig, name):
    """Save a figure into figures/ and close it."""
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_comparison():
    """Per-bin success/alias/wrong/declined bars: hand vs ranker vs oracle, both dev modes and subsets."""
    c = pd.read_csv(os.path.join(DATA, "comparison.csv"))
    meths = ["hand arbitration", f"ranker {THR_MAIN}", "ranker match_hand_coverage", "oracle (best in list)"]
    labels = ["hand", "ranker\n(prec.)", "ranker\n(cov.)", "oracle"]
    cols = {"success": "#2a9d8f", "alias": "#e9c46a", "wrong": "#e76f51", "declined": "#cfd2d6"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharey=True)
    for i, d in enumerate(("fullyear", "assigned")):
        for j, s in enumerate(("all", "strict_quiet")):
            ax = axes[i, j]
            x0 = 0
            ticks, tl = [], []
            for b in ["all"] + BINS:
                for k, m in enumerate(meths):
                    r = c[(c.dev == d) & (c.subset == s) & (c.method == m) & (c.bin == b)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    bottom = 0
                    for o in ("success", "alias", "wrong", "declined"):
                        v = 0 if m.startswith("oracle") and o == "declined" else r[o]
                        ax.bar(x0 + k, v, bottom=bottom, color=cols[o], width=0.85,
                               label=o if (b == "all" and k == 0) else None,
                               hatch="//" if m.startswith("oracle") else None, edgecolor="white", lw=0.3)
                        bottom += v
                    ax.text(x0 + k, r.success + 0.01, f"{r.success:.2f}", ha="center", fontsize=6, rotation=90)
                ticks.append(x0 + 1.5)
                tl.append(f"{b}\n(n={int(r.n)})")
                x0 += len(meths) + 1.2
            ax.set_xticks(ticks, tl, fontsize=8)
            ax.set_title(f"dev_medium {d}, {s} stars  (bars per bin: {', '.join(l.replace(chr(10), ' ') for l in labels)})", fontsize=8)
            ax.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel("fraction of stars")
    axes[1, 0].set_ylabel("fraction of stars")
    _save(fig, "fig_comparison_by_bin.png")


def fig_coverage():
    """Coverage vs precision curves (train OOF and dev) with hand arbitration and the chosen thresholds."""
    cu = pd.read_csv(os.path.join(DATA, "coverage_curves.csv"))
    hp = json.load(open(os.path.join(DATA, "hand_points.json")))
    hm = json.load(open(os.path.join(DATA, "headline_model.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (setn, hpk) in zip(axes, (("dev fullyear", "fullyear"), ("dev assigned", "assigned"))):
        o = cu[cu.set == "train OOF"]
        ax.plot(o.coverage, o.precision, color="0.5", lw=1, label="train OOF (calibration)")
        for s, ls in (("all", "-"), ("strict_quiet", "--")):
            d = cu[(cu.set == setn) & (cu.subset == s)]
            ax.plot(d.coverage, d.precision, ls, color="C0", label=f"{setn}, {s}")
            for t, mk in (("match_hand_precision", "o"), ("match_hand_coverage", "s")):
                thr = hm["thresholds"][t]
                p = d[d.threshold >= thr].tail(1)
                if len(p) and s == "all":
                    ax.plot(p.coverage, p.precision, mk, color="C0", ms=7, mfc="white",
                            label=f"ranker @ {t}")
        ax.plot(hp[hpk]["coverage"], hp[hpk]["precision"], "*", color="C3", ms=14, label="hand arbitration (dev)")
        ax.plot(hp["train"]["coverage"], hp["train"]["precision"], "*", color="0.5", ms=10, label="hand arbitration (train)")
        for sr in (0.2, 0.3, 0.4):
            xx = np.linspace(sr, 1, 50)
            ax.plot(xx, sr / xx, ":", color="0.8", lw=0.8)
            ax.text(1.0, sr, f"success={sr}", fontsize=6, color="0.6", ha="right", va="bottom")
        ax.set_xlabel("coverage (fraction of stars answered)")
        ax.set_ylabel("precision (success among answered)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 1)
        ax.set_title(setn)
        ax.legend(fontsize=7, loc="lower left")
    _save(fig, "fig_coverage_accuracy.png")


def fig_learning():
    """Learning curve: success at the matched-precision threshold and without decline vs training stars."""
    r = pd.read_csv(os.path.join(DATA, "runs_overall.csv"))
    hm = json.load(open(os.path.join(DATA, "headline_model.json")))
    r = r[(r.variant == "base") & (r.objective == hm["objective"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, t in zip(axes, (THR_MAIN, "no_decline")):
        for (d, s), g in r[r.threshold == t].groupby(["dev", "subset"]):
            g = g.sort_values("n_train")
            err = np.sqrt(g.success * (1 - g.success) / g.n)
            ax.errorbar(g.n_train, g.success, err, marker="o", capsize=3, label=f"{d} / {s}")
        ax.set_xscale("log")
        ax.set_xticks(sorted(r.n_train.unique()), [str(x) for x in sorted(r.n_train.unique())])
        ax.set_xlabel("training stars")
        ax.set_ylabel("success fraction (all stars in set)")
        ax.set_title(f"threshold: {t}")
        ax.legend(fontsize=7)
    _save(fig, "fig_learning_curve.png")


def fig_importance():
    """Importance grouped by family and by guess_* method: SHAP, gain and grouped permutation."""
    imp = pd.read_csv(os.path.join(DATA, "importance.csv"))
    perm = pd.read_csv(os.path.join(DATA, "perm_importance.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, key in enumerate(("family", "method")):
        g = imp[imp[key].fillna("") != ""].groupby(key)[["mean_abs_shap", "gain"]].sum()
        g = g.sort_values("mean_abs_shap")
        axes[row, 0].barh(g.index, g.mean_abs_shap, color="C0")
        axes[row, 0].set_title(f"sum of mean |SHAP| by {key} (dev full-year rows)", fontsize=9)
        axes[row, 1].barh(g.index, g.gain / g.gain.sum(), color="C1")
        axes[row, 1].set_title(f"share of total gain by {key}", fontsize=9)
        p = perm[perm.group.str.startswith(key + ":")].copy()
        p["name"] = p.group.str.split(":").str[1]
        p = p.set_index("name").reindex(g.index)
        axes[row, 2].barh(p.index, p.drop_mean, xerr=p.drop_std, color="C2")
        axes[row, 2].set_title(f"drop in OOF top-1 success when {key} permuted", fontsize=9)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    _save(fig, "fig_importance.png")
    top = imp.sort_values("mean_abs_shap", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.feature + "  [" + top.family + "]", top.mean_abs_shap, color="C0")
    ax.set_xlabel("mean |SHAP| (dev full-year candidate rows)")
    ax.tick_params(labelsize=7)
    _save(fig, "fig_top_features.png")


def fig_failures():
    """Where the ranker loses to (and beats) hand arbitration, by bin and reason."""
    fm = pd.read_csv(os.path.join(DATA, "per_star_dev.csv.gz"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, d in zip(axes[:2], ("fullyear", "assigned")):
        t = fm[(fm.dev == d) & (fm.case == "hand wins")]
        whys = {"declined (top was right)": "#f4d35e", "declined (top was wrong)": "#b0b0b0",
                "answered alias": "#e9a03b", "answered wrong": "#e76f51"}
        ct = pd.crosstab(t.period_bin, t.why).reindex(index=BINS, columns=list(whys)).fillna(0)
        ct.plot.bar(stacked=True, ax=ax, rot=0, color=list(whys.values()))
        w = fm[(fm.dev == d) & (fm.case == "ranker wins")].period_bin.value_counts().reindex(BINS).fillna(0)
        ax.plot(range(len(BINS)), w.values, "k_", ms=25, mew=2, label="ranker wins (count)")
        ax.set_title(f"{d}: stars hand arbitration gets and ranker misses", fontsize=9)
        ax.set_ylabel("stars")
        ax.legend(fontsize=7)
    t = fm[(fm.dev == "fullyear") & (fm.case.isin(["hand wins", "ranker wins"]))]
    ax = axes[2]
    for case, c in (("hand wins", "C3"), ("ranker wins", "C2")):
        s = t[t.case == case]
        ax.scatter(s.final_period / s.true_period, s.top_period / s.true_period, s=12, c=c, label=case, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for v in (0.5, 1, 2):
        ax.axhline(v, color="0.8", lw=0.7)
        ax.axvline(v, color="0.8", lw=0.7)
    ax.set_xlabel("hand-arbitration period / P_eq (NaN = declined, not shown)")
    ax.set_ylabel("ranker top-1 period / P_eq")
    ax.set_title("full-year disagreements", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, "fig_failure_modes.png")


def fig_examples():
    """Light curve plus phase folds at truth, hand and ranker periods for the example stars."""
    ex = pd.read_csv(os.path.join(DATA, "examples.csv"))
    fig, axes = plt.subplots(len(ex), 4, figsize=(16, 2.6 * len(ex)))
    for i, r in ex.iterrows():
        z = np.load(os.path.join(DATA, f"lc_{r.star_id}.npz"))
        t, f = z["time"], z["flux"]
        axes[i, 0].plot(t, f, ",k")
        axes[i, 0].set_title(f"{r.star_id[-6:]}  {r.case}  P_eq={r.true_period:.3g} d", fontsize=8)
        for j, (lab, P, o) in enumerate((("truth", r.true_period, "correct"),
                                         ("hand", r.hand_period, r.hand_outcome),
                                         ("ranker top-1", r.ranker_period, f"{r.ranker_outcome} / top={r.top_outcome}, {r.top_kind}"))):
            ax = axes[i, j + 1]
            if np.isfinite(P):
                ph = (t % P) / P
                ax.plot(ph, f, ",", color="C0", alpha=0.4)
                ax.set_title(f"{lab}: {P:.3g} d ({o})", fontsize=7)
            else:
                ax.set_title(f"{lab}: declined", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[i, 0].tick_params(labelsize=6)
    fig.tight_layout()
    _save(fig, "fig_examples.png")


def figures():
    """Regenerate every README figure from data/."""
    fig_comparison()
    fig_coverage()
    fig_learning()
    fig_importance()
    fig_failures()
    fig_examples()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "runs", "t11_ranker"))
    ap.add_argument("--compute", action="store_true")
    a = ap.parse_args()
    if a.compute:
        compute(a.run)
    figures()
