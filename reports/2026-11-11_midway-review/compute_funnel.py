"""Midway review: per-period-bin loss funnel and chance-level checks on dev_medium.

Reads existing outputs only (no pipeline runs):
  * <main>/runs/t10_dev_medium_{fullyear,assigned}/{stars,candidates}.csv.gz  (T10 tables)
  * <main>/runs/t27_ranker/v0_{fullyear,assigned}_top1.csv                   (ranker v0 top-1)
  * benchmarks/detectability_dev.csv                                         (T26 side table)

Writes to this report's data/:
  * funnel.csv        stage fractions per (mode, bin), plus chance levels
  * weighting.csv     headline success under different population weightings
  * chance_final.csv  observed vs permutation-null success of the hand pipeline's answers

Run from the repo root (worktree) with the ss_openblas python:
  python reports/2026-11-11_midway-review/compute_funnel.py --main /path/to/main/checkout
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "claude_files"))
from evaluation import REL_TOL, classify_period  # noqa: E402

BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
THRESH = 1.3571535348892212   # ranker v0 decline threshold (runs/t27_ranker/results.json)
DET_CUT = 5.0                 # T26 proposed detectable cut on D_band*sqrt(N_cyc) (snr_rot)
N_PERM = 200


def is_success(measured, row):
    """True if `measured` scores correct or diffrot against the star in `row`."""
    rng = (row.period_lo, row.period_hi)
    return classify_period(measured, row.true_period, REL_TOL, rng) in ("correct", "diffrot")


def load(main, mode):
    """Join the T10 star table, ranker v0 top-1 and T26 detectability for one coverage mode."""
    s = pd.read_csv(f"{main}/runs/t10_dev_medium_{mode}/stars.csv.gz")
    t = pd.read_csv(f"{main}/runs/t27_ranker/v0_{mode}_top1.csv")
    t = t[t.is_rotator].set_index("star_id")
    s = s.set_index("star_id")
    s["top_rel"] = t.top_relevance
    s["top_score"] = t.top_score
    d = pd.read_csv(os.path.join(REPO, "benchmarks/detectability_dev.csv"))
    d = d[d.modes.str.contains(mode)].drop_duplicates("star_id").set_index("star_id")
    s["detectability"] = d.snr_rot
    s["detectable"] = s.detectability >= DET_CUT
    c = pd.read_csv(f"{main}/runs/t10_dev_medium_{mode}/candidates.csv.gz",
                    usecols=["star_id", "period", "fitted_P", "beyond_reach", "fit_error", "relevance"])
    c["P_eval"] = c.fitted_P.where(c.fitted_P.notna(), c.period)
    fitted = c[~c.beyond_reach]
    return s.reset_index(), fitted


def chance_candidate(s, fitted, rng):
    """Fraction of stars per bin with a success-window candidate when truth is shuffled within the bin."""
    groups = {k: g.P_eval.to_numpy() for k, g in fitted.groupby("star_id")}
    out = {b: [] for b in BINS}
    for _ in range(N_PERM // 10):
        for b in BINS:
            sb = s[s.period_bin == b]
            perm = sb.iloc[rng.permutation(len(sb))]
            hits = 0
            for sid, (_, tr) in zip(sb.star_id, perm.iterrows()):
                ps = groups.get(sid, np.array([]))
                hits += any(is_success(p, tr) for p in ps)
            out[b].append(hits / len(sb))
    return {b: float(np.mean(v)) for b, v in out.items()}


def chance_final(s, rng):
    """Observed hand-pipeline success among answered stars vs answers shuffled within the bin."""
    rows = []
    for b in BINS:
        a = s[(s.period_bin == b) & s.final_period.notna()]
        obs = np.mean([is_success(p, r) for p, (_, r) in zip(a.final_period, a.iterrows())])
        null = []
        for _ in range(N_PERM):
            ps = a.final_period.to_numpy()[rng.permutation(len(a))]
            null.append(np.mean([is_success(p, r) for p, (_, r) in zip(ps, a.iterrows())]))
        rows.append(dict(bin=b, n_answered=len(a), obs_success_given_answer=obs,
                         null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95))))
    return pd.DataFrame(rows)


def main():
    """Compute and save the funnel, weighting and chance tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default="/Users/raeholcomb/Documents/GitHub/ss_redevelopment")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    funnel, weights, finals = [], [], []
    for mode in ("fullyear", "assigned"):
        s, fitted = load(args.main, mode)
        ch = chance_candidate(s, fitted, rng)
        s["hand_ok"] = s.final_outcome.isin(["correct", "diffrot"])
        s["cand"] = s.n_relevant_fitted > 0
        s["top1"] = s.top_rel >= 1
        s["ranker_ok"] = s.top1 & (s.top_score >= THRESH)
        for b in BINS:
            g = s[s.period_bin == b]
            det = g.detectable
            funnel.append(dict(
                mode=mode, bin=b, n=len(g),
                detectable=det.mean(),
                det_cand=(det & g.cand).mean(),
                det_cand_top1=(det & g.cand & g.top1).mean(),
                det_ranker_answer_ok=(det & g.ranker_ok).mean(),
                cand_any=g.cand.mean(), cand_chance=ch[b],
                top1_any=g.top1.mean(),
                ranker_answer_ok=g.ranker_ok.mean(),
                hand_success=g.hand_ok.mean(), hand_answered=g.pipeline_success.mean(),
                hand_wrong=(g.final_outcome == "wrong").mean(),
            ))
        # headline under different population weightings (per-bin means re-weighted)
        per_bin = s.groupby("period_bin")[["hand_ok", "ranker_ok", "top1"]].mean().loc[BINS]
        logw = np.log([1 / 0.1, 3, 10 / 3, 3, 180 / 30])           # log-uniform period, 0.1-180 d
        corpus = s.groupby("period_bin").sampling_weight.sum().loc[BINS].to_numpy()  # SMARTS uniform-in-P
        for name, w in [("dev_medium as stratified", s.groupby("period_bin").size().loc[BINS].to_numpy()),
                        ("SMARTS corpus (uniform in P)", corpus),
                        ("log-uniform P, 0.1-180 d", logw),
                        ("P <= 30 d only, log-uniform", np.r_[logw[:4], 0.0])]:
            w = np.asarray(w, float) / np.sum(w)
            weights.append(dict(mode=mode, weighting=name,
                                hand_success=float(per_bin.hand_ok @ w),
                                ranker_v0_at_threshold=float(per_bin.ranker_ok @ w),
                                ranker_v0_top1_no_decline=float(per_bin.top1 @ w)))
        f = chance_final(s, rng)
        f.insert(0, "mode", mode)
        finals.append(f)
        if mode == "assigned":
            s["P_le_30"] = s.true_period <= 30
            cov = (s.groupby(["coverage_class", "P_le_30"])
                   .agg(success=("hand_ok", "mean"), ranker_answer_ok=("ranker_ok", "mean"),
                        n=("hand_ok", "size"), coverage_weight=("coverage_weight", "first"))
                   .reset_index())
    out = os.path.join(HERE, "data")
    cov.to_csv(f"{out}/coverage_class.csv", index=False, float_format="%.4f")
    pd.DataFrame(funnel).to_csv(f"{out}/funnel.csv", index=False, float_format="%.4f")
    pd.DataFrame(weights).to_csv(f"{out}/weighting.csv", index=False, float_format="%.4f")
    pd.concat(finals).to_csv(f"{out}/chance_final.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(funnel).to_string())
    print(pd.DataFrame(weights).to_string())
    print(pd.concat(finals).to_string())


if __name__ == "__main__":
    main()
