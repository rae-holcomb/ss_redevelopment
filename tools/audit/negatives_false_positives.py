#!/usr/bin/env python3
"""T27: false positives of the current pipeline (hand arbitration) on non-rotator negatives.

Reads the ``--nonrotator`` feature tables (``tools/build_feature_tables.py``, which runs the
scorecard pipeline step for step) for the negatives and the dev_medium rotator tables for
comparison, and writes:

    fp_by_coverage.csv     FP rate (any period reported) per set x coverage class (+ Wilson 95%)
    fp_by_source_class.csv per set x draw kind x quiet / systematic_common_mode
    fp_by_period_bin.csv   reported-period bins of the FPs, vs rotators' answers in the same bins
    decline_reasons.csv    decline_kind / failed gates on negatives vs rotators
    gate_pass_rates.csv    per gate: pass rate of fitted candidates, negatives vs rotators
    feature_auc.csv        per feature: AUC separating negatives' answers from rotators' successes
    fp_sources.csv         per source: answered draws, reported periods, stack LS P1
    fp_summary.json        headline numbers

Usage::

    python tools/audit/negatives_false_positives.py \\
        --neg-train /Volumes/Data_Vault/smarts_data/extracted/t27_negatives/train \\
        --neg-dev /Volumes/Data_Vault/smarts_data/extracted/t27_negatives/dev_medium
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402

MAIN = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment")
DEV_ROT = {"assigned": MAIN / "runs" / "t10_dev_medium_assigned",
           "fullyear": MAIN / "runs" / "t10_dev_medium_fullyear"}
OUTDIR = ROOT / "reports" / "2026-11-10_t27-nonrotator-negatives" / "data"
COV_ORDER = ["single", "consecutive-2", "gapped-2", "consecutive-3-5", "gapped-3-5",
             "consecutive-6-12", "gapped-6-12", "full", "fullyear"]
from comb_fit import GATE_NAMES as GATES  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval (lo, hi) for k successes in n trials."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def answered(stars: pd.DataFrame) -> np.ndarray:
    """True where the pipeline reported a period."""
    return stars["final_outcome"].ne(ev.NONE).to_numpy()


def draw_kind(stars: pd.DataFrame) -> pd.Series:
    """'fullyear' for the full-year row, 'assigned' for seeded coverage draws."""
    return np.where(stars["draw_id"].astype(str) == "full", "fullyear", "assigned")


def rate_table(stars: pd.DataFrame, by: list) -> pd.DataFrame:
    """FP rate with Wilson interval per group of ``by``."""
    rows = []
    for key, g in stars.groupby(by, sort=False):
        k, n = int(answered(g).sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), n=n, n_answered=k,
                         fp_rate=k / n if n else np.nan, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def load(table_dir) -> tuple:
    """(candidates, stars) of a feature-table directory."""
    d = Path(table_dir)
    return (pd.read_csv(d / "candidates.csv.gz", low_memory=False),
            pd.read_csv(d / "stars.csv.gz", low_memory=False))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC P(pos > neg) ignoring NaN; NaN if either side is empty."""
    from scipy.stats import mannwhitneyu
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--neg-train", required=True)
    ap.add_argument("--neg-dev", required=True)
    ap.add_argument("--fairness", default=str(OUTDIR / "fairness_sources.csv"))
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    neg_c, neg_s = [], []
    for d in (args.neg_train, args.neg_dev):
        c, s = load(d)
        neg_c.append(c)
        neg_s.append(s)
    nc, ns = pd.concat(neg_c, ignore_index=True), pd.concat(neg_s, ignore_index=True)
    ns["draw_kind"] = draw_kind(ns)
    ns["cov"] = np.where(ns.draw_kind == "fullyear", "fullyear", ns["assigned_coverage_class"])
    ns["answered"] = answered(ns)
    ns["reported_bin"] = [ev.period_bin(p) if a else "none" for p, a in zip(ns.final_period, ns.answered)]
    nc = nc.merge(ns[["star_id", "eval_set", "draw_kind", "cov"]], on="star_id", how="left")
    rot = {k: load(v) for k, v in DEV_ROT.items()}

    # --- FP rate by coverage, source class
    cov_t = rate_table(ns, ["eval_set", "cov"])
    cov_t["cov"] = pd.Categorical(cov_t["cov"], COV_ORDER, ordered=True)
    cov_t = cov_t.sort_values(["eval_set", "cov"])
    tot = rate_table(ns.assign(cov="all_" + ns.draw_kind), ["eval_set", "cov"])
    cov_t = pd.concat([cov_t, tot], ignore_index=True)
    # rotators' answer rate per coverage class, for comparison
    rrows = []
    for kind, (_, rs) in rot.items():
        rs = rs.assign(cov="fullyear" if kind == "fullyear" else rs["coverage_class"])
        for key, g in rs.groupby("cov"):
            a = answered(g)
            rrows.append(dict(cov=key, kind=kind, n=len(g), rot_answer_rate=a.mean(),
                              rot_success_rate=np.isin(g.final_outcome, ev.SUCCESS_OUTCOMES).mean()))
        a = answered(rs)
        rrows.append(dict(cov=f"all_{kind}", kind=kind, n=len(rs), rot_answer_rate=a.mean(),
                          rot_success_rate=np.isin(rs.final_outcome, ev.SUCCESS_OUTCOMES).mean()))
    rot_rates = pd.DataFrame(rrows)
    cov_t = cov_t.merge(rot_rates[["cov", "rot_answer_rate", "rot_success_rate"]], on="cov", how="left")
    cov_t.to_csv(out / "fp_by_coverage.csv", index=False)
    rate_table(ns, ["eval_set", "draw_kind", "source_class"]).to_csv(out / "fp_by_source_class.csv", index=False)

    # --- reported-period bins: negatives' FPs vs rotators' answers (dev_medium, same draw kind)
    prow = []
    for kind, (_, rs) in rot.items():
        rs = rs.assign(rb=[ev.period_bin(p) if o != ev.NONE else "none"
                           for p, o in zip(rs.final_period, rs.final_outcome)])
        nd = ns[(ns.eval_set == "dev_medium") & (ns.draw_kind == kind)]
        for b in ev.PERIOD_BIN_LABELS:
            r_ans = rs[rs.rb == b]
            prow.append(dict(kind=kind, reported_bin=b,
                             neg_fp=int((nd.reported_bin == b).sum()), neg_n=len(nd),
                             neg_fp_rate=float((nd.reported_bin == b).mean()),
                             rot_answers=len(r_ans), rot_n=len(rs),
                             rot_success=int(np.isin(r_ans.final_outcome, ev.SUCCESS_OUTCOMES).sum()),
                             rot_wrong=int(r_ans.final_outcome.isin([ev.WRONG, ev.HALF, ev.DOUBLE]).sum())))
    pb = pd.DataFrame(prow)
    # precision per reported bin if negatives were as common as rotators (rates, so n-independent)
    pb["precision_rot_only"] = pb.rot_success / pb.rot_answers.replace(0, np.nan)
    pb["precision_1to1"] = (pb.rot_success / pb.rot_n) / ((pb.rot_answers / pb.rot_n) + pb.neg_fp_rate)
    pb.to_csv(out / "fp_by_period_bin.csv", index=False)

    # --- decline reasons
    drow = []
    for name, s in [("neg_" + k, ns[ns.draw_kind == k]) for k in ("assigned", "fullyear")] + \
                   [("rot_" + k, v[1]) for k, v in rot.items()]:
        n = len(s)
        for k, c in s.decline_kind.fillna("").replace("", "answered").value_counts().items():
            drow.append(dict(table=name, what="decline_kind", value=k, frac=c / n, n=n))
        gates = s.decline_gates.dropna().astype(str).str.split(";").explode()
        for k, c in gates[gates != ""].value_counts().items():
            drow.append(dict(table=name, what="decline_gate", value=k, frac=c / n, n=n))
    pd.DataFrame(drow).to_csv(out / "decline_reasons.csv", index=False)

    # --- per-gate pass rates on fitted candidates
    grow = []
    fitted = lambda c: c[(c.fit_error.astype(str) == "False") & (c.beyond_reach.astype(str) == "False")]  # noqa: E731
    tabs = [("neg_" + k, fitted(nc[nc.draw_kind == k])) for k in ("assigned", "fullyear")]
    for k, (rc, _) in rot.items():
        f = fitted(rc)
        tabs += [("rot_" + k, f), ("rot_" + k + "_relevant", f[f.relevance >= 1])]
    for name, c in tabs:
        for g in GATES:
            col = f"gate_{g}_pass"
            if col in c:
                v = c[col].astype(str).str.lower()
                grow.append(dict(table=name, gate=g, n=len(c), pass_rate=float((v == "true").mean())))
        grow.append(dict(table=name, gate="ALL (passed_gates)", n=len(c),
                         pass_rate=float((c.n_failed_gates == 0).mean())))
    pd.DataFrame(grow).to_csv(out / "gate_pass_rates.csv", index=False)

    # --- features of the answer: negatives' FPs vs rotators' successes (same draw kind)
    sys.path.insert(0, str(ROOT / "tools"))
    frow = []
    for kind, (rc, rs) in rot.items():
        ok = set(rs.star_id[np.isin(rs.final_outcome, ev.SUCCESS_OUTCOMES)])
        rsel = rc[(rc.selected.astype(str) == "True") & rc.star_id.isin(ok)]
        nsel = nc[(nc.selected.astype(str) == "True") & (nc.draw_kind == kind)]
        cols = [c for c in rsel.columns if c in nsel.columns and c not in (
            "star_id", "cand_index") and pd.api.types.is_numeric_dtype(rsel[c])
            and c not in ("true_period", "period_lo", "period_hi", "relevance", "noise_tic")]
        for c in cols:
            a = auc(nsel[c].to_numpy(float), rsel[c].to_numpy(float))
            frow.append(dict(kind=kind, feature=c, auc_neg_gt_rot=a, sep=abs(a - 0.5) if np.isfinite(a) else np.nan,
                             neg_median=float(np.nanmedian(nsel[c])) if nsel[c].notna().any() else np.nan,
                             rot_median=float(np.nanmedian(rsel[c])) if rsel[c].notna().any() else np.nan,
                             n_neg=len(nsel), n_rot=len(rsel)))
    fa = pd.DataFrame(frow).sort_values(["kind", "sep"], ascending=[True, False])
    fa.to_csv(out / "feature_auc.csv", index=False)

    # --- per source
    fair = pd.read_csv(args.fairness)[["tic", "P1", "pow1", "amp1_ppm"]] if Path(args.fairness).exists() else None
    src = ns.groupby("noise_tic").agg(eval_set=("eval_set", "first"), source_class=("source_class", "first"),
                                      n_draws=("star_id", "size"), n_answered=("answered", "sum"),
                                      fullyear_answered=("answered", lambda s: bool(
                                          s[ns.loc[s.index, "draw_kind"] == "fullyear"].any())),
                                      periods=("final_period", lambda s: " ".join(
                                          f"{p:.3f}" for p in s.dropna()))).reset_index()
    if fair is not None:
        src = src.merge(fair.rename(columns={"tic": "noise_tic"}), on="noise_tic", how="left")
    src.to_csv(out / "fp_sources.csv", index=False)
    # do repeated FPs on one source agree (a source-level signal) or scatter (noise)?
    rep = []
    for _, g in ns[ns.answered].groupby("noise_tic"):
        p = g.final_period.to_numpy(float)
        if len(p) >= 2:
            rep.append(float(np.max(p) / np.min(p)) < 1.1)
    # orbit-related reported periods
    fp = ns[ns.answered]
    near = lambda P, x: np.abs(P / x - 1) < ev.REL_TOL  # noqa: E731
    summ = dict(
        n_negative_rows=int(len(ns)), n_sources=int(ns.noise_tic.nunique()),
        fp_rate={f"{s}|{k}": float(g.answered.mean()) for (s, k), g in ns.groupby(["eval_set", "draw_kind"])},
        fp_rate_by_class={f"{s}|{k}|{c}": float(g.answered.mean())
                          for (s, k, c), g in ns.groupby(["eval_set", "draw_kind", "source_class"])},
        rot_answer_rate={k: float(answered(v[1]).mean()) for k, v in rot.items()},
        rot_success_rate={k: float(np.isin(v[1].final_outcome, ev.SUCCESS_OUTCOMES).mean()) for k, v in rot.items()},
        frac_sources_any_fp=float((src.n_answered > 0).mean()),
        frac_multi_fp_sources_consistent_period=float(np.mean(rep)) if rep else np.nan,
        n_multi_fp_sources=len(rep),
        fp_period_median=float(fp.final_period.median()) if len(fp) else np.nan,
        fp_frac_near_13p7d=float(near(fp.final_period, 13.7).mean()) if len(fp) else np.nan,
        fp_frac_near_27p4d=float(near(fp.final_period, 27.4).mean()) if len(fp) else np.nan,
        fp_frac_gt_10d=float((fp.final_period > 10).mean()) if len(fp) else np.nan)
    if fair is not None and len(fp):
        m = fp.merge(fair.rename(columns={"tic": "noise_tic"}), on="noise_tic", how="left")
        summ["fp_frac_match_stack_LS_P1"] = float(near(m.final_period, m.P1).mean())
    with open(out / "fp_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
