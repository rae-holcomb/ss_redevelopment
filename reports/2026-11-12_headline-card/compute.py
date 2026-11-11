"""P28/P29: compute the headline card, its sensitivity tables and the Roman-season proxy numbers.

Reads existing per-star outputs only (no pipeline runs, ~30 s):
  * benchmarks/baselines/dev_medium_{assigned,fullyear}_{t5,v5}_tables/per_star.csv  (hand, post/pre T5)
  * <main>/runs/t27_ranker/v0_{assigned,fullyear}_top1.csv  (ranker v0 top-1 + galaxy-stack negatives)
  * datasets/dev_medium.csv  (coverage_weight)

Writes this report's data/. Run from the repo root with the ss_openblas python:
  python reports/2026-11-12_headline-card/compute.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
import headline_card as hc  # noqa: E402

BINS = list(hc.PERIOD_BIN_LABELS)
CLASSES = ["single", "consecutive-2", "gapped-2", "consecutive-3-5", "gapped-3-5",
           "consecutive-6-12", "gapped-6-12", "full"]
N_WORKERS = 4


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k successes of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def systems(main, mode):
    """{label: (kind, frame with outcomes)} for hand T5, hand pre-T5 and ranker v0."""
    b = REPO / "benchmarks" / "baselines"
    return {
        "hand (T5)": ("hand", hc.add_outcomes(hc.load_hand(b / f"dev_medium_{mode}_t5_tables/per_star.csv"))),
        "hand (pre-T5)": ("hand", hc.add_outcomes(hc.load_hand(b / f"dev_medium_{mode}_v5_tables/per_star.csv"))),
        "ranker v0": ("ranker", hc.add_outcomes(hc.load_ranker(f"{main}/runs/t27_ranker/v0_{mode}_top1.csv"))),
    }


def main():
    """Compute and save every table the notebook plots."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default="/Users/raeholcomb/Documents/GitHub/ss_redevelopment")
    args = ap.parse_args()
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    manifest = pd.read_csv(REPO / "datasets" / "dev_medium.csv")
    real_share = hc.real_coverage_shares(manifest)

    cards, sens, per_bin, cls_rows, prec = {}, [], [], [], []
    for mode in ("assigned", "fullyear"):
        neg = hc.load_negatives(f"{args.main}/runs/t27_ranker/v0_{mode}_top1.csv")
        cards[mode] = {}
        for label, (kind, f) in systems(args.main, mode).items():
            col = "hand_answered" if kind == "hand" else "ranker_answered"
            fp = hc.negatives_fp_rate(neg, col, mode, manifest)
            card = hc.compute_card(f, mode, manifest, hc.DEFAULT_POPULATION, fp)
            card["n_neg"] = int(len(neg))
            card["n_neg_answered"] = int(neg[col].sum())
            cards[mode][label] = card
            v = card["values"]
            prec.append(dict(mode=mode, system=label, S=v["R2_success_pop"], A=v["answer_rate_pop"], FP=fp,
                             n_neg=len(neg), k_neg=int(neg[col].sum()),
                             fp_upper95=1 - 0.05 ** (1 / len(neg)) if neg[col].sum() == 0 else np.nan))
            # population sensitivity (R2, R3)
            for pop in ["mcq14_like", "loguniform", "smarts_uniform", "none"]:
                c = hc.compute_card(f, mode, manifest, pop, fp, n_boot=0)["values"]
                sens.append(dict(mode=mode, system=label, population=pop, success=c["R2_success_pop"],
                                 correct=c["R3_correct_pop"], answer_rate=c["answer_rate_pop"]))
            # per period bin
            w_cov = hc.coverage_weights(f, manifest) if mode == "assigned" else np.ones(len(f))
            for bn in BINS:
                sel = (f.period_bin == bn).to_numpy()
                g = f[sel]
                per_bin.append(dict(mode=mode, system=label, bin=bn, n=int(sel.sum()),
                                    success=g.success.mean(), correct=g.correct.mean(),
                                    success_covw=float(np.sum(w_cov[sel] * g.success) / w_cov[sel].sum())))
            # coverage classes (assigned)
            if mode == "assigned":
                for P30 in (True, False):
                    for cl in CLASSES:
                        g = f[(f.coverage_class == cl) & ((f.true_period <= 30) == P30)]
                        k = int(g.success.sum())
                        lo, hi = wilson(k, len(g))
                        cls_rows.append(dict(system=label, P_le_30=P30, coverage_class=cl, n=len(g),
                                             success=g.success.mean(), success_lo=lo, success_hi=hi,
                                             correct=g.correct.mean(), answered=g.answered.mean(),
                                             within20_answer=(g.within20 & g.answered).mean(),
                                             real_share=float(real_share.get(cl, np.nan))))
    shares = pd.DataFrame({"bin": BINS,
                           "dev_medium": manifest.true_period.map(hc.period_bin).value_counts(normalize=True)
                           .reindex(BINS).to_numpy(),
                           **{p: [hc.population_bin_shares(p)[b] for b in BINS] for p in hc.POPULATIONS}})

    # P29 by period bin inside consecutive-3-5 (assigned)
    p29 = []
    for label, (kind, f) in systems(args.main, "assigned").items():
        g = f[f.coverage_class == "consecutive-3-5"]
        for bn in BINS + ["P<=30 d", "all"]:
            h = g if bn == "all" else g[g.true_period <= 30] if bn == "P<=30 d" else g[g.period_bin == bn]
            lo, hi = wilson(int(h.success.sum()), len(h))
            p29.append(dict(system=label, bin=bn, n=len(h), success=h.success.mean(), lo=lo, hi=hi,
                            correct=h.correct.mean()))

    # P29 cost estimate for a true 70 d window, from measured per-star wall seconds (assigned T5 baseline)
    ps = pd.read_csv(REPO / "benchmarks/baselines/dev_medium_assigned_t5_tables/per_star.csv")
    sec = ps.groupby("coverage_class").seconds.mean()
    span = ps.groupby("coverage_class")[["total_baseline_days", "longest_contiguous_days", "acf_reach"]].median()
    cost = dict(sec_per_star_consecutive2=float(sec["consecutive-2"]),
                sec_per_star_consecutive35=float(sec["consecutive-3-5"]),
                minutes_1000_stars_4_workers=float(sec[["consecutive-2", "consecutive-3-5"]].mean() * 1000 / N_WORKERS / 60),
                median_span_days=span.to_dict(orient="index"))

    with open(out / "cards.json", "w") as fh:
        json.dump(cards, fh, indent=1)
    for mode in cards:
        (out / f"card_{mode}.md").write_text(hc.card_markdown(cards[mode], f"Headline card, dev_medium {mode}, "
                                                                          f"population `{hc.DEFAULT_POPULATION}`"))
    pd.DataFrame(sens).to_csv(out / "population_sensitivity.csv", index=False, float_format="%.4f")
    pd.DataFrame(per_bin).to_csv(out / "per_bin.csv", index=False, float_format="%.4f")
    pd.DataFrame(cls_rows).to_csv(out / "coverage_class.csv", index=False, float_format="%.4f")
    pd.DataFrame(prec).to_csv(out / "precision_inputs.csv", index=False, float_format="%.6f")
    pd.DataFrame(p29).to_csv(out / "p29_consecutive35.csv", index=False, float_format="%.4f")
    shares.to_csv(out / "bin_shares.csv", index=False, float_format="%.4f")
    with open(out / "p29_cost.json", "w") as fh:
        json.dump(cost, fh, indent=1)
    for mode in cards:
        print((out / f"card_{mode}.md").read_text())
    print(pd.DataFrame(sens).to_string())
    print(pd.DataFrame(p29).to_string())
    print(shares.to_string())
    print(json.dumps(cost, indent=1))
    print(pd.DataFrame(prec).to_string())


if __name__ == "__main__":
    main()
