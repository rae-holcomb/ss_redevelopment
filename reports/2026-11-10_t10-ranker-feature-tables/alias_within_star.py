"""alias_within_star.py -- paired, within-star version of the diffrot alias-feature check.

For stars that have both a fitted `correct` and a fitted `diffrot` candidate, compare the
feature at the best (lowest-redchi) diffrot candidate with the best correct one on the same
light curve. Removes star-level confounds (amplitude, coverage, band width). Also reports the
half_lag_consistency gate pass rate for correct vs diffrot rows.

Usage: python <this> runs/t10_dmfy_fitted.csv.gz smarts_dev_medium_fullyear
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FEATS = ["fold_half_corr", "fold_half_corr_at_2P", "acf_half_lag_contrast", "acf_odd_even_contrast",
         "acf_height_at_P", "redchi_ratio_to_global_best"]


def paired(c: pd.DataFrame) -> pd.DataFrame:
    """Per feature: n stars, median paired difference (diffrot - correct), fraction diffrot lower."""
    best = (c[c.outcome.isin(["correct", "diffrot"])].sort_values("redchi")
            .groupby(["star_id", "outcome"]).first().unstack("outcome"))
    rows = []
    for f in FEATS:
        d = (best[(f, "diffrot")] - best[(f, "correct")]).dropna()
        rows.append(dict(feature=f, n_stars=len(d), median_diff=float(d.median()),
                         frac_diffrot_lower=float((d < 0).mean())))
    return pd.DataFrame(rows)


def gate_rates(c: pd.DataFrame) -> pd.DataFrame:
    """half_lag_consistency (and all-gates) pass rate per outcome."""
    g = c.groupby("outcome")
    return pd.DataFrame(dict(n=g.size(), pass_half_lag=g.gate_half_lag_consistency_pass.mean(),
                             pass_all=g.passed_gates.mean()))


if __name__ == "__main__":
    c = pd.read_csv(sys.argv[1], low_memory=False)
    p, gr = paired(c), gate_rates(c)
    p.to_csv(HERE / "data" / f"{sys.argv[2]}_alias_within_star.csv", index=False)
    gr.to_csv(HERE / "data" / f"{sys.argv[2]}_gate_by_outcome.csv")
    print(p.round(3).to_string(index=False)); print(gr.round(3).to_string())
