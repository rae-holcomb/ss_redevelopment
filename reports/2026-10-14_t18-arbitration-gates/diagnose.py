"""diagnose.py -- T18 step 2: categorize every lost synth star and every noise
false positive from a diag_cache. Writes data/lost_<tag>.csv and data/fp_<tag>.csv.

Usage: python diagnose.py [pb cache tags ...]   (run from this folder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import arbitration as A

GATES = ["g_peaks", "g_pos", "g_snr", "g_vertex"]
FEATS = ["P", "redchi", "n_peaks_used", "n_peaks_required", "few_peaks", "frac_pos", "mean_hsnr",
         "frac_vertex", "mean_h", "acf_height_at_P", "acf_half_lag_contrast", "fold_half_corr",
         "fold_half_corr_at_2P", "pdm_theta", "n_methods"]


def lost_table(cache: str) -> pd.DataFrame:
    """One row per non-success signal star.

    category: never_proposed (no fitted candidate is a success),
    succ_failed_gate (the success candidate failing the fewest gates still
    fails one), or lost_on_selection (a success candidate passed every gate
    but another one was selected). Also records that success candidate's and
    the selected candidate's gate values / features.
    """
    stars, cands = A.load(cache)
    rows = []
    for s in stars.itertuples():
        if s.kind == "noise" or s.final_outcome in ("correct", "diffrot"):
            continue
        c = cands[cands.star_id == s.star_id]
        succ = c[c.label.isin(["correct", "diffrot"])]
        sel = c[c.selected]
        r = dict(star_id=s.star_id, P_true=s.true_period, bin=s.period_bin, snr=getattr(s, "snr", np.nan),
                 outcome=s.final_outcome, n_cands=len(c), n_succ_cands=len(succ))
        if len(succ) == 0:
            r["category"] = "never_proposed"
        else:
            nfail = (~succ[GATES].astype(bool)).sum(axis=1)
            b = succ.loc[nfail.idxmin()]
            r["category"] = "lost_on_selection" if b.passed else "succ_failed_gate"
            r["failed_gates"] = ",".join(g for g in GATES if not b[g])
            for k in FEATS:
                r["succ_" + k] = b[k]
        if len(sel):
            q = sel.iloc[0]
            for k in FEATS + ["label"]:
                r["sel_" + k] = q[k]
        rows.append(r)
    return pd.DataFrame(rows)


def fp_table(cache: str) -> pd.DataFrame:
    """One row per noise false positive: the selected candidate's gate values and features."""
    stars, cands = A.load(cache)
    rows = []
    for s in stars[stars.kind == "noise"].itertuples():
        if s.final_outcome != "false_pos":
            continue
        q = cands[(cands.star_id == s.star_id) & cands.selected].iloc[0]
        rows.append(dict(star_id=s.star_id, red=s.red, n_sectors=s.n_sectors, lag_max=s.lag_max,
                         **{k: q[k] for k in FEATS + ["n_teeth_available"]}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    out = Path("data")
    out.mkdir(exist_ok=True)
    L = lost_table("../../runs/t18_cache_s0")
    L.to_csv(out / "lost_s0.csv", index=False)
    print(L.groupby(["category", "outcome"]).size())
    print(L[L.category == "succ_failed_gate"].failed_gates.value_counts())
    for tag in sys.argv[1:]:
        F = fp_table(f"../../runs/t18_cache_{tag}")
        F.to_csv(out / f"fp_{tag}.csv", index=False)
        print(tag)
        print(F.round(3).to_string())
