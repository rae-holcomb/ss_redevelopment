"""Per-star before/after comparison for the T15 report.

"Before" per-star tables come from runs/synth200_s0_v2 in the main checkout:
the pre-T4 run on synthetic:n=200,seed=0. The committed baseline JSON has
aggregates only. T4 showed that its final outcomes are identical to v2, and
only pairwise_histogram's Stage 1 changed in T4. "After" is
runs/t15_after.
Writes data/stage1_diff_by_method.csv and data/final_before_after.csv.

Usage: python <report>/compare_runs.py <before_run_dir> <after_run_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Compare Stage-1 per-method rows and final answers between two runs."""
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(__file__).parent / "data"
    bm, am = pd.read_csv(before / "per_method.csv"), pd.read_csv(after / "per_method.csv")
    mm = bm.merge(am, on=["star_id", "method"], suffixes=("_before", "_after"))
    rows = []
    for meth, g in mm.groupby("method"):
        t1b, t1a = g.top1_before.to_numpy(float), g.top1_after.to_numpy(float)
        both = np.isfinite(t1b) & np.isfinite(t1a)
        rows.append(dict(
            method=meth, n=len(g),
            errors_before=int(g.error_before.notna().sum()), errors_after=int(g.error_after.notna().sum()),
            outcome_changed=int((g.outcome_before != g.outcome_after).sum()),
            n_candidates_changed=int((g.n_candidates_before != g.n_candidates_after).sum()),
            top1_max_rel_diff=float(np.max(np.abs(t1a[both] - t1b[both]) / t1b[both])) if both.any() else 0.0,
        ))
    pd.DataFrame(rows).to_csv(out / "stage1_diff_by_method.csv", index=False)

    bs, as_ = pd.read_csv(before / "per_star.csv"), pd.read_csv(after / "per_star.csv")
    keep = ["star_id", "true_period", "final_outcome", "final_period", "n_pooled"]
    m = bs[keep].merge(as_[keep], on=["star_id", "true_period"], suffixes=("_before", "_after"))
    m.to_csv(out / "final_before_after.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
