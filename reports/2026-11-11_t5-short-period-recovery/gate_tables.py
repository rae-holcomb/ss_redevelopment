"""T5: which comb-fit gates reject fitted *success* candidates, by period bin (after the fix).

Usage: python reports/2026-11-11_t5-short-period-recovery/gate_tables.py
Writes data/gates_success_candidates.csv: per mode and period bin, the fraction of success
candidates (correct/diffrot, fitted, in reach) failing each gate, and the number of stars whose
only success candidates all fail half_lag_consistency.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
SUC = {"correct", "diffrot"}
GATES = ["n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window", "half_lag_consistency"]


def main() -> None:
    """Write the gate table."""
    rows = []
    for mode, run in (("fy", "t5_after_fullyear"), ("asg", "t5_after_assigned")):
        c = pd.read_csv(ROOT / "runs" / run / "per_candidate.csv")
        c = c[c.error.isna() & ~c.beyond_reach.astype(bool) & c.outcome.isin(SUC)]
        fg = c.failed_gates.fillna("").str.split(";")
        for g in GATES:
            c[f"fail_{g}"] = fg.apply(lambda x, g=g: g in x)
        c["fail_only_half_lag"] = fg.apply(lambda x: [y for y in x if y] == ["half_lag_consistency"])
        for b, gb in c.groupby("period_bin"):
            row = dict(mode=mode, bin=b, n_candidates=len(gb), n_stars=gb.star_id.nunique(),
                       passed=gb.passed_gates.astype(bool).mean(), only_half_lag=gb.fail_only_half_lag.mean())
            row.update({g: gb[f"fail_{g}"].mean() for g in GATES})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "gates_success_candidates.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
