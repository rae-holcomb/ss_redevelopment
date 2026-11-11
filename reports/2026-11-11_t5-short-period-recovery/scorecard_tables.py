"""T5: before/after scorecard tables (final outcomes by bin, union recall, fitted-candidate ceiling).

Before = committed v5 baselines (benchmarks/baselines/dev_medium_{fullyear,assigned}_v5.json); pipeline
code is unchanged between that commit and this branch's parent, so they are the same code as main.
Fitted ceiling before = T10 dev_medium candidate tables (main checkout runs/t10_dev_medium_*; same
pipeline code); after = this branch's scorecard per_candidate.csv.

Usage: python reports/2026-11-11_t5-short-period-recovery/scorecard_tables.py
Writes data/scorecard_bins.csv and data/fitted_ceiling.csv.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
MAIN = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment")
SUC = {"correct", "diffrot"}
RUNS = {"fy": ("benchmarks/baselines/dev_medium_fullyear_v5.json", "runs/t5_after_fullyear", "fullyear"),
        "asg": ("benchmarks/baselines/dev_medium_assigned_v5.json", "runs/t5_after_assigned", "assigned")}


def main() -> None:
    """Write the two tables."""
    rows, ceil = [], []
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv").set_index("star_id")
    snr = pd.read_csv(OUT / "t26_snr_rot.csv").set_index("star_id")
    for mode, (base, after, t10) in RUNS.items():
        for when, path in (("before", ROOT / base), ("after", ROOT / after / "metrics.json")):
            m = json.load(open(path))
            for b, fr in m["final"]["by_bin"].items():
                rows.append(dict(mode=mode, when=when, bin=b, union_recall=m["union_recall"]["by_bin"][b], **fr))
            rows.append(dict(mode=mode, when=when, bin="all", union_recall=m["union_recall"]["overall"],
                             **m["final"]["overall"]))
        c0 = pd.read_csv(MAIN / f"runs/t10_dev_medium_{t10}/candidates.csv.gz",
                         usecols=["star_id", "fit_error", "beyond_reach", "outcome"])
        c0 = c0[~c0.fit_error.astype(bool) & ~c0.beyond_reach.astype(bool)]
        c1 = pd.read_csv(ROOT / after / "per_candidate.csv", usecols=["star_id", "error", "beyond_reach", "outcome"])
        c1 = c1[c1.error.isna() & ~c1.beyond_reach.astype(bool)]
        for when, c in (("before", c0), ("after", c1)):
            s = c.assign(s=c.outcome.isin(SUC)).groupby("star_id").s.any().reindex(man.index, fill_value=False)
            for sid, v in s.items():
                ceil.append(dict(star_id=sid, mode=mode, when=when, eval_pbin=man.at[sid, "eval_pbin"],
                                 detectable=snr.at[sid, f"snr_rot_{mode}"] >= 5, fitted_success=bool(v)))
    pd.DataFrame(rows).to_csv(OUT / "scorecard_bins.csv", index=False)
    pd.DataFrame(ceil).to_csv(OUT / "fitted_ceiling.csv", index=False)
    print("ok")


if __name__ == "__main__":
    main()
