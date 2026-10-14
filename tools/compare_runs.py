#!/usr/bin/env python3
"""
compare_runs.py -- check whether two scorecard runs produced identical results.

Compares per_star.csv final periods/outcomes and per_method.csv per-method
top-1 candidates and candidate counts between two scorecard output
directories, star by star. Exits 0 if everything is bit-identical, 1
otherwise. Timing columns are ignored.

Usage
-----
    python tools/compare_runs.py runs/A runs/B
    python tools/compare_runs.py runs/A runs/B --verbose   # list differing stars
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise exact equality that treats NaN == NaN as equal."""
    a, b = a.to_numpy(), b.to_numpy()
    both_nan = pd.isna(a) & pd.isna(b)
    with np.errstate(invalid="ignore"):
        return both_nan | (a == b)


def compare(dir_a: Path, dir_b: Path) -> dict:
    """Compare two scorecard runs.

    Returns
    -------
    dict with:
        n_stars : stars in common
        final_period_diff : DataFrame of stars whose final_period differs
            (columns final_period_a, final_period_b, true_period, rel_diff)
        final_outcome_diff : number of stars whose final_outcome differs
        method_top1_diff : {method: n stars whose top1 differs}
        method_ncand_diff : {method: n stars whose n_candidates differs}
    """
    sa = pd.read_csv(dir_a / "per_star.csv").set_index("star_id").sort_index()
    sb = pd.read_csv(dir_b / "per_star.csv").set_index("star_id").sort_index()
    common = sa.index.intersection(sb.index)
    sa, sb = sa.loc[common], sb.loc[common]
    same_p = _same(sa["final_period"], sb["final_period"])
    diff = pd.DataFrame({
        "final_period_a": sa["final_period"], "final_period_b": sb["final_period"],
        "true_period": sa["true_period"],
    })[~same_p]
    diff["rel_diff"] = (diff.final_period_a - diff.final_period_b).abs() / diff.final_period_a.abs()

    ma = pd.read_csv(dir_a / "per_method.csv").set_index(["star_id", "method"]).sort_index()
    mb = pd.read_csv(dir_b / "per_method.csv").set_index(["star_id", "method"]).sort_index()
    mcommon = ma.index.intersection(mb.index)
    ma, mb = ma.loc[mcommon], mb.loc[mcommon]
    top1 = ~_same(ma["top1"], mb["top1"])
    ncand = ~_same(ma["n_candidates"], mb["n_candidates"])
    methods = ma.index.get_level_values("method")
    return dict(
        n_stars=len(common),
        final_period_diff=diff,
        final_outcome_diff=int((sa["final_outcome"] != sb["final_outcome"]).sum()),
        method_top1_diff={m: int(top1[methods == m].sum()) for m in sorted(set(methods))},
        method_ncand_diff={m: int(ncand[methods == m].sum()) for m in sorted(set(methods))},
    )


def main(argv=None) -> int:
    """CLI entry point; returns the process exit code."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    r = compare(args.run_a, args.run_b)
    d = r["final_period_diff"]
    print(f"{r['n_stars']} stars in common")
    print(f"final_period differs: {len(d)} stars"
          + (f" (max rel diff {d.rel_diff.max():.3g})" if len(d) else ""))
    print(f"final_outcome differs: {r['final_outcome_diff']} stars")
    for m in r["method_top1_diff"]:
        print(f"  {m:22s} top1 differs: {r['method_top1_diff'][m]:3d}   "
              f"n_candidates differs: {r['method_ncand_diff'][m]:3d}")
    if args.verbose and len(d):
        print(d.to_string())
    identical = (len(d) == 0 and r["final_outcome_diff"] == 0
                 and not any(r["method_top1_diff"].values()) and not any(r["method_ncand_diff"].values()))
    print("IDENTICAL" if identical else "DIFFERENT")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
