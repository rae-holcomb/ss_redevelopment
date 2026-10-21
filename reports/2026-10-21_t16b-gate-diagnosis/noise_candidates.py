#!/usr/bin/env python3
"""
noise_candidates.py -- per-candidate gate records on pb_scan's noise-only
light curves (T16b), for offline false-positive estimates of relaxed gates.

Builds exactly benchmarks/pb_scan.py's noise task set (--n-noise per
n_sectors in {1, 2, 13}; half white, half white + AR(1) red noise), runs the
default pipeline, and writes one row per tested candidate with every gate's
value / threshold / pass (same columns as the scorecard's per_candidate.csv)
plus kind, red, n_sectors, seed and whether the pipeline answered at all.

Usage
-----
    python reports/2026-10-21_t16b-gate-diagnosis/noise_candidates.py \
        --outdir runs/t16b_noise --n-workers 2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import pandas as pd  # noqa: E402


def run_one(task: dict) -> list:
    """Run the default pipeline on one pb_scan noise light curve and return
    its per-candidate gate rows. Module-level so ProcessPoolExecutor can pickle it."""
    warnings.simplefilter("ignore")
    import numpy as np
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period
    from evaluation import REL_TOL
    from guesses import gather_initial_guesses
    from scorecard import _candidate_rows
    from synthetic import make_lightcurve

    lc = make_lightcurve(1.0, amplitude=0.0, white_noise=0.001, n_sectors=task["n_sectors"],
                         seed=task["seed"], red_noise_sigma=0.001 if task["red"] else 0.0,
                         red_noise_tau_days=2.0)
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags1, acf1, n_guesses=10, long_view=(lags, acf))
    res = fit_rotation_period(lags, acf, g, acf_counts=counts)
    # no true period: score against an impossible one so every candidate is "wrong"
    rows = _candidate_rows(res, lags, counts, 1e9, REL_TOL, None)
    for r in rows:
        r.update(kind="noise", red=bool(task["red"]), n_sectors=task["n_sectors"], seed=task["seed"],
                 pipeline_success=bool(res.success), acf_lag_max=float(lags[-1]),
                 decline_gates=";".join((res.decline_reason or {}).get("gates", [])))
        if r["outcome"] is not None:
            r["outcome"] = "noise"
    if not rows:
        rows = [dict(kind="noise", red=bool(task["red"]), n_sectors=task["n_sectors"], seed=task["seed"],
                     pipeline_success=False, acf_lag_max=float(lags[-1]))]
    return rows


def main(argv=None):
    """CLI entry point."""
    import pb_scan
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n-noise", type=int, default=40)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="pb_scan seed offset (0 = pb_scan's evaluation seeds; 1500 = T18's extra dev noise set)")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in pb_scan.build_tasks(0, args.n_noise, args.seed_offset) if t["kind"] == "noise"]
    tasks.sort(key=lambda t: -t["n_sectors"])
    with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
        rows = [r for rs in ex.map(run_one, tasks) for r in rs]
    out = args.outdir / "noise_candidates.csv"
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)
    print(f"wrote {out} ({len(rows)} rows, {len(tasks)} light curves)")


if __name__ == "__main__":
    main()
