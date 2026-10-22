#!/usr/bin/env python3
"""
examples.py -- re-run the scorecard pipeline on a few chosen stars and save
what the example figures need (light curve, ACF, every fitted candidate's
comb parameters and gate values) to data/examples.npz + data/examples.json.

Stars are given as dataset:star_id, where dataset is a synthetic coverage
label (full, single, consecutive-2, gapped-2; synthetic n=200 seed 0,
coverage seed 0) or dev_medium (assigned coverage).

Usage
-----
    python reports/2026-10-21_t16b-gate-diagnosis/examples.py single:synth-s0-00012 ...
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402

DEV_ROOT = ROOT / "data" / "smarts_dev"


def _task(dataset: str, star_id: str, dev_root: Path) -> dict:
    """The scorecard task dict for one star of a named dataset."""
    import scorecard
    if dataset == "dev_medium":
        args = types.SimpleNamespace(synthetic=None, manifest=ROOT / "datasets/dev_medium.csv",
                                     data_root=str(dev_root), ignore_manifest_sectors=False,
                                     coverage="full", coverage_seed=0, coverage_weighting="frequency",
                                     coverage_library=None)
    else:
        args = types.SimpleNamespace(synthetic=200, seed=0, coverage=dataset, coverage_seed=0,
                                     coverage_weighting="frequency", coverage_library=None)
    for t in scorecard.build_tasks(args):
        if t["star_id"] == star_id:
            return t
    raise KeyError(f"{star_id} not in {dataset}")


def run_star(dataset: str, star_id: str, dev_root: Path) -> tuple:
    """Re-run the scorecard pipeline on one star; return (arrays, record)."""
    import scorecard
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period
    from evaluation import REL_TOL, classify_period
    from guesses import two_view_guesses

    star, t, f, true_p, prange, meta, sectors = scorecard._load(_task(dataset, star_id, dev_root))
    lags, acf, counts = compute_acf(t, f, return_counts=True)
    lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    fns = scorecard.discover_guess_functions()
    pooled = []
    for name, fn in fns.items():  # same pooling order as scorecard.score_one_star
        if name not in scorecard.DEFAULT_PIPELINE_METHODS:
            continue
        pooled += two_view_guesses(fn, t, f, (lags1, acf1), (lags, acf),
                                   n_guesses=scorecard.N_GUESSES.get(name, scorecard.N_GUESSES_DEFAULT))
    res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
    cands = []
    for r in res.candidates:
        fit = r.fit
        cands.append(dict(
            P_fit=fit.P, t0=fit.t0, redchi=fit.redchi, method=r.source_guess.method,
            outcome=classify_period(fit.P, true_p, REL_TOL, prange), passed=r.passed_gates,
            failed_gates=r.failed_gates, few_peaks=r.few_peaks, n_teeth_available=r.n_teeth_available,
            gates={g: {k: v for k, v in d.items()} for g, d in r.gate_results.items()},
            windows=[dict(n=int(w.n), lo=float(w.lag_lo), hi=float(w.lag_hi),
                          center=float(fit.per_peak[w.n]["center"]),
                          height=float(fit.per_peak[w.n]["height"]),
                          curvature=float(fit.per_peak[w.n]["curvature"]))
                     for w in fit.windows if w.n in fit.per_peak],
        ))
    rec = dict(dataset=dataset, star_id=star, true_period=true_p,
               period_range=list(prange) if prange else None, sectors=list(map(int, sectors or [])),
               success=res.success, final_P=(res.best_fit.P if res.success else None),
               message=res.message, decline_reason=res.decline_reason, candidates=cands)
    return dict(time=np.asarray(t), flux=np.asarray(f), lags=lags, acf=acf), rec


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stars", nargs="+", help="dataset:star_id")
    p.add_argument("--dev-root", type=Path, default=DEV_ROOT)
    p.add_argument("--out", type=Path, default=HERE / "data" / "examples")
    args = p.parse_args(argv)
    warnings.simplefilter("ignore")
    arrays, recs = {}, []
    for i, s in enumerate(args.stars):
        ds, sid = s.split(":", 1)
        arr, rec = run_star(ds, sid, args.dev_root)
        for k, v in arr.items():
            arrays[f"{i}_{k}"] = v
        recs.append(rec)
        print(f"{s}: success={rec['success']} P_true={rec['true_period']:.3f}")
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    args.out.with_suffix(".json").write_text(json.dumps(recs, indent=1, default=float))


if __name__ == "__main__":
    main()
