#!/usr/bin/env python3
"""
build_feature_table.py -- T9 pass 1: candidate-level feature table on synthetic data.

For every star in synthetic.make_suite(N, seed=S):
  1. compute the gap-aware ACF;
  2. run the scorecard's default pipeline methods (pairwise_histogram,
     lombscargle, acf_fft; n_guesses=10 each) and pool their candidates;
  3. fit_rotation_period on the pool (same defaults as benchmarks/scorecard.py);
  4. extract_candidate_features (existing ~100 features);
  5. label every candidate row with evaluation.classify_period(fitted_P,
     P_eq, REL_TOL, period_range) -> correct/diffrot/half/double/wrong, and
     mark which candidate the current hand-tuned arbitration selected.

The fitted per-peak comb heights are kept (as JSON) so pass 2
(new_features.py) can compute prototype features without refitting.

Resumable: one pickle per star under --outdir/stars/, skipped if present.

Usage (repo root):
    /opt/anaconda3/envs/ss_local/bin/python reports/2026-10-14_t9-feature-search/build_feature_table.py \
        --n 200 --seed 0 --n-workers 2 --outdir runs/t9_feature_table
"""

from __future__ import annotations

import os

# Pin BLAS threads to 1 per worker BEFORE numpy is imported (CLAUDE.md).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))

import evaluation as ev  # noqa: E402
import guesses  # noqa: E402
from acf_utils import compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402
from ml_features import extract_candidate_features  # noqa: E402

PIPELINE_METHODS = ("pairwise_histogram", "lombscargle", "acf_fft")
N_GUESSES = 10


def process_one_star(lc, outdir: Path) -> str:
    """Fit one synthetic star and pickle its labelled candidate table.

    Module-level (picklable) worker for ProcessPoolExecutor.

    Parameters
    ----------
    lc : synthetic.SyntheticLightCurve.
    outdir : directory; writes outdir/stars/<star_id>.pkl.

    Returns
    -------
    str : status line for logging.
    """
    warnings.simplefilter("ignore")
    out_path = outdir / "stars" / f"{lc.star_id}.pkl"
    t_start = time.perf_counter()
    prange = lc.params.get("period_range")
    star_meta = dict(
        star_id=lc.star_id, true_period=lc.true_period, period_bin=ev.period_bin(lc.true_period),
        period_lo=prange[0] if prange else np.nan, period_hi=prange[1] if prange else np.nan,
        snr=lc.params["snr"], shear=lc.params["shear"], inclination_deg=lc.params["inclination_deg"],
        n_spots=lc.params["n_spots"], spot_lifetime_days=lc.params["spot_lifetime_days"],
        red_noise_sigma=lc.params["red_noise_sigma"], ramp_amplitude=lc.params["ramp_amplitude"],
    )
    try:
        lags, acf = compute_acf(lc.time, lc.flux)
        pooled = []
        for name in PIPELINE_METHODS:
            fn = getattr(guesses, f"guess_{name}")
            try:
                pooled.extend(fn(lc.time, lc.flux, lags, acf, n_guesses=N_GUESSES))
            except Exception:  # noqa: BLE001 -- same tolerance as the scorecard
                pass
        res = fit_rotation_period(lags, acf, pooled)
        df = extract_candidate_features(res, pooled, lc.time, lc.flux, lags, acf, star_id=lc.star_id)
        final_P = float(res.best_fit.P) if res.success else None
        final_outcome = ev.classify_period(final_P, lc.true_period, ev.REL_TOL, prange)
        if len(df):
            df["outcome"] = [ev.classify_period(p, lc.true_period, ev.REL_TOL, prange)
                             for p in df["fitted_P"]]
            df["cand_index"] = np.arange(len(df))
            df["selected"] = (df["cand_index"] == 0) & bool(res.success)
            df["fitted_t0"] = [c.fit.t0 for c in res.candidates]
            df["peak_ns_json"] = [json.dumps([int(n) for n in sorted(c.fit.per_peak)])
                                  for c in res.candidates]
            df["peak_heights_json"] = [json.dumps([float(c.fit.per_peak[n]["height"])
                                                   for n in sorted(c.fit.per_peak)])
                                       for c in res.candidates]
        for k, v in star_meta.items():
            if k != "star_id":
                df[k] = v
        df["pipeline_success"] = bool(res.success)
        df["final_outcome"] = final_outcome
        df["n_pooled"] = len(pooled)
        df["star_seconds"] = time.perf_counter() - t_start
        # stars with zero fitted candidates still get a stub row so decline
        # analysis can count them
        if len(df) == 0:
            df = pd.DataFrame([dict(star_meta, pipeline_success=False, final_outcome=final_outcome,
                                    n_pooled=len(pooled), star_seconds=time.perf_counter() - t_start,
                                    outcome=None)])
        df.to_pickle(out_path)
        return f"{lc.star_id}: {len(df)} rows, final={final_outcome}, {time.perf_counter() - t_start:.1f}s"
    except Exception as exc:  # noqa: BLE001
        (outdir / "stars" / f"{lc.star_id}.err").write_text(traceback.format_exc())
        return f"{lc.star_id}: ERROR {type(exc).__name__}: {exc}"


def main(argv=None):
    """CLI entry point: build (or resume) the candidate table, then combine."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--outdir", default="runs/t9_feature_table")
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip stars already pickled (default on)")
    args = p.parse_args(argv)

    from synthetic import make_suite
    outdir = Path(args.outdir)
    (outdir / "stars").mkdir(parents=True, exist_ok=True)
    suite = make_suite(args.n, seed=args.seed)
    todo = [lc for lc in suite if not (outdir / "stars" / f"{lc.star_id}.pkl").exists()]
    print(f"{len(suite)} stars, {len(todo)} to do, {args.n_workers} workers", flush=True)

    worker = partial(process_one_star, outdir=outdir)
    if args.n_workers == 1:
        for lc in todo:
            print(worker(lc), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futs = [ex.submit(worker, lc) for lc in todo]
            for f in as_completed(futs):
                print(f.result(), flush=True)

    frames = [pd.read_pickle(outdir / "stars" / f"{lc.star_id}.pkl") for lc in suite
              if (outdir / "stars" / f"{lc.star_id}.pkl").exists()]
    table = pd.concat(frames, ignore_index=True)
    table.to_pickle(outdir / "candidates_pass1.pkl")
    print(f"wrote {outdir / 'candidates_pass1.pkl'}: {table.shape}")


if __name__ == "__main__":
    main()
