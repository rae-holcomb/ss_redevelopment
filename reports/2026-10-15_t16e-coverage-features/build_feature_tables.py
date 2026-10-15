#!/usr/bin/env python3
"""
build_feature_tables.py -- T16e: labelled candidate feature tables per coverage class.

For each coverage spec, the same synthetic stars (make_suite(N, seed=S)) are
cut to a pattern drawn by coverage.sample_patterns(spec, N, seed=coverage_seed)
(exactly what `scorecard.py --synthetic N --seed S --coverage spec
--coverage-seed C` does). For every star:
  1. ACF for the comb fit (~0.7 B, with pair counts) and the Stage-1 view (B/3);
  2. the scorecard's pipeline methods (pairwise_histogram, lombscargle,
     acf_fft; n_guesses=10) through guesses.two_view_guesses, pooled;
  3. fit_rotation_period(lags, acf, pooled, acf_counts=counts) (default gates);
  4. extract_candidate_features (incl. the new T16e coverage features);
  5. each candidate labelled with evaluation.classify_period(fitted_P, P_eq,
     REL_TOL, period_range) -> correct / diffrot / half / double / wrong.

Resumable: one pickle per star under <outdir>/<spec>/stars/, written atomically.

Usage (repo root):
    /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-10-15_t16e-coverage-features/build_feature_tables.py \\
        --n 200 --seed 0 --coverage-seed 0 --n-workers 2 --outdir runs/t16e_features \\
        --coverage full single consecutive-2 gapped-2 3-5 real-weighted
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))

import determinism  # noqa: E402

determinism.pin_blas_threads()  # before numpy (CLAUDE.md, T14)

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import guesses  # noqa: E402
import ml_features as mf  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402

PIPELINE_METHODS = ("pairwise_histogram", "lombscargle", "acf_fft")
N_GUESSES = 10


def process_one_star(lc, outdir: Path, spec: str) -> str:
    """Fit one synthetic star and pickle its labelled candidate table.

    Module-level (picklable) worker for ProcessPoolExecutor.

    Parameters
    ----------
    lc : synthetic.SyntheticLightCurve (already cut to its coverage pattern).
    outdir : the spec's directory; writes outdir/stars/<star_id>.pkl.
    spec : coverage spec name (recorded in every row).

    Returns
    -------
    str : status line.
    """
    warnings.simplefilter("ignore")
    out_path = outdir / "stars" / f"{lc.star_id}.pkl"
    t_start = time.perf_counter()
    prange = lc.params.get("period_range")
    pat = lc.params.get("sectors")
    star_meta = dict(
        star_id=lc.star_id, coverage_spec=spec,
        pattern=cov.format_pattern(pat) if pat else "full",
        coverage_class=cov.coverage_class(pat) if pat else "full",
        true_period=lc.true_period, period_bin=ev.period_bin(lc.true_period),
        period_lo=prange[0] if prange else np.nan, period_hi=prange[1] if prange else np.nan,
        snr=lc.params["snr"], shear=lc.params["shear"], ramp_amplitude=lc.params["ramp_amplitude"],
    )
    try:
        lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
        lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
        pooled = []
        for name in PIPELINE_METHODS:
            fn = getattr(guesses, f"guess_{name}")
            try:
                pooled.extend(guesses.two_view_guesses(fn, lc.time, lc.flux, (lags1, acf1), (lags, acf),
                                                       n_guesses=N_GUESSES))
            except Exception:  # noqa: BLE001 -- same tolerance as the scorecard
                pass
        res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
        t_feat = time.perf_counter()
        df = mf.extract_candidate_features(res, pooled, lc.time, lc.flux, lags, acf, star_id=lc.star_id)
        feat_seconds = time.perf_counter() - t_feat
        # cost of the new T16e features alone (star part once + per-candidate part)
        t_cov = time.perf_counter()
        star_cov, seg = mf._star_coverage_features(lc.time, lc.flux)
        for c in res.candidates:
            mf._candidate_coverage_features(c.fit.P, star_cov, seg)
        cov_seconds = time.perf_counter() - t_cov

        final_P = float(res.best_fit.P) if res.success else None
        final_outcome = ev.classify_period(final_P, lc.true_period, ev.REL_TOL, prange)
        if len(df):
            df["outcome"] = [ev.classify_period(p, lc.true_period, ev.REL_TOL, prange) for p in df["fitted_P"]]
            df["cand_index"] = np.arange(len(df))
            df["selected"] = (df["cand_index"] == 0) & bool(res.success)
            df["passed_base_gates"] = [bool(getattr(c, "passed_base_gates", c.passed_gates))
                                       for c in res.candidates]
        else:
            df = pd.DataFrame([dict(outcome=None)])
        for k, v in star_meta.items():
            df[k] = v
        df["pipeline_success"] = bool(res.success)
        df["final_outcome"] = final_outcome
        df["n_candidates"] = len(res.candidates)
        df["feature_seconds"] = feat_seconds
        df["coverage_feature_seconds"] = cov_seconds
        df["star_seconds"] = time.perf_counter() - t_start
        tmp = out_path.with_suffix(".tmp")
        df.to_pickle(tmp)
        os.replace(tmp, out_path)
        return f"{spec} {lc.star_id}: {len(res.candidates)} cands, final={final_outcome}, " \
               f"{time.perf_counter() - t_start:.1f}s"
    except Exception as exc:  # noqa: BLE001
        (outdir / "stars" / f"{lc.star_id}.err").write_text(traceback.format_exc())
        return f"{spec} {lc.star_id}: ERROR {type(exc).__name__}: {exc}"


def main(argv=None):
    """CLI entry point: build (or resume) one table per coverage spec, then combine."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coverage-seed", type=int, default=0)
    p.add_argument("--coverage", nargs="+",
                   default=["full", "single", "consecutive-2", "gapped-2", "3-5", "real-weighted"])
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--outdir", default="runs/t16e_features")
    args = p.parse_args(argv)

    from synthetic import make_suite
    root = Path(args.outdir)
    for spec in args.coverage:
        outdir = root / spec
        (outdir / "stars").mkdir(parents=True, exist_ok=True)
        pats = cov.sample_patterns(spec, args.n, seed=args.coverage_seed)
        sectors = None if all(pt is None for pt in pats) else pats
        suite = make_suite(args.n, seed=args.seed, sectors=sectors)
        todo = [lc for lc in suite if not (outdir / "stars" / f"{lc.star_id}.pkl").exists()]
        print(f"[{spec}] {len(suite)} stars, {len(todo)} to do, {args.n_workers} workers", flush=True)
        worker = partial(process_one_star, outdir=outdir, spec=spec)
        if args.n_workers == 1:
            for lc in todo:
                print(worker(lc), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.n_workers, initializer=determinism.init_worker) as ex:
                for f in as_completed([ex.submit(worker, lc) for lc in todo]):
                    print(f.result(), flush=True)

    frames = []
    for spec in args.coverage:
        sd = root / spec / "stars"
        frames += [pd.read_pickle(f) for f in sorted(sd.glob("*.pkl"))]
    table = pd.concat(frames, ignore_index=True)
    table.to_pickle(root / "candidates.pkl")
    print(f"wrote {root / 'candidates.pkl'}: {table.shape}")


if __name__ == "__main__":
    main()
