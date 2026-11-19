#!/usr/bin/env python
"""Stage-2-only re-scoring harness (T32): cache Stage 1 once, re-run the comb fit per variant.

The scorecard runs every guess_* method (wavelet included) on every star, so
scoring a comb-fit variant costs ~7 min on dev_medium even though only Stage 2
changes. This harness splits the work:

  cache  -- per star: load the light curve exactly as scorecard._load does,
            compute both ACF views, run ONLY the pooled methods
            (scorecard.DEFAULT_PIPELINE_METHODS, same order and n_guesses),
            and pickle (lags, acf, counts, pooled guesses, truth) to
            <cache>/<star_id>.pkl (atomic write; resumable).
  fit    -- per star: load the pickle, run fit_rotation_period(..., comb_variant=V)
            and score it with the scorecard's own classify / gate-descriptor
            code; writes <out>.csv (one row per star) and <out>_cands.csv.

Every input to fit_rotation_period is the same as in score_one_star, so
`fit` reproduces a scorecard's per-star final_period / final_outcome for the
same variant (checked per star in reports/2026-11-19_t32-clean-rerun/).
Screening only: official numbers still come from benchmarks/scorecard.py.

Usage:
  python tools/stage2_harness.py cache --manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev \
      [--ignore-manifest-sectors] --cache runs/t32_clean/cache_fy --n-workers 3
  python tools/stage2_harness.py fit --cache runs/t32_clean/cache_fy --variant local --out runs/t32_clean/s2/local_fy
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scorecard as sc  # noqa: E402  (pins BLAS too; imports the pipeline exactly as the scorecard does)


def cache_one(task: dict, cache_dir: str) -> str:
    """Run Stage 1 (pooled methods only) for one task and pickle the Stage-2 inputs. Returns star_id."""
    warnings.simplefilter("ignore")
    out = Path(cache_dir) / f"{task['star_id']}.pkl"
    if out.exists():
        return task["star_id"]
    star_id, t, f, true_p, prange, meta, sectors = sc._load(task)
    lags, acf, counts = sc.compute_acf(t, f, return_counts=True)
    lags1, acf1 = sc.compute_acf(t, f, max_lag_frac=sc.STAGE1_MAX_LAG_FRAC)
    pooled, errors = [], {}
    for name, fn in sc.discover_guess_functions().items():
        if name not in sc.DEFAULT_PIPELINE_METHODS:
            continue
        try:
            pooled.extend(sc.two_view_guesses(fn, t, f, (lags1, acf1), (lags, acf),
                                              n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT)))
        except Exception as exc:  # noqa: BLE001 -- same accounting as score_one_star
            errors[name] = f"{type(exc).__name__}: {exc}"
    rec = dict(star_id=star_id, true_period=true_p, prange=prange, lags=lags, acf=acf, counts=counts,
               pooled=pooled, errors=errors, period_bin=sc.ev.period_bin(true_p), time=t, flux=f)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return star_id


def fit_one(path: str, variant: str, rel_tol: float, fit_kwargs: dict | None = None) -> tuple:
    """Stage 2 for one cached star under comb variant `variant` -> (star row, candidate rows)."""
    warnings.simplefilter("ignore")
    with open(path, "rb") as fh:
        rec = pickle.load(fh)
    true_p, prange = rec["true_period"], rec["prange"]
    t0 = time.perf_counter()
    res = sc.fit_rotation_period(rec["lags"], rec["acf"], rec["pooled"], acf_counts=rec["counts"],
                                 comb_variant=variant, **(fit_kwargs or {}))
    secs = time.perf_counter() - t0
    P = float(res.best_fit.P) if res.success else None
    g = sc._gate_descriptors(res, rec["lags"], rec["counts"], true_p, rel_tol, prange)
    cands = g.pop("candidates")
    row = dict(star_id=rec["star_id"], true_period=true_p, period_bin=rec["period_bin"], final_period=P,
               final_outcome=sc.ev.classify_period(P, true_p, rel_tol, prange), fit_seconds=secs,
               n_pooled=len(rec["pooled"]), **g)
    for c in cands:
        c["star_id"] = rec["star_id"]
    return row, cands


def main(argv=None):
    """CLI entry point (see module docstring)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--data-root", default=None)
    c.add_argument("--ignore-manifest-sectors", action="store_true")
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--n-workers", type=int, default=3)
    f = sub.add_parser("fit")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--variant", required=True)
    f.add_argument("--out", type=Path, required=True, help="output prefix (writes <out>.csv, <out>_cands.csv)")
    f.add_argument("--n-workers", type=int, default=3)
    f.add_argument("--rel-tol", type=float, default=sc.ev.REL_TOL)
    f.add_argument("--stars", default=None, help="comma-separated star_ids (default: all cached)")
    args = p.parse_args(argv)

    if args.cmd == "cache":
        if "_clean" not in args.manifest.name:
            raise SystemExit("develop on the noise-clean manifests only (datasets/*_clean.csv)")
        ns = argparse.Namespace(synthetic=None, manifest=args.manifest, data_root=args.data_root,
                                ignore_manifest_sectors=args.ignore_manifest_sectors, coverage=sc.cov.FULL,
                                coverage_seed=0, coverage_weighting="frequency", coverage_library=None,
                                noise_subset="all", allow_noise_excluded=False, noise_table=None)
        tasks = sc.build_tasks(ns)
        args.cache.mkdir(parents=True, exist_ok=True)
        todo = [t for t in tasks if not (args.cache / f"{t['star_id']}.pkl").exists()]
        print(f"{len(tasks) - len(todo)} cached, {len(todo)} to run")
        with ProcessPoolExecutor(max_workers=args.n_workers, initializer=determinism.init_worker) as ex:
            futs = [ex.submit(cache_one, t, str(args.cache)) for t in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        return

    paths = sorted(args.cache.glob("*.pkl"))
    if args.stars:
        keep = set(args.stars.split(","))
        paths = [q for q in paths if q.stem in keep]
    rows, cands = [], []
    t0 = time.time()
    if args.n_workers <= 1:
        for q in paths:
            r, cc = fit_one(str(q), args.variant, args.rel_tol)
            rows.append(r)
            cands.extend(cc)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers, initializer=determinism.init_worker) as ex:
            futs = [ex.submit(fit_one, str(q), args.variant, args.rel_tol) for q in paths]
            for fut in as_completed(futs):
                r, cc = fut.result()
                rows.append(r)
                cands.extend(cc)
    df = pd.DataFrame(rows).sort_values("star_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out}.csv", index=False)
    pd.DataFrame(cands).to_csv(f"{args.out}_cands.csv.gz", index=False)
    vc = df.final_outcome.value_counts(normalize=True)
    succ = vc.get("correct", 0) + vc.get("diffrot", 0)
    print(f"{args.variant}: n={len(df)} success {succ:.3f} wrong {vc.get('wrong', 0):.3f} "
          f"alias {vc.get('half', 0) + vc.get('double', 0):.3f} none {vc.get('none', 0):.3f} "
          f"median fit {df.fit_seconds.median():.3f}s  wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
