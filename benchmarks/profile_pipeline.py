#!/usr/bin/env python3
"""profile_pipeline.py -- per-stage timing / cProfile harness for the per-star pipeline (T12).

Times exactly the steps ``benchmarks/scorecard.score_one_star`` runs (and, with
``--features``, the ``ml_features`` step of ``tools/build_feature_tables.build_one_star``),
one stage at a time, serially in this process. Nothing is scored here; the point is
where the wall time goes.

Stages (columns of the output CSV, seconds):
    load            scorecard._load (FITS read + preprocessing, or synthetic unpack)
    descriptors     coverage + searchability descriptor columns (incl. evaluation.acf_reach)
    acf_long        compute_acf(..., return_counts=True)        (Stage 2 view, ~0.7 B)
    acf_short       compute_acf(..., max_lag_frac=1/3)           (Stage 1 view)
    guess_<name>    two_view_guesses for every discovered guess_* method
    fit             comb_fit.fit_rotation_period on the pooled default methods
    gate_records    scorecard._gate_descriptors (per-candidate bookkeeping)
    features        build_feature_tables.candidate_table (ml_features; only with --features)

Usage
-----
    python benchmarks/profile_pipeline.py --manifest datasets/dev_medium.csv --data-root data/smarts_dev \
        --star-ids a,b,c [--ignore-manifest-sectors] --repeats 3 --out runs/profile/times.csv
    python benchmarks/profile_pipeline.py --synthetic 20 --seed 0 --out runs/profile/synth.csv
    ... --cprofile runs/profile/prof.txt     # cProfile of one pass over the stars, top 60 by tottime/cumtime

Columns of --out: star_id, mode, repeat, stage seconds..., total. Run it with the machine idle.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import pandas as pd  # noqa: E402

import scorecard as sc  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402
from guesses import two_view_guesses  # noqa: E402


def time_one_star(task: dict, features: bool = False, pipeline_methods=sc.DEFAULT_PIPELINE_METHODS) -> dict:
    """Run the scorecard's per-star steps on one task, timing each stage.

    Parameters
    ----------
    task : a scorecard task dict (from scorecard.build_tasks).
    features : also time build_feature_tables.candidate_table (ml_features).
    pipeline_methods : methods pooled into the fit.

    Returns
    -------
    dict : {stage: seconds, ..., "total": seconds, "n_pooled": int, "n_fit": int, "final_period": float|None}.
    """
    warnings.simplefilter("ignore")
    out = {}
    clock = time.perf_counter

    def tick(name, t0):
        """Record clock() - t0 under `name` and return a fresh start time."""
        now = clock()
        out[name] = now - t0
        return now

    t_all = t0 = clock()
    star_id, t, f, true_p, prange, meta, sectors = sc._load(task)
    t0 = tick("load", t0)
    sc._coverage_columns(t, f, sectors)
    sc._searchability_columns(t, f, true_p, prange)
    t0 = tick("descriptors", t0)
    lags, acf, counts = compute_acf(t, f, return_counts=True)
    t0 = tick("acf_long", t0)
    lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    t0 = tick("acf_short", t0)
    pooled, all_guesses = [], []
    for name, fn in sc.discover_guess_functions().items():
        try:
            flux_in = sc._gap_filled(f) if name == "wavelet" else f
            g = two_view_guesses(fn, t, flux_in, (lags1, acf1), (lags, acf),
                                 n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT))
        except Exception:  # noqa: BLE001 -- same policy as the scorecard
            g = []
        all_guesses.extend(g)
        if name in pipeline_methods:
            pooled.extend(g)
        t0 = tick(f"guess_{name}", t0)
    res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
    t0 = tick("fit", t0)
    sc._gate_descriptors(res, lags, counts, true_p, sc.ev.REL_TOL, prange)
    t0 = tick("gate_records", t0)
    if features:
        import build_feature_tables as bft
        bft.candidate_table(res, pooled, all_guesses, t, f, lags, acf, counts, star_id, true_p, prange,
                            sc.ev.REL_TOL)
        t0 = tick("features", t0)
    out["total"] = clock() - t_all
    out["n_pooled"] = len(pooled)
    out["n_fit"] = len(res.candidates)
    out["final_period"] = float(res.best_fit.P) if res.success else None
    return out


def build_tasks(args) -> list:
    """Scorecard tasks for the requested dataset, optionally restricted to --star-ids."""
    tasks = sc.build_tasks(args)
    if args.star_ids:
        keep = [s.strip() for s in args.star_ids.split(",") if s.strip()]
        by_id = {t["star_id"]: t for t in tasks}
        tasks = [by_id[s] for s in keep if s in by_id]
    return tasks


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int)
    src.add_argument("--manifest", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default=None)
    p.add_argument("--ignore-manifest-sectors", action="store_true")
    p.add_argument("--star-ids", default=None, help="comma-separated subset of star_ids (manifest order otherwise)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--features", action="store_true", help="also time the ml_features step")
    p.add_argument("--out", type=Path, default=None, help="per-star per-stage timings CSV")
    p.add_argument("--cprofile", type=Path, default=None, help="write cProfile top-N text here (one pass)")
    p.add_argument("--mode-label", default=None, help="value for the `mode` column (default fullyear/assigned)")
    args = p.parse_args(argv)
    args.coverage = sc.cov.FULL
    tasks = build_tasks(args)
    mode = args.mode_label or ("synthetic" if args.synthetic else
                               ("fullyear" if args.ignore_manifest_sectors else "assigned"))
    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        for t in tasks:
            time_one_star(t, features=args.features)
        pr.disable()
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        for key in ("tottime", "cumulative"):
            buf.write(f"===== sorted by {key} =====\n")
            pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(60)
        args.cprofile.write_text(buf.getvalue())
        print(f"wrote {args.cprofile}")
    rows = []
    for rep in range(args.repeats):
        for t in tasks:
            r = time_one_star(t, features=args.features)
            rows.append(dict(star_id=t["star_id"], mode=mode, repeat=rep, **r))
            print(f"[{rep}] {t['star_id']}: {r['total']:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    stage_cols = [c for c in df.columns if c not in ("star_id", "mode", "repeat", "n_pooled", "n_fit",
                                                     "final_period")]
    print(df[stage_cols].median().round(4).to_string())


if __name__ == "__main__":
    main()
