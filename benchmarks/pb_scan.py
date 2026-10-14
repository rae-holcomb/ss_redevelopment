#!/usr/bin/env python3
"""
pb_scan.py -- recovery vs P/baseline on short (1-2 sector) and long (13
sector) synthetic baselines, plus a pure-noise false-positive check.

Complements scorecard.py (which draws periods log-uniformly on a fixed
13-sector baseline and so almost never probes P/B > 0.1). Here the period is
set as an explicit fraction of the light-curve baseline B, so the ACF
lag-range / peak-count limit of the comb fit shows up directly.

Grid
----
    n_sectors in {1, 2, 13} (B ~ 27, 55, 356 d)
    P/B in FRACS (0.02 ... 0.40)
    --n-per-cell light curves per (n_sectors, P/B) cell: amplitude 0.01,
        white noise 0.001 (SNR 10), n_spots cycles 1..3, spot lifetime
        alternates infinite / 3P, solid-body rotation (shear 0).
    --n-noise noise-only light curves per n_sectors, half white
        (sigma 0.001), half white + AR(1) red noise (sigma 0.001, tau 2 d).

Outputs (in --outdir): rows.csv (one row per light curve) and summary.json
(outcome fractions per (n_sectors, P/B), noise false-positive rates,
runtime percentiles).

Usage
-----
    python benchmarks/pb_scan.py --outdir runs/pb_scan_after
    python benchmarks/pb_scan.py --outdir runs/pb_scan_quick --n-per-cell 3 --n-noise 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

FRACS = (0.02, 0.04, 0.07, 0.10, 0.13, 0.17, 0.20, 0.25, 0.30, 0.33, 0.40)
SECTORS = (1, 2, 13)
DEFAULT_N_WORKERS = 10


def run_one(task: dict) -> dict:
    """Generate one synthetic light curve, run the default pipeline
    (compute_acf -> gather_initial_guesses -> fit_rotation_period), and score
    it. Module-level so ProcessPoolExecutor can pickle it.

    Parameters
    ----------
    task : dict with keys kind ('signal' or 'noise'), n_sectors, frac (P/B,
        signal only), seed, red (noise only: add red noise).

    Returns
    -------
    dict : one result row.
    """
    warnings.simplefilter("ignore")
    from acf_utils import compute_acf
    try:
        from acf_utils import STAGE1_MAX_LAG_FRAC
    except ImportError:  # pre-2026-10-13 code: one ACF for both stages
        STAGE1_MAX_LAG_FRAC = None
    from comb_fit import fit_rotation_period
    from evaluation import classify_period
    from guesses import gather_initial_guesses
    from synthetic import make_lightcurve, tess_like_time_grid

    ns, seed = task["n_sectors"], task["seed"]
    t_grid, _, _ = tess_like_time_grid(n_sectors=ns)
    B = float(t_grid[-1] - t_grid[0])
    if task["kind"] == "signal":
        P = task["frac"] * B
        lc = make_lightcurve(P, amplitude=0.01, white_noise=0.001, n_sectors=ns, seed=seed,
                             n_spots=1 + seed % 3,
                             spot_lifetime_days=np.inf if seed % 2 == 0 else 3 * P)
    else:
        P = 1.0  # placeholder; no signal
        lc = make_lightcurve(P, amplitude=0.0, white_noise=0.001, n_sectors=ns, seed=seed,
                             red_noise_sigma=0.001 if task["red"] else 0.0, red_noise_tau_days=2.0)

    t0 = time.perf_counter()
    try:  # current code: pair-count-weighted comb fit
        lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
        fit_kw = {"acf_counts": counts}
    except TypeError:  # pre-2026-10-13 code (no pair counts)
        lags, acf = compute_acf(lc.time, lc.flux)
        fit_kw = {}
    fit_kw.update(task.get("fit_kwargs", {}))
    if STAGE1_MAX_LAG_FRAC is not None:
        lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    else:
        lags1, acf1 = lags, acf
    two = {"long_view": (lags, acf)} if STAGE1_MAX_LAG_FRAC is not None else {}
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags1, acf1, n_guesses=10, **two)
    t1 = time.perf_counter()
    res = fit_rotation_period(lags, acf, g, **fit_kw)
    t2 = time.perf_counter()

    Pm = float(res.best_fit.P) if res.success else None
    if task["kind"] == "signal":
        outcome = classify_period(Pm, P)
    else:
        outcome = "none" if Pm is None else "false_pos"
    best = res.candidates[0] if (res.success and res.candidates) else None
    return dict(
        kind=task["kind"], red=task.get("red", False), n_sectors=ns, baseline=B,
        P_over_B=task.get("frac", np.nan), true_period=P if task["kind"] == "signal" else np.nan,
        seed=seed, success=bool(res.success), measured_P=Pm, outcome=outcome,
        acf_lag_max=float(lags[-1]),
        n_peaks_used=(res.best_fit.n_peaks_used if res.success else np.nan),
        n_teeth_available=(getattr(best, "n_teeth_available", np.nan) if best else np.nan),
        few_peaks=(getattr(best, "few_peaks", np.nan) if best else np.nan),
        n_candidates_tried=res.n_candidates_tried,
        stage1_seconds=t1 - t0, fit_seconds=t2 - t1,
    )


def build_tasks(n_per_cell: int, n_noise: int, seed_offset: int = 0, fit_kwargs: dict = None) -> list:
    """Build the (deterministic) task list for the signal grid and the noise set.

    seed_offset shifts every seed (use a nonzero offset for tuning/ablation
    runs so the default-seed evaluation set stays untouched); fit_kwargs are
    passed through to fit_rotation_period.
    """
    fit_kwargs = fit_kwargs or {}
    tasks = []
    for ns in SECTORS:
        for fi, frac in enumerate(FRACS):
            for s in range(n_per_cell):
                tasks.append(dict(kind="signal", n_sectors=ns, frac=frac, fit_kwargs=fit_kwargs,
                                  seed=10000 * ns + 100 * fi + s + seed_offset))
        for s in range(n_noise):
            tasks.append(dict(kind="noise", n_sectors=ns, seed=900000 + 1000 * ns + s + seed_offset,
                              red=bool(s % 2), fit_kwargs=fit_kwargs))
    return tasks


def summarize(df: pd.DataFrame) -> dict:
    """Outcome fractions per (n_sectors, P/B), noise false-positive rates and runtime."""
    sig = df[df.kind == "signal"]
    cats = ["correct", "diffrot", "half", "double", "wrong", "none"]
    cells = {}
    for (ns, fr), g in sig.groupby(["n_sectors", "P_over_B"]):
        cells[f"{ns}s_{fr:.2f}"] = {c: float((g.outcome == c).mean()) for c in cats} | {"n": int(len(g))}
    noise = df[df.kind == "noise"]
    fp = {
        "overall": float((noise.outcome == "false_pos").mean()) if len(noise) else np.nan,
        "n": int(len(noise)),
        "by_sectors": {str(k): float((g.outcome == "false_pos").mean()) for k, g in noise.groupby("n_sectors")},
        "white": float((noise[~noise.red].outcome == "false_pos").mean()) if len(noise) else np.nan,
        "red": float((noise[noise.red].outcome == "false_pos").mean()) if len(noise) else np.nan,
    }
    target = sig[sig.P_over_B <= 0.33 + 1e-9]
    overall = {
        "correct_all": float((sig.outcome == "correct").mean()),
        "correct_PB_le_0.33": float((target.outcome == "correct").mean()),
        "correct_PB_le_0.07": float((sig[sig.P_over_B <= 0.07 + 1e-9].outcome == "correct").mean()),
        "alias_all": float(sig.outcome.isin(["half", "double"]).mean()),
        "wrong_all": float((sig.outcome == "wrong").mean()),
    }
    rt = {}
    for ns, g in df.groupby("n_sectors"):
        tot = g.stage1_seconds + g.fit_seconds
        rt[str(ns)] = dict(median_total=float(tot.median()), p90_total=float(tot.quantile(0.9)),
                          median_fit=float(g.fit_seconds.median()), p90_fit=float(g.fit_seconds.quantile(0.9)))
    return dict(cells=cells, noise_false_positive=fp, overall=overall, runtime_seconds=rt)


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n-per-cell", type=int, default=12)
    p.add_argument("--n-noise", type=int, default=40, help="noise LCs per n_sectors value")
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="shift all seeds (tuning/ablation runs); keep 0 for the evaluation set")
    p.add_argument("--fit-kwargs", default="{}",
                   help='JSON dict passed to fit_rotation_period, e.g. \'{"alias_guard": false}\'')
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.n_per_cell, args.n_noise, args.seed_offset, json.loads(args.fit_kwargs))
    # longest tasks first for better load balance
    tasks.sort(key=lambda t: -t["n_sectors"])
    t0 = time.time()
    if args.n_workers <= 1:
        rows = [run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(args.n_workers) as ex:
            rows = list(ex.map(run_one, tasks, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "rows.csv", index=False)
    summ = summarize(df)
    summ["config"] = dict(n_per_cell=args.n_per_cell, n_noise=args.n_noise, fracs=list(FRACS),
                          seed_offset=args.seed_offset, fit_kwargs=json.loads(args.fit_kwargs),
                          sectors=list(SECTORS), wall_seconds=time.time() - t0,
                          code_root=str(ROOT))
    (args.outdir / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ["overall"], indent=1))
    print("noise FP:", summ["noise_false_positive"])
    print("wall seconds:", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
