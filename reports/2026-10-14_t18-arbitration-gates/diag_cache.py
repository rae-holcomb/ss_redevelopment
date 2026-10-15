#!/usr/bin/env python3
"""
diag_cache.py -- T18 diagnostic cache: run the default pipeline on every star
of a benchmark (synth200 or pb_scan) and save EVERY fitted candidate with its
gate values plus the T9 alias features, PDM theta and cross-method support.

The point: candidate *fits* don't depend on the gating/selection rule, so with
this cache any arbitration rule can be replayed offline in milliseconds
(see arbitration.py). The final chosen rule is then re-verified with the real
scorecard / pb_scan runs.

Outputs (in --outdir)
---------------------
    cands.csv     one row per fitted candidate (star_id, gate values, features, label)
    stars.csv     one row per star (truth, period_range, pipeline outcome)
    acf/<star_id>.npz   lags, acf (for plotting / replaying the factor-2 guard)

Usage
-----
    python diag_cache.py --synthetic 200 --seed 0 --outdir runs/t18_cache_s0 --n-workers 2
    python diag_cache.py --pb --seed-offset 500 --n-per-cell 6 --n-noise 20 --outdir runs/t18_cache_pbdev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PIPELINE_METHODS = ("acf_fft", "lombscargle", "pairwise_histogram")  # scorecard pool order (alphabetical)


def _make_lc(task: dict):
    """Materialize (star_id, time, flux, true_P or None, period_range or None, meta) for a task."""
    if task["kind"] == "synthetic":
        lc = task["lc"]
        return lc.star_id, lc.time, lc.flux, lc.true_period, lc.params.get("period_range"), dict(
            snr=lc.params.get("snr"), kind="synthetic")
    from synthetic import make_lightcurve, tess_like_time_grid
    ns, seed = task["n_sectors"], task["seed"]
    t_grid, _, _ = tess_like_time_grid(n_sectors=ns)
    B = float(t_grid[-1] - t_grid[0])
    if task["kind"] == "signal":
        P = task["frac"] * B
        lc = make_lightcurve(P, amplitude=0.01, white_noise=0.001, n_sectors=ns, seed=seed,
                             n_spots=1 + seed % 3,
                             spot_lifetime_days=np.inf if seed % 2 == 0 else 3 * P)
        sid = f"pb-{ns}s-{task['frac']:.2f}-{seed}"
        return sid, lc.time, lc.flux, P, None, dict(kind="signal", n_sectors=ns, P_over_B=task["frac"])
    lc = make_lightcurve(1.0, amplitude=0.0, white_noise=0.001, n_sectors=ns, seed=seed,
                         red_noise_sigma=0.001 if task["red"] else 0.0, red_noise_tau_days=2.0)
    sid = f"pb-noise-{'red' if task['red'] else 'white'}-{ns}s-{seed}"
    return sid, lc.time, lc.flux, None, None, dict(kind="noise", n_sectors=ns, red=task["red"])


def process_one(task: dict, outdir: str, fit_kwargs: dict) -> dict:
    """Run Stage 1 + fit_rotation_period on one star; write its ACF npz and
    return {'star': {...}, 'cands': [{...}, ...]}. Module-level (picklable)."""
    warnings.simplefilter("ignore")
    import evaluation as ev
    import guesses as G
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period, phase_dispersion_stat
    from ml_features import _alias_discrimination_features

    sid, t, f, P_true, prange, meta = _make_lc(task)
    t_start = time.perf_counter()
    lags, acf, counts = compute_acf(t, f, return_counts=True)
    lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    pooled = []
    for name in PIPELINE_METHODS:
        fn = getattr(G, f"guess_{name}")
        try:
            pooled.extend(G.two_view_guesses(fn, t, f, (lags1, acf1), (lags, acf), n_guesses=10))
        except Exception:  # noqa: BLE001
            pass
    t1 = time.perf_counter()
    res = fit_rotation_period(lags, acf, pooled, acf_counts=counts, **fit_kwargs)
    t2 = time.perf_counter()
    np.savez_compressed(Path(outdir) / "acf" / f"{sid}.npz", lags=lags, acf=acf)

    def lab(P):
        if P_true is None:
            return "none" if P is None else "false_pos"
        return ev.classify_period(P, P_true, ev.REL_TOL, prange)

    Pm = float(res.best_fit.P) if res.success else None
    star = dict(star_id=sid, true_period=P_true, period_lo=prange[0] if prange else np.nan,
                period_hi=prange[1] if prange else np.nan,
                period_bin=ev.period_bin(P_true) if P_true else "noise",
                final_period=Pm, final_outcome=lab(Pm), n_pooled=len(pooled),
                lag_max=float(lags[-1]), stage1_seconds=t1 - t_start, fit_seconds=t2 - t1, **meta)
    rows = []
    sel = res.candidates[0] if res.success else None
    for i, c in enumerate(res.candidates):
        fit, d = c.fit, c.diagnostics
        hs = np.asarray(d["height_snr"], float)
        heights = np.array([fit.per_peak[n]["height"] for n in sorted(fit.per_peak)]) if fit.per_peak else np.array([])
        r = dict(
            star_id=sid, idx=i, selected=c is sel, P0=c.period, P=fit.P, t0=fit.t0, redchi=fit.redchi,
            method=c.source_guess.method, strength=c.source_guess.strength,
            n_dup=c.n_duplicate_guesses, n_methods=len(c.contributing_methods),
            methods="|".join(c.contributing_methods),
            n_peaks_used=fit.n_peaks_used, n_peaks_dropped=fit.n_peaks_dropped,
            n_peaks_required=c.n_peaks_required, n_teeth_available=c.n_teeth_available,
            few_peaks=c.few_peaks,
            frac_pos=d["frac_positive_heights"], mean_hsnr=float(np.nanmean(hs)) if hs.size else np.nan,
            frac_vertex=d["frac_vertex_in_window"],
            mean_h=float(np.nanmean(heights)) if heights.size else np.nan,
            passed=c.passed_gates,
            g_peaks=fit.n_peaks_used >= c.n_peaks_required,
            g_pos=d["frac_positive_heights"] >= 0.8,
            g_snr=(float(np.nanmean(hs)) if hs.size else -np.inf) >= 1.0,
            g_vertex=d["frac_vertex_in_window"] >= 0.8,
            pdm_theta=phase_dispersion_stat(t, f, fit.P),
            guard_ratio=(c.alias_guard or {}).get("half_tooth_ratio", np.nan),
            label=lab(fit.P),
        )
        r.update(_alias_discrimination_features(t, f, lags, acf, fit.P))
        rows.append(r)
    return dict(star=star, cands=rows)


def build_tasks(args) -> list:
    """Synthetic-suite or pb_scan task list."""
    if args.synthetic:
        from synthetic import make_suite
        return [dict(kind="synthetic", lc=lc) for lc in make_suite(args.synthetic, seed=args.seed)]
    import pb_scan
    tasks = pb_scan.build_tasks(args.n_per_cell, args.n_noise, args.seed_offset)
    if args.noise_only:
        tasks = [t for t in tasks if t["kind"] == "noise"]
    return tasks


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pb", action="store_true")
    p.add_argument("--noise-only", action="store_true")
    p.add_argument("--n-per-cell", type=int, default=12)
    p.add_argument("--n-noise", type=int, default=40)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--fit-kwargs", default="{}")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--n-workers", type=int, default=2)
    args = p.parse_args(argv)
    (args.outdir / "acf").mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    work = partial(process_one, outdir=str(args.outdir), fit_kwargs=json.loads(args.fit_kwargs))
    t0 = time.time()
    with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
        out = list(ex.map(work, tasks, chunksize=1))
    pd.DataFrame([o["star"] for o in out]).to_csv(args.outdir / "stars.csv", index=False)
    pd.DataFrame([r for o in out for r in o["cands"]]).to_csv(args.outdir / "cands.csv", index=False)
    print(f"{len(out)} stars in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
