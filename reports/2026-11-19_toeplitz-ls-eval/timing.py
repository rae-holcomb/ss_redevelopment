#!/usr/bin/env python3
"""timing.py -- single-thread wall time of astropy LombScargle vs Toeplitz-LS (P50).

Three experiments, all in one process with BLAS pinned to 1 thread (determinism.pin_blas_threads):
  scan_N   synthetic gappy 30-min light curves, N = 500 .. 100k points, fixed Nf = 4000;
  scan_Nf  fixed N = 14,000 (a full-year SMARTS star), Nf = 300 .. 200k;
  real     the 5 accuracy stars, assigned + full-year, on guess_lombscargle's own grids
           ("lombscargle" spp=1, and "lombscargle_short" max_period=15 spp=5).
Backends (each timed as the call guess_lombscargle would make, including the Python wrapper):
  astropy_fast  LombScargle(t, y, normalization="standard").autopower(...)  -- today's code
  tls_generic   toeplitz_ls.tls.power(..., nterms=1) from `make generic`
  tls_native    same from `make native` (-march=native)
  tlsf_native   float32 variant, `make native`
Each point: median of `repeats` timeit-style loops (loop count chosen so one loop >= 0.2 s).

The two builds are loaded in separate subprocesses (both are packages named toeplitz_ls).
Usage:
    python timing.py --generic <scratch>/Toeplitz-LS/toeplitz-ls --native <scratch>/Toeplitz-LS-native/toeplitz-ls
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()
import numpy as np  # noqa: E402

N_GRID = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
NF_GRID = [300, 1000, 3000, 10000, 30000, 100000, 200000]
DT = 1.0 / 48  # 30 min in days


def synth(N, seed=0):
    """Gappy 30-min light curve with N finite points: TESS-like 13.7-d orbits with 1-d gaps, ~10% random drops."""
    rng = np.random.default_rng(seed)
    n_slots = int(N / 0.85) + 200
    t = np.arange(n_slots) * DT
    keep = ((t % 13.7) > 1.0) & (rng.random(n_slots) > 0.1)
    t = t[keep][:N]
    y = 1 + 0.01 * np.sin(2 * np.pi * t / 3.7) + 0.003 * rng.normal(size=t.size)
    return t, y


def best_time(fn, repeats):
    """Median per-call seconds over `repeats` loops (timeit.autorange-sized)."""
    timer = timeit.Timer(fn)
    n, _ = timer.autorange()
    n = max(1, n)
    return float(np.median([timer.timeit(n) / n for _ in range(repeats)]))


def run_backend(backend, pkg, cases, repeats):
    """Time one backend on every case; cases = list of dict(t, y, f0, df, Nf, spp, fmax)."""
    out = []
    if backend.startswith("tls"):
        sys.path.insert(0, pkg)
        from toeplitz_ls import tls, tlsf
        mod = tlsf if backend.startswith("tlsf") else tls
    else:
        from astropy.timeseries import LombScargle
    for c in cases:
        t, y = np.asarray(c["t"]), np.asarray(c["y"])
        if backend == "astropy_fast":
            def fn():
                return LombScargle(t, y, normalization="standard").autopower(
                    minimum_frequency=c["f0"], maximum_frequency=c["fmax"], samples_per_peak=c["spp"])
        else:
            def fn():
                return mod.power(c["Nf"], c["df"], c["f0"], t, y, nterms=1, normalization="standard")
        out.append(dict(c["label"], backend=backend, seconds=best_time(fn, repeats)))
        print(out[-1], flush=True)
    return out


def make_cases():
    """Synthetic scans plus the real-star grids; returns JSON-serialisable case dicts."""
    cases = []

    def case(t, y, Nf, label, spp=1):
        B = t.max() - t.min()
        df = 1.0 / (spp * B)
        f0 = 2.0 / B
        fmax = f0 + df * (Nf - 1) + 0.5 * df
        return dict(t=t.tolist(), y=y.tolist(), f0=f0, df=df, Nf=Nf, spp=spp, fmax=fmax, label=label)

    for N in N_GRID:
        t, y = synth(N)
        cases.append(case(t, y, 4000, dict(exp="scan_N", N=N, Nf=4000)))
    t, y = synth(14000)
    for Nf in NF_GRID:
        cases.append(case(t, y, Nf, dict(exp="scan_Nf", N=14000, Nf=Nf)))
    # Roman WG6-like: 12.1-min cadence, one 71-d season, fmax = 1/(4 dt)
    rng = np.random.default_rng(1)
    tr = np.arange(0, 71, 12.1 / 1440)
    yr = 1 + 0.01 * np.sin(2 * np.pi * tr / 5.0) + 0.003 * rng.normal(size=tr.size)
    nf_r = int((1 / (4 * 12.1 / 1440) - 2 / 71) * 71) + 1
    cases.append(case(tr, yr, nf_r, dict(exp="roman_season", N=tr.size, Nf=nf_r)))

    import pandas as pd
    from preprocessing import load_smarts_fits
    sys.path.insert(0, str(HERE))
    from accuracy import STARS, ls_grid
    man = pd.read_csv(ROOT / "datasets" / "dev_small_clean.csv").set_index("star_id")
    for sid in STARS:
        r = man.loc[sid]
        assigned = [int(s) for s in str(r["sectors"]).split("-")]
        for mode, sectors in (("assigned", assigned), ("fullyear", None)):
            pre = load_smarts_fits(ROOT / "data" / "smarts_dev" / r["path"], sectors=sectors)
            m = np.isfinite(pre.time) & np.isfinite(pre.flux)
            t, y = pre.time[m], pre.flux[m]
            for grid, kw in (("lombscargle", dict(spp=1)), ("lombscargle_short", dict(max_period=15.0, spp=5))):
                f0, fmax, df, nf, _ = ls_grid(t, **kw)
                cases.append(dict(t=t.tolist(), y=y.tolist(), f0=f0, df=df, Nf=nf, spp=kw["spp"], fmax=fmax,
                                  label=dict(exp="real", star_id=sid, mode=mode, grid=grid, N=len(t), Nf=nf)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic")
    ap.add_argument("--native")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--worker", nargs=3, metavar=("BACKEND", "PKG", "CASES"))
    ap.add_argument("--out", default=str(HERE / "data" / "timing.csv"))
    args = ap.parse_args()
    if args.worker:
        backend, pkg, cases_path = args.worker
        cases = json.loads(Path(cases_path).read_text())
        res = run_backend(backend, pkg, cases, args.repeats)
        Path(cases_path + f".{backend}.json").write_text(json.dumps(res))
        return
    cases_path = HERE / "data" / "_timing_cases.json"
    cases_path.write_text(json.dumps(make_cases()))
    results = []
    for backend, pkg in (("astropy_fast", "-"), ("tls_generic", args.generic),
                         ("tls_native", args.native), ("tlsf_native", args.native)):
        subprocess.run([sys.executable, __file__, "--repeats", str(args.repeats),
                        "--worker", backend, pkg, str(cases_path)], check=True)
        p = Path(str(cases_path) + f".{backend}.json")
        results += json.loads(p.read_text())
        p.unlink()
    cases_path.unlink()
    import pandas as pd
    pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
