"""T20 phase 2: periodicity search on the recovered noise light curves (stacks from galaxy_stack.py).

For each of the 1,719 noise sources, three stacks exist (halves A and B, and
A+B). This tool

- **fits**: writes each stack as a SMARTS-layout FITS file (dummy PERIOD = 1),
  so the production loader (``load_smarts_fits``: gap re-detection, bad-orbit
  flagging, even cadence) and pipeline see it exactly like a SMARTS file;
- **ls**: Lomb-Scargle on the loaded (gap-dropped, bad-orbit-dropped) flux of
  each stack (A, B, all) over 0.1-60 d: top peaks, power, semi-amplitude, plus
  ``persist`` = the A and B top peaks agree within 2% (Rae's disjoint-subset test);
- **pipeline**: runs ``scorecard.score_one_star`` (Stage 1 + comb fit, default
  methods) on the A+B stack: the period the production pipeline would report
  for a SMARTS star with no injected signal at all.

Outputs under ``<root>/extracted/galaxy_stack/``: ``fits/<galaxy>_<stack>.fits``,
``ls_peaks.csv``, ``ls_power.npy`` (+ ``ls_freq.npy``), ``pipeline_on_stacks.csv``.

Usage::

    $PY tools/audit/galaxy_periods.py fits
    $PY tools/audit/galaxy_periods.py ls --n-workers 3
    $PY tools/audit/galaxy_periods.py pipeline --n-workers 3
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))
sys.path.insert(0, str(REPO / "benchmarks"))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

STACK_NAMES = ("A", "B", "all")
PMIN, PMAX = 0.1, 60.0
N_PEAKS = 3


def time_grid() -> np.ndarray:
    """The common SMARTS 30-min grid (t = 0 .. 364.98 d)."""
    return (np.arange(gp.N_CAD) / 48.0).astype(np.float32)


def write_fits(path: Path, flux: np.ndarray, tic: int) -> None:
    """Write ``flux`` as a SMARTS-layout FITS file (PRIMARY + LIGHTCURVE), atomically."""
    from astropy.io import fits
    prim = fits.PrimaryHDU()
    prim.header["PERIOD"] = (1.0, "dummy; recovered noise LC, no injection")
    prim.header["TICID"] = int(tic)
    lc = fits.BinTableHDU.from_columns(
        [fits.Column("time", "E", array=time_grid()),
         fits.Column("flux", "E", array=np.asarray(flux, np.float32))], name="LIGHTCURVE")
    with sd.atomic_path(path) as tmp:
        fits.HDUList([prim, lc]).writeto(tmp)


def freq_grid() -> np.ndarray:
    """Frequency grid (1/d), oversampled 5x over the 365-d baseline, PMIN..PMAX."""
    return np.linspace(1 / PMAX, 1 / PMIN, int((1 / PMIN - 1 / PMAX) * 365 * 5))


def ls_one(path: str) -> dict:
    """Load one stack with load_smarts_fits and run Lomb-Scargle; module-level for the pool."""
    from astropy.timeseries import LombScargle
    from preprocessing import load_smarts_fits
    warnings.simplefilter("ignore")
    try:
        lc = load_smarts_fits(path)
    except Exception as e:  # noqa: BLE001 -- e.g. every orbit flagged
        return dict(path=path, error=f"{type(e).__name__}: {e}")
    ok = np.isfinite(lc.flux)
    t, y = lc.time[ok], lc.flux[ok]
    f = freq_grid()
    ls = LombScargle(t, y)
    p = ls.power(f, method="fast")
    # top peaks: local maxima, separated by > 2% in period
    loc = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    loc = loc[np.argsort(-p[loc])]
    peaks = []
    for k in loc:
        if all(abs(1 / f[k] - q) / q > 0.02 for q in peaks):
            peaks.append(1 / f[k])
        if len(peaks) == N_PEAKS:
            break
    rec = dict(path=path, n_points=int(ok.sum()), n_orbits_flagged=lc.n_orbits_flagged,
               rms_ppm=float(np.std(y) * 1e6), error=None)
    for i, q in enumerate(peaks):
        model = ls.model(t, 1 / q)
        rec[f"P{i + 1}"] = float(q)
        rec[f"pow{i + 1}"] = float(p[np.argmin(np.abs(f - 1 / q))])
        rec[f"amp{i + 1}_ppm"] = float(0.5 * (model.max() - model.min()) * 1e6)
    rec["power"] = p.astype(np.float32)
    return rec


def pipeline_one(path: str, galaxy: int) -> dict:
    """Run the scorecard's full per-star pipeline on one stack FITS (dummy true period)."""
    import scorecard as sc
    task = dict(kind="fits", path=path, star_id=f"galaxy{galaxy:04d}", true_period=1.0,
                shear=0.0, minlat=0.0, maxlat=0.0)
    r = sc.score_one_star(task, sc.DEFAULT_PIPELINE_METHODS, 0.10)
    return dict(galaxy=galaxy, final_period=r.get("final_period"), success=r.get("pipeline_success"),
                message=(r.get("pipeline_message") or "")[:200], error=r.get("error"),
                top1={m: v.get("top1") for m, v in (r.get("methods") or {}).items()},
                seconds=r.get("seconds"))


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["fits", "ls", "pipeline"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-workers", type=int, default=3)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    sdir = Path(root, sd.EXTRACT_DIR, "galaxy_stack")
    fdir = sd.extract_dir(root, "galaxy_stack", "fits")
    idx = pd.read_csv(sdir / "stack_index.csv")

    if args.step == "fits":
        S = np.load(sdir / "stacks.npy", mmap_mode="r")
        for i, r in idx.iterrows():
            for j, name in enumerate(STACK_NAMES):
                p = fdir / f"{r.galaxy:04d}_{name}.fits"
                if not p.exists():
                    write_fits(p, S[i, j], r.tic)
        print(f"wrote FITS for {len(idx)} galaxies -> {fdir}")
        return

    t0 = time.time()
    if args.step == "ls":
        jobs = [(r.galaxy, n, str(fdir / f"{r.galaxy:04d}_{n}.fits")) for r in idx.itertuples()
                for n in STACK_NAMES]
        with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
            res = list(ex.map(ls_one, [j[2] for j in jobs], chunksize=20))
        power = np.full((len(idx), len(STACK_NAMES), len(freq_grid())), np.nan, np.float32)
        recs = []
        for (g, n, _), r in zip(jobs, res):
            if "power" in r:
                power[np.searchsorted(idx.galaxy.values, g), STACK_NAMES.index(n)] = r.pop("power")
            recs.append(dict(galaxy=g, stack=n, **r))
        df = pd.DataFrame(recs)
        w = df.pivot(index="galaxy", columns="stack", values="P1")
        persist = (np.abs(w["A"] - w["B"]) / w["all"] < 0.02)
        df = df.merge(persist.rename("persist").reset_index(), on="galaxy")
        with sd.atomic_path(sdir / "ls_peaks.csv") as tmp:
            df.to_csv(tmp, index=False)
        with sd.atomic_path(sdir / "ls_power.npy") as tmp:
            np.save(tmp, power)
        with sd.atomic_path(sdir / "ls_freq.npy") as tmp:
            np.save(tmp, freq_grid())
        print(f"LS on {len(jobs)} stacks in {time.time() - t0:.0f} s; "
              f"A/B top peak persists in {int(persist.sum())}/{len(persist)} galaxies")
        return

    # pipeline
    paths = [str(fdir / f"{g:04d}_all.fits") for g in idx.galaxy]
    with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
        res = list(ex.map(pipeline_one, paths, idx.galaxy.tolist(), chunksize=10))
    df = pd.DataFrame(res)
    df = pd.concat([df.drop(columns="top1"), pd.json_normalize(df.top1).add_prefix("top1_")], axis=1)
    with sd.atomic_path(sdir / "pipeline_on_stacks.csv") as tmp:
        df.to_csv(tmp, index=False)
    print(f"pipeline on {len(df)} stacks in {time.time() - t0:.0f} s; "
          f"{int(df.success.sum())} returned a period")


if __name__ == "__main__":
    main()
