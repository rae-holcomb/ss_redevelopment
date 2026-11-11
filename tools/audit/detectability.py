"""T26: per-star detectability of the injected rotation signal in SMARTS files.

SMARTS injects multiplicatively (T20): f = g * s, with g the TESS noise light curve
of the star's noise source (TIC) and s the butterpy spot signal. T20 recovered g for
every source by stacking low-amplitude members (``extracted/galaxy_stack/stacks.npy``,
shape (n_tic, 3, 17520): halves A, B and A+B). So for each dev star:

    s = f / g            (the injected signal, essentially noise-free: T20 found
                          member - stack p2p residuals of 0.25 ppm)
    n = g - 1            (the noise the pipeline had to see the signal through)

If the star is itself a stacking member (role A or B), the disjoint other half is used
for g, so the star's own signal cannot leak into its noise estimate.

Both series are restricted to exactly the cadences the pipeline saw
(``load_smarts_fits`` with the scorecard's sectors: all native sectors for full-year,
the manifest's ``sectors`` for assigned coverage), then compared on rotation timescales:

    D_band = sqrt( sum_band LS_psd(s - 1) / sum_band LS_psd(n) )

where the band is the frequency range of periods [P_lo / 1.5, 1.5 * P_hi] (P_lo/P_hi =
the active-latitude-band period range; = P for solid-body stars), on a uniform frequency
grid (df = 1 / (5 * baseline)). Both periodograms use the same timestamps, so D_band is a
band-limited signal-to-noise amplitude ratio that is independent of the normalisation
convention. The script also stores the ingredients (signal and noise RMS, observed time
span, number of cycles) so alternative statistics can be built later
(e.g. D_band * sqrt(n_cycles)).

Outputs (one CSV per manifest and coverage mode, written atomically; default
``reports/<t26 folder>/data/detectability_<mode>.csv``, or ``--out``). Each row records the
loader's ``sectors`` pattern, the key the scorecard joins on. Resumable: checkpoints every
``--chunk`` stars to ``<root>/extracted/t26_detectability/parts_v2/<manifest>/<mode>/``.
``tools/build_detectability_table.py`` runs this for the dev sets and writes the committed
``benchmarks/detectability_dev.csv`` (scorecard v6).

Usage::

    $PY tools/audit/detectability.py --mode fullyear --n-workers 4 --resume
    $PY tools/audit/detectability.py --mode assigned --n-workers 4 --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import argparse  # noqa: E402
import time as _time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

import coverage as cov  # noqa: E402
import smarts_data as sd  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402

REPORT = REPO / "reports" / "2026-11-10_t26-detectability"
STACK_SUBDIR = ("extracted", "galaxy_stack")
BAND_FACTOR = 1.5
NYQ = 0.5 / (1800 / 86400.0)  # c/d at 30-min cadence
NOISE_TABLE = REPO / "benchmarks" / "noise_sources_dev.csv"  # sim_id -> noise TIC for the dev sets
HALF = {"A": 1, "B": 0}  # star in half A -> use stack B (index 1), and vice versa; else A+B (2)


def stack_paths(root: Path) -> dict:
    """Paths of the T20 stack products under the data root."""
    d = Path(root, *STACK_SUBDIR)
    return dict(stacks=d / "stacks.npy", index=d / "stack_index.csv", selection=d / "selection.csv")


def check_stacks_readable(root: Path) -> None:
    """Real-read check that the drive is mounted and the T20 stacks are there; exit with a clear message if not."""
    p = stack_paths(root)
    try:
        arr = np.load(p["stacks"], mmap_mode="r")
        float(arr[0, 2, 100])  # touch actual bytes on the drive
        pd.read_csv(p["index"], nrows=2)
    except (FileNotFoundError, OSError) as exc:
        sys.exit(f"Cannot read the T20 noise stacks under {root} ({exc}).\n"
                 "Plug in the Data_Vault drive, or point --data-root / $SMARTS_DATA_ROOT at it.")


def stack_row_for_stars(stars: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Per star: stacks.npy row (TIC) and which estimate to use (0=A, 1=B, 2=A+B)."""
    p = stack_paths(root)
    idx = pd.read_csv(p["index"])
    sel = pd.read_csv(p["selection"], usecols=["row", "galaxy", "role"])
    gal_col = "galaxy" if "galaxy" in idx.columns else idx.columns[0]
    idx = idx.reset_index().rename(columns={"index": "stack_row"})
    out = stars.merge(idx[[gal_col, "stack_row"]].rename(columns={gal_col: "galaxy"}), on="galaxy", how="left")
    out = out.merge(sel[["row", "role"]], on="row", how="left")
    out["stack_which"] = out.role.map(HALF).fillna(2).astype(int)
    return out


def band_freqs(p_lo: float, p_hi: float, baseline: float) -> np.ndarray:
    """Uniform frequency grid (c/d) covering periods [p_lo/1.5, 1.5*p_hi], capped at Nyquist."""
    fmin = 1.0 / (BAND_FACTOR * p_hi)
    fmax = min(BAND_FACTOR / p_lo, 0.98 * NYQ)
    df = 1.0 / (5.0 * baseline)
    n = max(int(np.ceil((fmax - fmin) / df)), 50)
    return np.linspace(fmin, fmax, n)


def detectability_one(t: np.ndarray, sig: np.ndarray, noise: np.ndarray,
                      p_true: float, p_lo: float, p_hi: float) -> dict:
    """Band-limited signal/noise statistics for one star on the cadences it was observed.

    t: times (d); sig: injected signal s - 1; noise: noise light curve g - 1 (same cadences,
    finite only). Returns a dict of the statistic and its ingredients.
    """
    baseline = float(t.max() - t.min())
    f = band_freqs(p_lo, p_hi, baseline)
    ps = LombScargle(t, sig, fit_mean=False, center_data=True, normalization="psd").power(f)
    pn = LombScargle(t, noise, fit_mean=False, center_data=True, normalization="psd").power(f)
    near = np.abs(1.0 / f - p_true) <= 0.1 * p_true
    n_cyc = float(len(t) * (1800 / 86400.0) / p_true)  # observed time / P
    d_band = float(np.sqrt(ps.sum() / pn.sum()))
    return dict(
        D_band=d_band,
        D_peak=float(np.sqrt(ps[near].max() / np.median(pn))) if near.any() else np.nan,
        sig_rms=float(np.std(sig)), sig_p2p_5_95=float(np.subtract(*np.percentile(sig, [95, 5]))),
        noise_rms=float(np.std(noise)),
        noise_p2p_ppm=float(np.median(np.abs(np.diff(noise))) * 1.4826 / np.sqrt(2) * 1e6),
        n_obs=int(len(t)), obs_days=float(len(t) * 1800 / 86400.0), baseline_days=baseline,
        n_cycles_obs=n_cyc, D_band_sqrt_ncyc=d_band * np.sqrt(n_cyc),
    )


def star_series(path: Path, g: np.ndarray, sectors, return_sectors: bool = False) -> tuple:
    """Return (t, s-1, g-1, f-1) on exactly the cadences the pipeline sees for this star.

    With return_sectors=True, also return the loader's sectors_used (the pattern the
    scorecard records in its `sectors` column), as a 5th element.
    """
    with fits.open(path) as h:
        t_raw = np.asarray(h["LIGHTCURVE"].data["time"], float)
        f_raw = np.asarray(h["LIGHTCURVE"].data["flux"], float)
    pre = load_smarts_fits(path, sectors=sectors)
    ok = np.isfinite(pre.flux)
    te = pre.time[ok]
    j = np.clip(np.searchsorted(t_raw, te), 1, len(t_raw) - 1)
    j = np.where(np.abs(t_raw[j - 1] - te) < np.abs(t_raw[j] - te), j - 1, j)
    good = (np.abs(t_raw[j] - te) < 0.25 * 1800 / 86400.0) & np.isfinite(g[j]) & (g[j] > 0)
    j = j[good]
    out = (t_raw[j], f_raw[j] / g[j] - 1.0, g[j] - 1.0, f_raw[j] - 1.0)
    return out + (list(pre.sectors_used),) if return_sectors else out


def process_one(rec: dict, data_dir: str, stacks_path: str) -> dict:
    """Module-level worker: detectability for one star (record from the joined manifest)."""
    stacks = np.load(stacks_path, mmap_mode="r")
    out = dict(star_id=rec["star_id"])
    try:
        g = np.asarray(stacks[int(rec["stack_row"]), int(rec["stack_which"])], float)
        sectors = None if rec["sectors_use"] is None else list(rec["sectors_use"])
        t, s, n, _, used = star_series(Path(data_dir) / rec["path"], g, sectors, return_sectors=True)
        out["sectors"] = cov.format_pattern(used)
        out.update(detectability_one(t, s, n, rec["true_period"], rec["P_lo"], rec["P_hi"]))
        out["error"] = ""
    except Exception as exc:  # recorded, not raised: one bad star must not kill the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def join_stack_rows(man: pd.DataFrame, root: Path, galaxy_map: str | None = None) -> pd.DataFrame:
    """Manifest rows + noise TIC (committed benchmarks/noise_sources_dev.csv by sim_id, or a
    star_id,noise_tic CSV) + stacks.npy row and which estimate to use."""
    if galaxy_map:
        stars = man.merge(pd.read_csv(galaxy_map, usecols=["star_id", "noise_tic"]), on="star_id", how="left")
    else:
        noise = pd.read_csv(NOISE_TABLE, usecols=["sim_id", "tic"]).rename(columns={"tic": "noise_tic"})
        stars = man.merge(noise, on="sim_id", how="left")
    idx = pd.read_csv(stack_paths(root)["index"])
    stars = stars.merge(idx[["tic", "galaxy"]].rename(columns={"tic": "noise_tic"}), on="noise_tic", how="left")
    return stack_row_for_stars(stars, root)


def run(manifest: str, mode: str, out_csv: Path, data_dir: str, root: Path, n_workers: int = 2,
        chunk: int = 100, resume: bool = True, galaxy_map: str | None = None) -> pd.DataFrame:
    """Compute detectability for one manifest and coverage mode; resumable, atomic output CSV.

    mode 'fullyear' loads every native sector; 'assigned' the manifest's `sectors` column
    (exactly what the scorecard does with / without --ignore-manifest-sectors).
    """
    man = pd.read_csv(manifest)
    stars = join_stack_rows(man, root, galaxy_map)
    if stars.stack_row.isna().any():
        raise SystemExit(f"{int(stars.stack_row.isna().sum())} stars have no noise stack (TIC not mapped)")
    stars["sectors_use"] = ([None] * len(stars) if mode == "fullyear"
                            else stars.sectors.map(cov.parse_pattern))
    parts = sd.extract_dir(root, "t26_detectability", "parts_v2", Path(manifest).stem, mode)
    done = set()
    if resume:
        for f in parts.glob("part_*.csv"):
            done |= set(pd.read_csv(f).star_id)
    todo = stars[~stars.star_id.isin(done)]
    print(f"{Path(manifest).stem} {mode}: {len(stars)} stars, {len(done)} done, {len(todo)} to do", flush=True)
    worker = partial(process_one, data_dir=data_dir, stacks_path=str(stack_paths(root)["stacks"]))
    recs = todo[["star_id", "path", "true_period", "P_lo", "P_hi", "stack_row", "stack_which",
                 "sectors_use"]].to_dict("records")
    t0 = _time.time()
    if recs:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=determinism.init_worker) as ex:
            for k in range(0, len(recs), chunk):
                res = list(ex.map(worker, recs[k:k + chunk]))
                with sd.atomic_path(parts / f"part_{int(_time.time() * 1e3)}_{k:05d}.csv") as tmp:
                    pd.DataFrame(res).to_csv(tmp, index=False)
                print(f"  {k + len(res)}/{len(recs)}  {_time.time() - t0:.0f}s", flush=True)
    allp = pd.concat([pd.read_csv(f, dtype={"sectors": str}) for f in sorted(parts.glob("part_*.csv"))])
    allp = allp.drop_duplicates("star_id", keep="last")
    keep = ["star_id", "sim_id", "noise_tic", "galaxy", "stack_row", "role", "stack_which", "amp_proxy", "amp_tier"]
    out = stars[[c for c in keep if c in stars.columns]].merge(allp, on="star_id", how="left")
    out.insert(1, "mode", mode)
    with sd.atomic_path(Path(out_csv)) as tmp:
        out.to_csv(tmp, index=False)
    print(f"wrote {out_csv}; errors: {(out.error.fillna('') != '').sum()}")
    return out


def main():
    """CLI entry point: compute detectability for one coverage mode of a manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fullyear", "assigned"], required=True)
    ap.add_argument("--manifest", default=str(REPO / "datasets" / "dev_medium.csv"))
    ap.add_argument("--data-dir", default=str(REPO / "data" / "smarts_dev"))
    ap.add_argument("--data-root", default=None, help="SMARTS data root on the drive ($SMARTS_DATA_ROOT)")
    ap.add_argument("--galaxy-map", default=None,
                    help="CSV with star_id,noise_tic (default: benchmarks/noise_sources_dev.csv by sim_id)")
    ap.add_argument("--out", default=None, help="output CSV (default: report data/detectability_<mode>.csv)")
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    root = sd.data_root(a.data_root)
    check_stacks_readable(root)
    run(a.manifest, a.mode, Path(a.out or REPORT / "data" / f"detectability_{a.mode}.csv"), a.data_dir, root,
        a.n_workers, a.chunk, a.resume, a.galaxy_map)


if __name__ == "__main__":
    main()
