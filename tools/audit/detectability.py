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

Outputs (one CSV per coverage mode, written atomically):
``reports/<t26 folder>/data/detectability_<mode>.csv``. Resumable: checkpoints every
``--chunk`` stars to ``<root>/extracted/t26_detectability/parts/``.

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
# star -> noise TIC map lives in the scorecard runs (gitignored, main checkout only)
MAIN_T10 = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment/runs")
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


def star_series(path: Path, g: np.ndarray, sectors) -> tuple:
    """Return (t, s-1, g-1, f-1) on exactly the cadences the pipeline sees for this star."""
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
    return t_raw[j], f_raw[j] / g[j] - 1.0, g[j] - 1.0, f_raw[j] - 1.0


def process_one(rec: dict, data_dir: str, stacks_path: str) -> dict:
    """Module-level worker: detectability for one star (record from the joined manifest)."""
    stacks = np.load(stacks_path, mmap_mode="r")
    out = dict(star_id=rec["star_id"])
    try:
        g = np.asarray(stacks[int(rec["stack_row"]), int(rec["stack_which"])], float)
        sectors = None if rec["sectors_use"] is None else list(rec["sectors_use"])
        t, s, n, _ = star_series(Path(data_dir) / rec["path"], g, sectors)
        out.update(detectability_one(t, s, n, rec["true_period"], rec["P_lo"], rec["P_hi"]))
        out["error"] = ""
    except Exception as exc:  # recorded, not raised: one bad star must not kill the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    """CLI entry point: compute detectability for one coverage mode of a manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fullyear", "assigned"], required=True)
    ap.add_argument("--manifest", default=str(REPO / "datasets" / "dev_medium.csv"))
    ap.add_argument("--data-dir", default=str(REPO / "data" / "smarts_dev"))
    ap.add_argument("--data-root", default=None, help="SMARTS data root on the drive ($SMARTS_DATA_ROOT)")
    ap.add_argument("--galaxy-map", default=None,
                    help="CSV with star_id,noise_tic (default: main checkout t10 stars.csv.gz)")
    ap.add_argument("--outdir", default=str(REPORT / "data"))
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    root = sd.data_root(a.data_root)
    check_stacks_readable(root)
    man = pd.read_csv(a.manifest)
    neg = pd.read_csv(REPO / "datasets" / "negatives_v0_sources.csv", usecols=["tic", "galaxy"])
    gmap = pd.read_csv(a.galaxy_map or MAIN_T10 / "t10_dev_medium_fullyear" / "stars.csv.gz",
                       usecols=["star_id", "noise_tic"])
    stars = man.merge(gmap, on="star_id", how="left")
    # galaxy index for each TIC: from the full stack index (covers every TIC, unlike the negatives list)
    idx = pd.read_csv(stack_paths(root)["index"])
    if "tic" in idx.columns:
        stars = stars.merge(idx[["tic", "galaxy"]].rename(columns={"tic": "noise_tic"}), on="noise_tic", how="left")
    else:
        stars = stars.merge(neg.rename(columns={"tic": "noise_tic"}), on="noise_tic", how="left")
    stars = stack_row_for_stars(stars, root)
    stars["sectors_use"] = ([None] * len(stars) if a.mode == "fullyear"
                            else stars.sectors.map(cov.parse_pattern))

    parts = sd.extract_dir(root, "t26_detectability", "parts", a.mode)
    done = set()
    if a.resume:
        for f in parts.glob("part_*.csv"):
            done |= set(pd.read_csv(f).star_id)
    todo = stars[~stars.star_id.isin(done)]
    print(f"{a.mode}: {len(stars)} stars, {len(done)} done, {len(todo)} to do", flush=True)
    worker = partial(process_one, data_dir=a.data_dir, stacks_path=str(stack_paths(root)["stacks"]))
    recs = todo[["star_id", "path", "true_period", "P_lo", "P_hi", "stack_row", "stack_which",
                 "sectors_use"]].to_dict("records")
    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=a.n_workers, initializer=determinism.init_worker) as ex:
        for k in range(0, len(recs), a.chunk):
            res = list(ex.map(worker, recs[k:k + a.chunk]))
            with sd.atomic_path(parts / f"part_{int(_time.time() * 1e3)}_{k:05d}.csv") as tmp:
                pd.DataFrame(res).to_csv(tmp, index=False)
            print(f"  {k + len(res)}/{len(recs)}  {_time.time() - t0:.0f}s", flush=True)

    allp = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("part_*.csv"))]).drop_duplicates("star_id", keep="last")
    keep = ["star_id", "noise_tic", "galaxy", "stack_row", "role", "stack_which", "amp_proxy", "amp_tier"]
    out = stars[keep].merge(allp, on="star_id", how="left")
    with sd.atomic_path(Path(a.outdir) / f"detectability_{a.mode}.csv") as tmp:
        out.to_csv(tmp, index=False)
    print(f"wrote {a.outdir}/detectability_{a.mode}.csv; errors: {(out.error.fillna('') != '').sum()}")


if __name__ == "__main__":
    main()
