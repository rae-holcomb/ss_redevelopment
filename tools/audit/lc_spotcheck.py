"""Spot-check sample SMARTS FITS files: structure, headers vs catalog, gaps, noise, label sanity.

Reads the sample manifest written by ``extract_sample.py`` (default
``data/smarts_sample/manifest.csv``, which already carries each file's
catalog row) and, for each file:

- HDU names/shapes; time grid identical to the first file's; cadence;
- primary-header injection keys vs the catalog row (max abs difference);
- WAVELET HDU vs the catalog image cube slice at the same row (needs the drive;
  skipped with ``--no-drive``);
- runs of linearly interpolated flux (second difference == 0 to float32
  precision, >= ``MIN_INTERP_RUN`` cadences): SMARTS' filled gaps;
- ``preprocessing.load_smarts_fits``: breaks found/matched, orbits flagged,
  sectors kept, finite points;
- noise: robust point-to-point scatter, per-orbit std (loader diagnostics);
- label sanity: Lomb-Scargle peak period on the loaded light curve, scored with
  ``evaluation.classify_period`` against PERIOD and the active-band range.

Writes ``sample_spotcheck.csv`` (one row per file), ``sample_interp_runs.csv``
and ``sample_segments.csv`` (one row per file x orbit) to ``--outdir``.
~1-2 min for 50 files.

Usage::

    python tools/audit/lc_spotcheck.py --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))
import smarts_data as sd  # noqa: E402
import evaluation  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402

HEADER_KEYS = ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL", "MINLAT", "MAXLAT",
               "DIFFROT", "TSPOT"]
MIN_INTERP_RUN = 5


def interp_runs(flux: np.ndarray, min_run: int = MIN_INTERP_RUN, tol: float = None) -> list:
    """(start, end) index pairs (inclusive) of runs where the flux is exactly linear.

    A run of k collinear points has k-2 consecutive zero second differences.
    ``tol`` defaults to a few float32 ulps of the flux level.
    """
    f = flux.astype(np.float64)
    tol = tol if tol is not None else 8 * np.finfo(np.float32).eps * np.nanmedian(np.abs(f))
    flat = np.abs(np.diff(f, 2)) <= tol
    runs, i, n = [], 0, len(flat)
    while i < n:
        if flat[i]:
            j = i
            while j + 1 < n and flat[j + 1]:
                j += 1
            if j - i + 3 >= min_run:
                runs.append((i, j + 2))
            i = j + 1
        else:
            i += 1
    return runs


def ls_peak(time: np.ndarray, flux: np.ndarray, pmin: float = 0.08, pmax: float = 200.0) -> tuple:
    """(peak period, peak power) of a Lomb-Scargle periodogram over [pmin, pmax] days."""
    m = np.isfinite(flux)
    t, y = time[m], flux[m]
    span = t.max() - t.min()
    fmin, fmax = 1 / pmax, 1 / pmin
    nf = int(np.ceil((fmax - fmin) * span * 5))
    freq = np.linspace(fmin, fmax, nf)
    pw = LombScargle(t, y).power(freq, method="fast")
    k = int(np.argmax(pw))
    return 1 / freq[k], float(pw[k])


def check_file(path: Path, cat_row: pd.Series, ref_time: np.ndarray | None, cube) -> tuple:
    """Run all per-file checks; returns (summary dict, interp-run rows, segment rows)."""
    rec = dict(file=path.name, row=int(cat_row["row"]), bundle=int(cat_row["bundle"]),
               size=path.stat().st_size)
    with fits.open(path) as h:
        rec["hdus"] = ";".join(f"{x.name}:{x.data.shape if x.data is not None else ()}" for x in h)
        hdr = h[0].header
        rec["hlsptarg"] = hdr.get("HLSPTARG")
        diffs = {k: abs(float(hdr[k]) - float(cat_row[k])) for k in HEADER_KEYS}
        rec["hdr_max_abs_diff"] = max(diffs.values())
        rec["hdr_worst_key"] = max(diffs, key=diffs.get)
        rec["hdr_bfly_match"] = bool(hdr["BFLY"]) == bool(cat_row["BFLY"])
        rec["hdr_keys_extra"] = ",".join(sorted(set(hdr) - {
            "SIMPLE", "BITPIX", "NAXIS", "EXTEND", "DATE-BEG", "DATE-END", "DATE-AVG", "DOI",
            "HLSPID", "HLSPLEAD", "HLSPNAME", "HLSPTARG", "HLSPVER", "INSTRUME", "LICENSE",
            "LICENURL", "OBSERVAT", "REFERENC", "TELAPSE", "TELESCOP", "XPOSURE", "SIMULATD",
            "BFLY", *HEADER_KEYS}))
        t = np.asarray(h["LIGHTCURVE"].data["time"], dtype=np.float64)
        f = np.asarray(h["LIGHTCURVE"].data["flux"], dtype=np.float64)
        wav = np.asarray(h["WAVELET"].data)
    rec["n_cadences"] = len(t)
    rec["dt_min"] = float(np.round(np.median(np.diff(t)) * 1440, 4))
    rec["time_span_d"] = float(t[-1] - t[0])
    rec["time_grid_identical"] = bool(ref_time is not None and len(t) == len(ref_time)
                                      and np.array_equal(t, ref_time))
    rec["n_nonfinite_flux"] = int((~np.isfinite(f)).sum())
    rec["flux_median"] = float(np.nanmedian(f))
    if cube is not None:
        rec["wavelet_matches_catalog_cube"] = bool(np.array_equal(wav, cube[rec["row"]]))
        rec["wavelet_matches_cube_T"] = bool(np.array_equal(wav, cube[rec["row"]].T))

    runs = interp_runs(f)
    rec["n_interp_runs"] = len(runs)
    rec["n_interp_pts"] = int(sum(e - s + 1 for s, e in runs))
    run_rows = [dict(file=path.name, start=s, end=e, t_start=t[s], t_end=t[e]) for s, e in runs]
    # Constant fill at the ends (mean-filled missing start/end cadences).
    rec["n_const_tail"] = int(np.argmax(f[::-1] != f[-1]))
    rec["n_const_head"] = int(np.argmax(f != f[0]))

    d1 = np.diff(f)
    rec["sigma_p2p"] = float(1.4826 * np.median(np.abs(d1[d1 != 0] - np.median(d1[d1 != 0])))
                             / np.sqrt(2))
    rec["flux_p05_p95"] = float(np.percentile(f, 95) - np.percentile(f, 5))

    lc = load_smarts_fits(path)
    dg = lc.diagnostics
    rec["n_breaks"] = len(dg["break_time"])
    rec["n_breaks_matched"] = int(np.sum(dg["matched"]))
    rec["n_orbits_total"] = lc.n_orbits_total
    rec["n_orbits_flagged"] = lc.n_orbits_flagged
    rec["sectors_used"] = " ".join(map(str, lc.sectors_used))
    rec["n_sectors_used"] = len(lc.sectors_used)
    rec["n_finite_loaded"] = int(np.isfinite(lc.flux).sum())
    rec["loaded_span_d"] = float(lc.time[-1] - lc.time[0])
    seg_rows = [dict(file=path.name, orbit=i + 1, std=float(s), flagged=bool(fl),
                     median_roll_std=float(r))
                for i, (s, fl, r) in enumerate(zip(dg["segment_std"], dg["segment_flagged"],
                                                    dg["segment_median_roll_std"]))]

    p_eq = float(cat_row["PERIOD"])
    prange = evaluation.active_band_period_range(p_eq, float(cat_row["DIFFROT"]),
                                                 float(cat_row["MINLAT"]), float(cat_row["MAXLAT"]))
    pk, pw = ls_peak(lc.time, lc.flux)
    rec["ls_peak_period"] = pk
    rec["ls_peak_power"] = pw
    rec["ls_outcome"] = evaluation.classify_period(pk, p_eq, period_range=prange)
    rec["P_lo"], rec["P_hi"] = prange
    return rec, run_rows, seg_rows


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(REPO / "data" / "smarts_sample" / "manifest.csv"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-drive", action="store_true", help="skip the catalog wavelet-cube check")
    args = ap.parse_args()
    man = pd.read_csv(args.manifest)
    fdir = Path(args.manifest).parent
    cube = None
    if not args.no_drive:
        root = sd.require_drive(sd.data_root(args.data_root))
        cat_fits = Path(root, sd.EXTRACT_DIR, "catalog", sd.CATALOG_FITS)
        hc = fits.open(cat_fits, memmap=True)
        cube = hc[2].data  # numpy shape (1_000_000, 64, 64), memory-mapped
    ref_time = None
    recs, runs, segs = [], [], []
    for _, r in man.iterrows():
        p = fdir / r["file"]
        if ref_time is None:
            with fits.open(p) as h:
                ref_time = np.asarray(h["LIGHTCURVE"].data["time"], dtype=np.float64)
        rec, rr, ss = check_file(p, r, ref_time, cube)
        recs.append(rec)
        runs += rr
        segs += ss
        print(f"{rec['file']} P={r['PERIOD']:.3f} hdr_diff={rec['hdr_max_abs_diff']:.2g} "
              f"interp_runs={rec['n_interp_runs']} breaks={rec['n_breaks_matched']}/{rec['n_breaks']} "
              f"flagged={rec['n_orbits_flagged']} LS={rec['ls_peak_period']:.3f} ({rec['ls_outcome']})",
              flush=True)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs).merge(man.drop(columns=["file", "bundle", "size"]), on="row")
    df.to_csv(out / "sample_spotcheck.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "sample_interp_runs.csv", index=False)
    pd.DataFrame(segs).to_csv(out / "sample_segments.csv", index=False)
    print(f"wrote {out}/sample_spotcheck.csv ({len(df)} files)")


if __name__ == "__main__":
    main()
