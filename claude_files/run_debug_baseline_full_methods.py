"""
run_debug_baseline_full_methods.py

Same as run_debug_baseline.py, but runs the FULL candidate-generation
method set (all seven guess_* functions: pairwise_histogram, lombscargle,
acf_fft, wavelet, lombscargle_short, acf_fft_short, acf_fft_highpass)
instead of gather_initial_guesses' three-method default. Intended as a
direct comparison against baseline_results.csv to separate "genuinely
hard" failures from "default config never even proposed the right
period" failures, since the debug set skews toward the <10-day regime
that the opt-in short-period methods exist to address.

Usage
-----
    python run_debug_baseline_full_methods.py --data-dir debug_set --outdir baseline_results_full
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "claude_files"))

from acf_utils import compute_acf  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402
from guesses import (  # noqa: E402
    guess_pairwise_histogram, guess_lombscargle, guess_acf_fft, guess_wavelet,
    guess_lombscargle_short, guess_acf_fft_short, guess_acf_fft_highpass,
)
from comb_fit import fit_rotation_period  # noqa: E402

DEFAULT_N_WORKERS = 10  # target the 10 P-cores; leave the 4 E-cores free

# method name -> (function, n_guesses). n_guesses=5 for acf_fft_highpass
# matches this project's established convention (it returns candidates
# per smoothing window, so 5 already yields several times that many
# total candidates); 10 for everything else.
METHODS = {
    "pairwise_histogram": (guess_pairwise_histogram, 10),
    "lombscargle": (guess_lombscargle, 10),
    "acf_fft": (guess_acf_fft, 10),
    "wavelet": (guess_wavelet, 10),
    "lombscargle_short": (guess_lombscargle_short, 10),
    "acf_fft_short": (guess_acf_fft_short, 10),
    "acf_fft_highpass": (guess_acf_fft_highpass, 5),
}


def _prep_wavelet_flux(flux: np.ndarray) -> np.ndarray:
    """Fill NaN gaps so guess_wavelet (which requires gap-free input) can
    run on the same preprocessed light curve as the other methods.
    Interior gaps linearly interpolated; edge NaNs nearest-filled.
    """
    s = pd.Series(np.asarray(flux, dtype=float))
    return s.interpolate(limit_direction="both").to_numpy()


def process_one_file(fits_path: str) -> dict:
    """Run every guess_* method, then fit_rotation_period, on one file.

    Parameters
    ----------
    fits_path : str
        Path to a SMARTS-format FITS file.

    Returns
    -------
    dict
        One row: star id, true period, per-method candidate counts and
        failures, pipeline success, recovered period, relative error,
        alias diagnosis.
    """
    star_id = Path(fits_path).stem
    row = dict(
        star_id=star_id, fits_path=str(fits_path), true_period=np.nan,
        success=False, recovered_period=np.nan, redchi=np.nan,
        n_peaks_used=np.nan, message="", rel_error=np.nan,
        alias_relation="", n_candidates_tried=np.nan,
        method_failures="", error="",
    )
    try:
        with fits.open(fits_path) as hdul:
            true_period = hdul[0].header.get("PERIOD", None)
        true_period = float(true_period) if true_period is not None else np.nan
        row["true_period"] = true_period

        pre = load_smarts_fits(fits_path)
        acf_lags, acf = compute_acf(pre.time, pre.flux)
        wavelet_flux = _prep_wavelet_flux(pre.flux)

        all_guesses = []
        failed_methods = {}
        for method_name, (fn, n_guesses) in METHODS.items():
            flux_in = wavelet_flux if method_name == "wavelet" else pre.flux
            try:
                all_guesses.extend(fn(pre.time, flux_in, acf_lags, acf, n_guesses=n_guesses))
            except Exception as exc:  # noqa: BLE001 -- one method failing shouldn't block the rest
                failed_methods[method_name] = f"{type(exc).__name__}: {exc}"

        row["method_failures"] = "; ".join(f"{k}: {v}" for k, v in failed_methods.items())

        result = fit_rotation_period(acf_lags, acf, all_guesses)

        row["success"] = bool(result.success)
        row["message"] = result.message
        row["n_candidates_tried"] = result.n_candidates_tried

        best = result.best_fit
        if best is not None:
            row["recovered_period"] = best.P
            row["redchi"] = best.redchi
            row["n_peaks_used"] = best.n_peaks_used

        if np.isfinite(row["recovered_period"]) and np.isfinite(true_period) and true_period > 0:
            rel_error = (row["recovered_period"] - true_period) / true_period
            row["rel_error"] = rel_error
            for name, factor in [("P/3", 1 / 3), ("P/2", 1 / 2), ("2P", 2.0), ("3P", 3.0)]:
                if abs(row["recovered_period"] - factor * true_period) / true_period < 0.05:
                    row["alias_relation"] = name
                    break

    except Exception as exc:  # noqa: BLE001 -- one file failing shouldn't kill the batch
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_traceback"] = traceback.format_exc()

    return row


def run_baseline(
    data_dir: str, outdir: str, n_workers: int = DEFAULT_N_WORKERS, resume: bool = False,
) -> pd.DataFrame:
    """Run process_one_file over every *.fits in data_dir, in parallel,
    writing results incrementally so a partial run can be resumed.

    Parameters
    ----------
    data_dir : str
        Directory containing FITS files.
    outdir : str
        Directory to write baseline_results_full.csv (created if needed).
    n_workers : int
        Number of worker processes (default targets the 10 P-cores).
    resume : bool
        If True and baseline_results_full.csv already exists in outdir,
        skip any star_id already present in it and append new results --
        useful for continuing a run that was interrupted (e.g. hit a
        wall-clock limit) partway through.

    Returns
    -------
    pd.DataFrame
        One row per file (all files, not just this call's new ones).
    """
    fits_paths = sorted(Path(data_dir).glob("*.fits"))
    if not fits_paths:
        raise FileNotFoundError(f"No .fits files found in {data_dir}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "baseline_results_full.csv"

    done_ids = set()
    existing_rows = []
    if resume and out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        existing_rows = existing_df.to_dict("records")
        done_ids = set(existing_df["star_id"])
        fits_paths = [p for p in fits_paths if p.stem not in done_ids]
        print(f"Resuming: {len(done_ids)} already done, {len(fits_paths)} remaining.")

    print(f"Running full-method-set pipeline on {len(fits_paths)} files using {n_workers} worker(s)...")
    rows = list(existing_rows)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_file, str(p)): p for p in fits_paths}
        for i, future in enumerate(as_completed(futures), start=1):
            p = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(star_id=p.stem, fits_path=str(p), error=str(exc)))
            print(f"  [{i}/{len(fits_paths)}] {p.name}", flush=True)
            # write incrementally so a wall-clock cutoff doesn't lose progress
            pd.DataFrame(rows).sort_values("true_period").reset_index(drop=True).to_csv(
                out_csv, index=False
            )

    df = pd.DataFrame(rows).sort_values("true_period").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df)} total rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="directory of FITS files")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    ap.add_argument("--resume", action="store_true", help="skip files already in the output CSV")
    args = ap.parse_args()
    run_baseline(args.data_dir, args.outdir, n_workers=args.n_workers, resume=args.resume)
