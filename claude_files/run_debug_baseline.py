"""
run_debug_baseline.py

Run the full rotation-period pipeline (candidate generation via
gather_initial_guesses, then joint comb fit + arbitration via
fit_rotation_period) over every FITS file in a directory, in parallel,
and save a per-file results table.

Unlike batch_test_guesses.py (which only exercises Stage 1 candidate
generation to measure each guess_* method's hit rate), this script runs
the complete pipeline end to end, so it reflects what the algorithm would
actually output for a given light curve -- including the fit_rotation_period
gates and success=False cases.

Usage
-----
    python run_debug_baseline.py --data-dir debug_set --outdir baseline_results
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
from guesses import gather_initial_guesses  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402

DEFAULT_N_WORKERS = 10  # target the 10 P-cores; leave the 4 E-cores free


def process_one_file(fits_path: str) -> dict:
    """Run the full pipeline on one FITS file and summarize the outcome.

    Parameters
    ----------
    fits_path : str
        Path to a SMARTS-format FITS file.

    Returns
    -------
    dict
        One row of results: star id, true period, whether the pipeline
        succeeded, the recovered period (if any), relative error against
        the true period, harmonic-alias diagnosis (P/2, 2P, etc.), and
        any error encountered.
    """
    star_id = Path(fits_path).stem
    row = dict(
        star_id=star_id, fits_path=str(fits_path), true_period=np.nan,
        success=False, recovered_period=np.nan, redchi=np.nan,
        n_peaks_used=np.nan, message="", rel_error=np.nan,
        alias_relation="", error="",
    )
    try:
        with fits.open(fits_path) as hdul:
            true_period = hdul[0].header.get("PERIOD", None)
        true_period = float(true_period) if true_period is not None else np.nan
        row["true_period"] = true_period

        pre = load_smarts_fits(fits_path)
        acf_lags, acf = compute_acf(pre.time, pre.flux)

        guesses, failed_methods = gather_initial_guesses(pre.time, pre.flux, acf_lags, acf)
        result = fit_rotation_period(acf_lags, acf, guesses)

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
            # flag common harmonic aliases (half period, double period, etc.)
            for name, factor in [("P/3", 1 / 3), ("P/2", 1 / 2), ("2P", 2.0), ("3P", 3.0)]:
                if abs(row["recovered_period"] - factor * true_period) / true_period < 0.05:
                    row["alias_relation"] = name
                    break

    except Exception as exc:  # noqa: BLE001 -- one file failing shouldn't kill the batch
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_traceback"] = traceback.format_exc()

    return row


def run_baseline(data_dir: str, outdir: str, n_workers: int = DEFAULT_N_WORKERS) -> pd.DataFrame:
    """Run process_one_file over every *.fits in data_dir, in parallel.

    Parameters
    ----------
    data_dir : str
        Directory containing FITS files.
    outdir : str
        Directory to write baseline_results.csv (created if needed).
    n_workers : int
        Number of worker processes (default targets the 10 P-cores).

    Returns
    -------
    pd.DataFrame
        One row per file.
    """
    fits_paths = sorted(Path(data_dir).glob("*.fits"))
    if not fits_paths:
        raise FileNotFoundError(f"No .fits files found in {data_dir}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running full pipeline on {len(fits_paths)} files using {n_workers} worker(s)...")
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_file, str(p)): p for p in fits_paths}
        for i, future in enumerate(as_completed(futures), start=1):
            p = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(star_id=p.stem, fits_path=str(p), error=str(exc)))
            print(f"  [{i}/{len(fits_paths)}] {p.name}")

    df = pd.DataFrame(rows).sort_values("true_period").reset_index(drop=True)
    df.to_csv(outdir / "baseline_results.csv", index=False)
    print(f"\nSaved: {outdir / 'baseline_results.csv'}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="directory of FITS files")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    args = ap.parse_args()
    run_baseline(args.data_dir, args.outdir, n_workers=args.n_workers)
