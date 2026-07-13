#!/usr/bin/env python3
"""
batch_test_acf_fft_highpass.py

Batch-runs guess_acf_fft_highpass (comb_fit.py) across a list of SMARTS
light curves, recording which smoothing window produced each candidate,
so you can check afterward which window recovers the true period best in
which true-period range.

IMPORTANT -- matches the real guess_acf_fft_highpass design, which is NOT
"one call per window": a single call internally loops over every window in
`smooth_windows` and returns all of their candidates concatenated
together, with each candidate's `method` already tagged by its
originating window (e.g. "acf_fft_hp5d", "acf_fft_hp20d"). This script
calls it exactly once per light curve and parses the window back out of
that tag -- it does not call the function once per window itself.

One consequence of that internal design: guess_acf_fft_highpass swallows
per-window exceptions itself (a bad window is silently skipped rather than
raised -- see its docstring), so failures.csv here only captures whole-
light-curve failures (e.g. load_smarts_fits/compute_acf errors), not
per-window failures inside an otherwise-successful call. A window that
contributed zero candidates for a given star is indistinguishable, from
this script's point of view, between "failed internally" and "found
genuinely nothing" -- both just show up as n_candidates=0 for that
(star, window) row in summary.csv.

Also note: rank is window-local, not global. guess_acf_fft_highpass calls
guess_acf_fft once per window, and each of those calls ranks its own top
n_guesses starting at rank=1 -- so "rank 1" appears once per window in the
concatenated output, not once overall. This script computes
found/rank-of-true-period per (star, window) group accordingly, never by
ranking across windows.

fit_rotation_period is never called: pure candidate-generation
benchmarking, same as batch_test_guesses.py.

Usage
-----
    python batch_test_acf_fft_highpass.py \\
        --files file_list.txt --outdir results_highpass/

    # override the window sweep or search band:
    python batch_test_acf_fft_highpass.py --files file_list.txt \\
        --outdir results_highpass/ --smooth-windows 1,2,3,5,8,12,20,40 \\
        --max-period 20

Outputs (written to --outdir)
------------------------------
    candidates.csv : one row per (light curve, window, rank<=n_guesses)
    summary.csv    : one row per (light curve, window) -- was the true
                      period found by that window, and at what
                      (window-local) rank
    failures.csv   : one row per light curve that raised an exception at
                      the load/ACF/whole-call level (see note above)
    hit_rate_by_window.png       : overall hit rate vs smoothing window
    hit_rate_window_x_period.png : heatmap of hit rate, smoothing window
                                    x true-period band
"""

from __future__ import annotations

import argparse
import glob as globmod
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from preprocessing import load_smarts_fits
from acf_utils import compute_acf
from comb_fit import guess_acf_fft_highpass

HEADER_KEYS = [
    "PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL",
    "MINLAT", "MAXLAT", "DIFFROT", "TSPOT", "BFLY",
]
# matches guess_acf_fft_highpass's own default smooth_windows
DEFAULT_SMOOTH_WINDOWS = (2.0, 5.0, 10.0, 20.0, 40.0)
DEFAULT_MAX_PERIOD = 50.0  # matches guess_acf_fft_highpass's own default
PERIOD_BAND_EDGES = [0, 1, 5, 10, 15, 20, 30, 50, np.inf]
PERIOD_BAND_LABELS = ["<1d", "1-5d", "5-10d", "10-15d", "15-20d", "20-30d", "30-50d", ">50d"]
# parses the window back out of a tag like "acf_fft_hp5d" or "acf_fft_hp2.5d"
_WINDOW_TAG_RE = re.compile(r"^acf_fft_hp([\d.]+)d$")


def parse_window_from_method(method: str) -> float:
    """Recover the smoothing window (days) from a guess_acf_fft_highpass
    method tag like "acf_fft_hp5d".

    Parameters
    ----------
    method : the `method` field of an InitialGuess returned by
        guess_acf_fft_highpass.

    Returns
    -------
    float

    Raises
    ------
    ValueError if `method` doesn't match the expected "acf_fft_hp{W}d" pattern.
    """
    m = _WINDOW_TAG_RE.match(method)
    if m is None:
        raise ValueError(f"method tag {method!r} doesn't match acf_fft_hp<window>d")
    return float(m.group(1))


def read_header_values(fits_path, keys=HEADER_KEYS) -> dict:
    """Read the requested primary-header keyword values from a SMARTS FITS file."""
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        return {k: header.get(k, None) for k in keys}


def rank_of_true_period_in_group(guesses: list, true_period, rel_tol: float = 0.15):
    """Check whether/where the true period appears within ONE window's
    candidate group (i.e. `guesses` should already be filtered to a
    single smoothing window before calling this -- rank is window-local).

    Parameters
    ----------
    guesses : list[InitialGuess], all sharing the same method/window tag.
    true_period : float or None.
    rel_tol : relative match tolerance. Default 0.15, matching this
        codebase's current batch_test_guesses.py convention.

    Returns
    -------
    found : bool
    rank : int or None (window-local rank, 1 = strongest for that window)
    matched_P0 : float or None
    """
    if true_period is None or not np.isfinite(true_period):
        return False, None, None
    for g in sorted(guesses, key=lambda g: g.rank):
        if abs(g.P0 - true_period) / true_period <= rel_tol:
            return True, g.rank, g.P0
    return False, None, None


def process_one_lightcurve(
    fits_path,
    smooth_windows: tuple,
    max_period: float,
    min_period,
    n_guesses: int,
    oversample: int,
    rel_tol: float = 0.15,
    **load_kwargs,
):
    """Run guess_acf_fft_highpass once on one light curve, then split its
    concatenated output back into per-window groups for scoring.

    Parameters
    ----------
    fits_path : path to a SMARTS-format FITS file.
    smooth_windows, max_period, min_period, n_guesses, oversample :
        forwarded to guess_acf_fft_highpass.
    rel_tol : relative tolerance for matching a candidate to the true period.
    **load_kwargs : forwarded to load_smarts_fits.

    Returns
    -------
    candidate_rows, summary_rows : list[dict]
    header_vals : dict
    """
    star_id = Path(fits_path).stem
    header_vals = read_header_values(fits_path)
    true_period = header_vals.get("PERIOD", None)
    if true_period is not None:
        true_period = float(true_period)

    pre = load_smarts_fits(fits_path, **load_kwargs)
    acf_lags, acf = compute_acf(pre.time, pre.flux)  # unfiltered ACF; passed through for interface consistency only

    guesses = guess_acf_fft_highpass(
        pre.time, pre.flux, acf_lags, acf,
        smooth_windows=smooth_windows, max_period=max_period, min_period=min_period,
        n_guesses=n_guesses, oversample=oversample,
    )

    # split the concatenated output back into per-window groups
    by_window: dict = {w: [] for w in smooth_windows}
    for g in guesses:
        w = parse_window_from_method(g.method)
        by_window.setdefault(w, []).append(g)

    candidate_rows, summary_rows = [], []
    for w, wguesses in by_window.items():
        found, rank, matched_P0 = rank_of_true_period_in_group(wguesses, true_period, rel_tol=rel_tol)
        summary_rows.append(dict(
            star_id=star_id, smoothing_window=w, n_candidates=len(wguesses),
            found_true_period=found, rank_of_true_period=rank,
            matched_P0=matched_P0, true_period=true_period,
        ))
        for g in wguesses:
            is_match = (
                true_period is not None and np.isfinite(true_period)
                and abs(g.P0 - true_period) / true_period <= rel_tol
            )
            candidate_rows.append(dict(
                star_id=star_id, smoothing_window=w, method=g.method, rank=g.rank,
                P0=g.P0, strength=g.strength, true_period=true_period,
                is_match=is_match,
            ))

    return candidate_rows, summary_rows, header_vals


def run_batch(
    fits_paths, outdir,
    smooth_windows: tuple = DEFAULT_SMOOTH_WINDOWS,
    max_period: float = DEFAULT_MAX_PERIOD,
    min_period=None,
    n_guesses: int = 5,
    oversample: int = 8,
    rel_tol: float = 0.15,
    **load_kwargs,
):
    """Run process_one_lightcurve over a list of FITS files and save the combined results.

    Parameters mirror guess_acf_fft_highpass's own; see module docstring
    for output files.

    Returns
    -------
    candidates_df, summary_df, failures_df : pd.DataFrame
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"smooth_windows={smooth_windows}  max_period={max_period}  "
          f"min_period={min_period}  n_guesses={n_guesses}  oversample={oversample}")

    all_candidates, all_summary, all_failures = [], [], []
    n_ok, n_err = 0, 0

    for i, fits_path in enumerate(fits_paths, start=1):
        try:
            cand, summ, header_vals = process_one_lightcurve(
                fits_path, smooth_windows, max_period, min_period,
                n_guesses, oversample, rel_tol=rel_tol, **load_kwargs,
            )
            for row in cand:
                row.update(header_vals)
            for row in summ:
                row.update(header_vals)
            all_candidates.extend(cand)
            all_summary.extend(summ)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 -- a whole-file failure shouldn't stop the batch
            n_err += 1
            all_failures.append(dict(
                star_id=Path(fits_path).stem,
                error=f"{type(exc).__name__}: {exc}",
            ))
            traceback.print_exc(file=sys.stderr)

        if i % 50 == 0 or i == len(fits_paths):
            print(f"  processed {i}/{len(fits_paths)} files "
                  f"({n_ok} ok, {n_err} failed to load/compute ACF/run)")

    candidates_df = pd.DataFrame(all_candidates)
    summary_df = pd.DataFrame(all_summary)
    failures_df = pd.DataFrame(all_failures)

    candidates_df.to_csv(outdir / "candidates.csv", index=False)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    failures_df.to_csv(outdir / "failures.csv", index=False)
    print(f"\nSaved: {outdir/'candidates.csv'}, {outdir/'summary.csv'}, {outdir/'failures.csv'}")

    if len(summary_df):
        make_diagnostic_plots(summary_df, outdir)

    return candidates_df, summary_df, failures_df


def make_diagnostic_plots(summary_df: pd.DataFrame, outdir):
    """Save diagnostic plots for choosing the best smoothing window.

    Produces:
      - hit_rate_by_window.png : overall hit rate vs smoothing_window.
      - hit_rate_window_x_period.png : heatmap of hit rate over
        (smoothing_window x true-period band).
    """
    outdir = Path(outdir)
    windows = sorted(summary_df["smoothing_window"].unique())

    hit_rate = summary_df.groupby("smoothing_window")["found_true_period"].mean().reindex(windows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(hit_rate.index, hit_rate.values, marker="o", color="steelblue")
    ax.set_xlabel("smoothing_window (days)")
    ax.set_ylabel("hit rate (true period in top-N for that window)")
    ax.set_ylim(0, 1.05)
    ax.set_title("guess_acf_fft_highpass: hit rate vs smoothing window")
    fig.tight_layout()
    fig.savefig(outdir / "hit_rate_by_window.png", dpi=150)
    plt.close(fig)

    valid = summary_df.dropna(subset=["true_period"]).copy()
    if len(valid):
        valid["period_band"] = pd.cut(
            valid["true_period"], bins=PERIOD_BAND_EDGES, labels=PERIOD_BAND_LABELS
        )
        pivot = valid.pivot_table(
            index="period_band", columns="smoothing_window",
            values="found_true_period", aggfunc="mean", observed=True,
        ).reindex(index=PERIOD_BAND_LABELS, columns=windows)

        fig, ax = plt.subplots(figsize=(1.1 * len(windows) + 2, 4.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(windows)
        ax.set_yticks(range(len(PERIOD_BAND_LABELS)))
        ax.set_yticklabels(PERIOD_BAND_LABELS)
        ax.set_xlabel("smoothing_window (days)")
        ax.set_ylabel("true period band")
        ax.set_title("hit rate: smoothing window x true-period range")
        for yi in range(pivot.shape[0]):
            for xi in range(pivot.shape[1]):
                v = pivot.values[yi, xi]
                if np.isfinite(v):
                    ax.text(xi, yi, f"{v:.2f}", ha="center", va="center",
                             color="white" if v < 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="hit rate")
        fig.tight_layout()
        fig.savefig(outdir / "hit_rate_window_x_period.png", dpi=150)
        plt.close(fig)


def _resolve_fits_paths(files_arg, glob_arg):
    """Resolve the --files or --glob CLI argument into a list of FITS paths."""
    if files_arg:
        return [line.strip() for line in Path(files_arg).read_text().splitlines() if line.strip()]
    return sorted(globmod.glob(glob_arg))


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", type=str, help="text file with one FITS path per line")
    src.add_argument("--glob", type=str, help="glob pattern for FITS files, e.g. '/data/*.fits'")
    p.add_argument("--outdir", type=str, required=True, help="output directory")
    p.add_argument(
        "--smooth-windows", type=str, default=None,
        help=f"comma-separated smoothing windows in days "
             f"(default: {','.join(str(w) for w in DEFAULT_SMOOTH_WINDOWS)}, "
             f"matching guess_acf_fft_highpass's own default)",
    )
    p.add_argument("--max-period", type=float, default=DEFAULT_MAX_PERIOD,
                    help=f"forwarded to guess_acf_fft_highpass (default {DEFAULT_MAX_PERIOD})")
    p.add_argument("--min-period", type=float, default=None,
                    help="forwarded to guess_acf_fft_highpass (default: its own default)")
    p.add_argument("--n-guesses", type=int, default=5,
                    help="top-N candidates per window (default 5, matching guess_acf_fft_highpass)")
    p.add_argument("--oversample", type=int, default=8,
                    help="forwarded to guess_acf_fft_highpass (default 8)")
    p.add_argument("--rel-tol", type=float, default=0.15, help="relative match tolerance (default 0.15)")
    p.add_argument(
        "--sectors", type=str, default=None,
        help="comma-separated sector list to keep, e.g. '1,2,6,7,12' (default: all available)",
    )
    args = p.parse_args()

    fits_paths = _resolve_fits_paths(args.files, args.glob)
    print(f"Found {len(fits_paths)} FITS files to process.")
    if not fits_paths:
        sys.exit("No FITS files found -- check --files/--glob.")

    smooth_windows = (
        tuple(float(w) for w in args.smooth_windows.split(","))
        if args.smooth_windows else DEFAULT_SMOOTH_WINDOWS
    )

    load_kwargs = {}
    if args.sectors:
        load_kwargs["sectors"] = [int(s) for s in args.sectors.split(",")]

    run_batch(
        fits_paths, args.outdir,
        smooth_windows=smooth_windows, max_period=args.max_period, min_period=args.min_period,
        n_guesses=args.n_guesses, oversample=args.oversample, rel_tol=args.rel_tol,
        **load_kwargs,
    )


if __name__ == "__main__":
    main()
