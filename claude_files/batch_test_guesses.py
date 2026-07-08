#!/usr/bin/env python3
"""
batch_test_guesses.py

Batch-runs every guess_* candidate-generation function found in your
comb_fit.py module against a list of SMARTS light curves, WITHOUT calling
fit_rotation_period (no fitting, no gate-checking, no arbitration between
methods). For each light curve x method, it keeps the top-N candidates and
records whether/where the true (injected) period appears among them.

This is meant to answer: "considered purely as a candidate-generation
step, how good is each guess_* function at putting the right period
somewhere in its short list?" -- independent of how well the downstream
joint comb fit later cleans things up.

Assumes preprocessing.py, acf_utils.py, and comb_fit.py (your existing
modules) are importable, i.e. this script is run from the same directory
as those files, or they're on your PYTHONPATH.

Usage
-----
    # from a text file with one FITS path per line
    python batch_test_guesses.py --files file_list.txt --outdir results/

    # or from a glob pattern
    python batch_test_guesses.py --glob "/data/smarts_lcs/*.fits" --outdir results/

Outputs (written to --outdir)
------------------------------
    candidates.csv         : one row per (light curve, method, rank<=N)
    summary.csv            : one row per (light curve, method) -- was the
                              true period found, and at what rank
    failures.csv            : one row per (light curve[, method]) that
                              raised an exception, with the error message
    hit_rate_by_method.png  : bar chart of overall hit rate per method
    rank_distribution.png   : histogram of the rank the true period landed
                              at, among hits, split by method
    hit_rate_vs_period.png  : hit rate vs. true period (log-binned), split
                              by method -- shows which period regimes each
                              method struggles with
"""

from __future__ import annotations

import argparse
import glob as globmod
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from preprocessing import load_smarts_fits
from acf_utils import compute_acf
import comb_fit

HEADER_KEYS = [
    "PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL",
    "MINLAT", "MAXLAT", "DIFFROT", "TSPOT", "BFLY",
]

# every guess_* function in comb_fit.py shares this call signature:
# fn(time, flux, acf_lags, acf, ..., n_guesses=...) -> list[InitialGuess]
REQUIRED_ARGS = {"time", "flux", "acf_lags", "acf"}


def discover_guess_functions(module=comb_fit) -> dict:
    """Auto-discover every guess_* candidate-generation function in `module`.

    Rather than hardcoding a fixed list of method names, this inspects
    `module` for functions named 'guess_*' whose signature accepts the
    standard (time, flux, acf_lags, acf, ...) call pattern shared by every
    guess_* function in comb_fit.py. That way, if you add a 5th (or 6th)
    method later, this script picks it up automatically with no changes.

    Parameters
    ----------
    module : the module to search (default: comb_fit).

    Returns
    -------
    dict[str, callable]
        Keyed by the part of the function name after 'guess_' (e.g.
        'lombscargle'), sorted alphabetically by that key.
    """
    found = {}
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("guess_"):
            continue
        params = set(inspect.signature(fn).parameters)
        if REQUIRED_ARGS.issubset(params):
            found[name[len("guess_"):]] = fn
    return dict(sorted(found.items()))


def read_header_values(fits_path, keys=HEADER_KEYS) -> dict:
    """Read the requested primary-header keyword values from a SMARTS FITS file.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file.
    keys : list of str
        Header keywords to extract.

    Returns
    -------
    dict
        Maps each requested key to its value (None if missing).
    """
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        return {k: header.get(k, None) for k in keys}


def _prep_wavelet_flux(flux: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in `flux` so guess_wavelet (which requires gap-free
    input) can be run on the same preprocessed light curve as the other
    guess_* methods, instead of being skipped entirely.

    Interior gaps are linearly interpolated; any leading/trailing NaNs
    (which linear interpolation can't reach) are filled by nearest-value
    extension.

    Parameters
    ----------
    flux : np.ndarray
        Flux array, possibly containing NaNs at missing cadences.

    Returns
    -------
    np.ndarray
        Same length as `flux`, with all NaNs filled.
    """
    s = pd.Series(np.asarray(flux, dtype=float))
    return s.interpolate(limit_direction="both").to_numpy()


def rank_of_true_period(guesses: list, true_period, rel_tol: float = 0.03):
    """Check whether/where the true period appears in one method's ranked candidate list.

    Parameters
    ----------
    guesses : list[InitialGuess]
        Sorted best-first (rank 1, 2, ...), as returned by any guess_*
        function.
    true_period : float or None
        Known injected period (days) to check against.
    rel_tol : float
        Relative tolerance for declaring a candidate a match to
        `true_period`. Default 0.03 (3%), matching this codebase's
        `label_rel_tol` default in ml_features.extract_candidate_features.

    Returns
    -------
    found : bool
        True if any candidate matches `true_period` within `rel_tol`.
    rank : int or None
        The `rank` of the best (lowest-rank-number) matching candidate,
        or None if no match was found.
    matched_P0 : float or None
        The P0 of that matching candidate.
    """
    if true_period is None or not np.isfinite(true_period):
        return False, None, None
    for g in sorted(guesses, key=lambda g: g.rank):
        if abs(g.P0 - true_period) / true_period <= rel_tol:
            return True, g.rank, g.P0
    return False, None, None


def process_one_lightcurve(
    fits_path,
    guess_fns: dict,
    n_guesses: int = 10,
    rel_tol: float = 0.03,
    **load_kwargs,
):
    """Run every discovered guess_* function on one SMARTS light curve.

    Loads and preprocesses the FITS file with `load_smarts_fits`, computes
    a gap-aware ACF with `compute_acf`, then calls every function in
    `guess_fns` directly -- fit_rotation_period is never called, so there
    is no fitting, gating, or cross-method arbitration here, only raw
    candidate generation.

    Parameters
    ----------
    fits_path : str or Path
        Path to a SMARTS-format FITS file.
    guess_fns : dict[str, callable]
        As returned by discover_guess_functions().
    n_guesses : int
        How many top candidates to request/keep per method.
    rel_tol : float
        Relative tolerance for matching a candidate to the true period.
    **load_kwargs
        Forwarded to load_smarts_fits (e.g. sectors=[1, 2, 6, 7, 12]).

    Returns
    -------
    candidate_rows : list[dict]
        One row per (method, rank) candidate produced.
    summary_rows : list[dict]
        One row per method: found/rank summary.
    failure_rows : list[dict]
        One row per method that raised an exception.
    header_vals : dict
        The requested FITS header keyword values.
    true_period : float or None
    """
    star_id = Path(fits_path).stem
    header_vals = read_header_values(fits_path)
    true_period = header_vals.get("PERIOD", None)
    if true_period is not None:
        true_period = float(true_period)

    pre = load_smarts_fits(fits_path, **load_kwargs)
    acf_lags, acf = compute_acf(pre.time, pre.flux)

    candidate_rows, summary_rows, failure_rows = [], [], []

    for method_name, fn in guess_fns.items():
        flux_in = pre.flux
        if method_name == "wavelet":
            # guess_wavelet requires gap-free flux; interpolate rather than
            # silently skipping this method for every gappy light curve.
            flux_in = _prep_wavelet_flux(pre.flux)
        try:
            guesses = fn(pre.time, flux_in, acf_lags, acf, n_guesses=n_guesses)
        except Exception as exc:  # noqa: BLE001 -- one method failing shouldn't stop the batch
            failure_rows.append(dict(
                star_id=star_id, method=method_name,
                error=f"{type(exc).__name__}: {exc}",
            ))
            summary_rows.append(dict(
                star_id=star_id, method=method_name, n_candidates=0,
                found_true_period=False, rank_of_true_period=np.nan,
                matched_P0=np.nan, true_period=true_period,
            ))
            continue

        found, rank, matched_P0 = rank_of_true_period(guesses, true_period, rel_tol=rel_tol)
        summary_rows.append(dict(
            star_id=star_id, method=method_name, n_candidates=len(guesses),
            found_true_period=found, rank_of_true_period=rank,
            matched_P0=matched_P0, true_period=true_period,
        ))
        for g in guesses:
            is_match = (
                true_period is not None and np.isfinite(true_period)
                and abs(g.P0 - true_period) / true_period <= rel_tol
            )
            candidate_rows.append(dict(
                star_id=star_id, method=method_name, rank=g.rank,
                P0=g.P0, strength=g.strength, true_period=true_period,
                is_match=is_match,
            ))

    return candidate_rows, summary_rows, failure_rows, header_vals, true_period


def run_batch(fits_paths, outdir, n_guesses: int = 10, rel_tol: float = 0.03, **load_kwargs):
    """Run process_one_lightcurve over a list of FITS files and save the combined results.

    Parameters
    ----------
    fits_paths : list of str or Path
        Paths to SMARTS FITS files.
    outdir : str or Path
        Directory to write candidates.csv / summary.csv / failures.csv
        (created if it doesn't exist).
    n_guesses : int
        Top-N candidates requested/kept per method.
    rel_tol : float
        Relative tolerance for matching a candidate to the true period.
    **load_kwargs
        Forwarded to load_smarts_fits for every file (e.g. sectors=[...]).

    Returns
    -------
    candidates_df, summary_df, failures_df : pd.DataFrame
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    guess_fns = discover_guess_functions()
    if not guess_fns:
        raise RuntimeError(
            "No guess_* functions found in comb_fit.py matching the "
            "expected (time, flux, acf_lags, acf, ...) signature."
        )
    print(f"Discovered {len(guess_fns)} guess_* function(s): {list(guess_fns)}")

    all_candidates, all_summary, all_failures = [], [], []
    n_ok, n_err = 0, 0

    for i, fits_path in enumerate(fits_paths, start=1):
        try:
            cand, summ, fail, header_vals, _true_period = process_one_lightcurve(
                fits_path, guess_fns, n_guesses=n_guesses, rel_tol=rel_tol, **load_kwargs
            )
            for row in cand:
                row.update(header_vals)
            for row in summ:
                row.update(header_vals)
            all_candidates.extend(cand)
            all_summary.extend(summ)
            all_failures.extend(fail)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 -- a whole-file failure shouldn't stop the batch
            n_err += 1
            all_failures.append(dict(
                star_id=Path(fits_path).stem, method="__load_or_acf__",
                error=f"{type(exc).__name__}: {exc}",
            ))
            traceback.print_exc(file=sys.stderr)

        if i % 50 == 0 or i == len(fits_paths):
            print(f"  processed {i}/{len(fits_paths)} files "
                  f"({n_ok} ok, {n_err} failed to load/compute ACF)")

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
    """Save diagnostic plots summarizing each guess_* method's hit rate and failure modes.

    Produces three PNGs in `outdir`:
      - hit_rate_by_method.png : overall fraction of light curves where
        each method's top-N candidates included the true period.
      - rank_distribution.png : histogram of the rank the true period
        landed at (among hits only), split by method -- a method that
        "finds" the period but always at rank 8-10 is much less useful
        than one that finds it at rank 1-2.
      - hit_rate_vs_period.png : hit rate vs. true period (log-binned),
        split by method -- shows which period regimes (short/fast
        rotators vs. long/slow rotators) each method struggles with.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The per-(star, method) summary dataframe from run_batch.
    outdir : str or Path
        Directory to save the .png files to.
    """
    outdir = Path(outdir)
    methods = sorted(summary_df["method"].unique())

    # --- hit rate by method ---
    hit_rate = summary_df.groupby("method")["found_true_period"].mean().reindex(methods)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(hit_rate.index, hit_rate.values, color="steelblue")
    ax.set_ylabel("fraction of LCs where true period was in top-N")
    ax.set_ylim(0, 1.08)
    ax.set_title("guess_* hit rate (true period in candidate pool)")
    for i, v in enumerate(hit_rate.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "hit_rate_by_method.png", dpi=150)
    plt.close(fig)

    # --- rank distribution among hits ---
    hits = summary_df[summary_df["found_true_period"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(hits):
        max_rank = int(hits["rank_of_true_period"].max())
        bins = np.arange(0.5, max_rank + 1.5, 1)
        for m in methods:
            sub = hits.loc[hits["method"] == m, "rank_of_true_period"]
            if len(sub):
                ax.hist(sub, bins=bins, alpha=0.5, label=m)
        ax.set_xlabel("rank at which the true period appeared")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    ax.set_title("Rank of true period among hits, by method")
    fig.tight_layout()
    fig.savefig(outdir / "rank_distribution.png", dpi=150)
    plt.close(fig)

    # --- hit rate vs true period (log-binned) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = summary_df.dropna(subset=["true_period"]).copy()
    if len(valid):
        bin_edges = np.geomspace(
            max(valid["true_period"].min(), 0.1), valid["true_period"].max(), 9
        )
        valid["period_bin"] = pd.cut(valid["true_period"], bin_edges)
        for m in methods:
            sub = valid[valid["method"] == m]
            rate = sub.groupby("period_bin", observed=True)["found_true_period"].mean()
            centers = [iv.mid for iv in rate.index]
            ax.plot(centers, rate.values, marker="o", label=m)
        ax.set_xscale("log")
        ax.set_xlabel("true period (days)")
        ax.set_ylabel("hit rate")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    ax.set_title("Hit rate vs. true period, by method\n(where does each method fail?)")
    fig.tight_layout()
    fig.savefig(outdir / "hit_rate_vs_period.png", dpi=150)
    plt.close(fig)


def _resolve_fits_paths(files_arg, glob_arg):
    """Resolve the --files or --glob CLI argument into a list of FITS paths.

    Parameters
    ----------
    files_arg : str or None
        Path to a text file listing one FITS path per line.
    glob_arg : str or None
        Glob pattern, e.g. '/data/smarts_lcs/*.fits'.

    Returns
    -------
    list[str]
    """
    if files_arg:
        return [line.strip() for line in Path(files_arg).read_text().splitlines() if line.strip()]
    return sorted(globmod.glob(glob_arg))


def main():
    """Command-line entry point: parse args, resolve the file list, run the batch."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", type=str, help="text file with one FITS path per line")
    src.add_argument("--glob", type=str, help="glob pattern for FITS files, e.g. '/data/*.fits'")
    p.add_argument("--outdir", type=str, required=True, help="output directory")
    p.add_argument("--n-guesses", type=int, default=10, help="top-N candidates per method (default 10)")
    p.add_argument("--rel-tol", type=float, default=0.03, help="relative match tolerance (default 0.03)")
    p.add_argument(
        "--sectors", type=str, default=None,
        help="comma-separated sector list to keep, e.g. '1,2,6,7,12' (default: all available)",
    )
    args = p.parse_args()

    fits_paths = _resolve_fits_paths(args.files, args.glob)
    print(f"Found {len(fits_paths)} FITS files to process.")
    if not fits_paths:
        sys.exit("No FITS files found -- check --files/--glob.")

    load_kwargs = {}
    if args.sectors:
        load_kwargs["sectors"] = [int(s) for s in args.sectors.split(",")]

    run_batch(
        fits_paths, args.outdir,
        n_guesses=args.n_guesses, rel_tol=args.rel_tol, **load_kwargs,
    )


if __name__ == "__main__":
    main()
