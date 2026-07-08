"""
example_from_fits.py

Full pipeline starting from a raw SMARTS FITS file:
    1. load_smarts_fits (preprocessing.py) -- re-detect and remove the
       original per-orbit gaps, drop bad orbits, optionally keep only a
       hand-picked subset of sectors, regrid to even cadence.
    2. compute_acf (acf_utils.py) -- gap-aware ACF.
    3. gather_initial_guesses + fit_rotation_period (comb_fit.py) --
       candidate generation, joint fit, and reliability-gated arbitration.
    4. extract_candidate_features (ml_features.py) -- flatten into a
       training-ready feature table (with a label, since we know the
       true injected period).
    5. Diagnostic plots.
"""

import numpy as np
from preprocessing import load_smarts_fits, plot_split_diagnostics
from acf_utils import compute_acf
from comb_fit import gather_initial_guesses, fit_rotation_period
from ml_features import extract_candidate_features
from plotting import plot_full_diagnostic


if __name__ == "__main__":
    fits_path = "/mnt/user-data/uploads/smarts-tess-v1_0-025300.fits"

    # --- 1. preprocess: FITS -> pipeline-ready (time, flux) ---
    # `sectors=[...]` could be added here to deliberately degrade this to a
    # sparser, more realistic TESS-like observing pattern, e.g.
    # sectors=[1, 2, 6, 7, 12]. Left as None to keep every good sector.
    pre = load_smarts_fits(fits_path)
    print(f"true_period: {pre.true_period:.4f} d")
    print(f"orbits: {pre.n_orbits_total} total, {pre.n_orbits_flagged} flagged/dropped")
    print(f"sectors used: {pre.sectors_used}")
    print(f"regridded: {len(pre.time)} points, "
          f"{np.isnan(pre.flux).mean():.1%} missing")

    # --- 2. gap-aware ACF ---
    acf_lags, acf = compute_acf(pre.time, pre.flux)

    # --- 3. candidate generation + fitting/arbitration ---
    guesses, failed_methods = gather_initial_guesses(pre.time, pre.flux, acf_lags, acf, n_guesses=5)
    result = fit_rotation_period(acf_lags, acf, guesses)
    print(f"\n{result.message}")
    if result.success:
        print(f"recovered P = {result.best_fit.P:.4f} d "
              f"(true = {pre.true_period:.4f}, "
              f"ratio = {result.best_fit.P / pre.true_period:.3f})")

    # --- 4. training-ready feature table (true_period known -> labeled) ---
    df = extract_candidate_features(
        result, guesses, pre.time, pre.flux, acf_lags, acf,
        star_id=fits_path, true_period=pre.true_period,
    )
    print(f"\nfeature table: {df.shape[0]} candidates x {df.shape[1]} columns")

    # --- 5. diagnostic plots ---
    # plot_split_diagnostics needs the RAW (pre-gap-removal) time/flux, not
    # pre.time/pre.flux -- re-read those directly from the FITS file.
    from astropy.io import fits
    with fits.open(fits_path) as hdul:
        lc = hdul["LIGHTCURVE"].data
        raw_time = np.asarray(lc["time"], dtype=float)
        raw_flux = np.asarray(lc["flux"], dtype=float) - 1.0
    fig1, axs1 = plot_split_diagnostics(raw_time, raw_flux, pre.diagnostics)
    fig1.savefig("from_fits_split_diagnostic.png", dpi=120)

    fig2, axes2 = plot_full_diagnostic(acf_lags, acf, guesses, result)
    fig2.savefig("from_fits_diagnostic.png", dpi=120)
    print("\nSaved diagnostic figures to from_fits_split_diagnostic.png "
          "and from_fits_diagnostic.png")
