"""
example_messy_data.py

Recommended workflow for real, messy light curves (gaps, weak/noisy
rotation signals) using the v2 comb_fit.py API:

    1. compute_acf (acf_utils) -- gap-aware, handles NaN-filled cadences
       (e.g. TESS downlink/momentum-dump gaps) without needing to
       interpolate or discard most of the data.
    2. gather_initial_guesses (comb_fit) -- collect candidate periods from
       all three guess_* methods (top n_guesses each, cheap, no ACF
       cross-checking yet).
    3. fit_rotation_period (comb_fit) -- fit the real joint comb model to
       every candidate and pick the one that passes the reliability gates
       with the lowest reduced chi-squared. If nothing passes, this
       reports failure explicitly (result.success == False) rather than
       returning an untrustworthy answer.
    4. Diagnostic plots to sanity-check the result.
"""

import numpy as np
from acf_utils import compute_acf
from comb_fit import gather_initial_guesses, fit_rotation_period, assess_rotation_candidate
from plotting import plot_full_diagnostic


if __name__ == "__main__":
    # --- point this at your own light curve file ---
    # Expects an .npz with 'time' (days) and 'flux' (normalized, may
    # contain NaN at missing cadences) on an evenly-sampled grid.
    lc_path = "/mnt/user-data/uploads/tess_lc.npz"

    d = np.load(lc_path)
    time, flux = d["time"], d["flux"]

    # --- 1. gap-aware ACF ---
    acf_lags, acf = compute_acf(time, flux, max_lag_frac=1 / 3, min_valid_frac=0.3)
    print(f"ACF computed: {len(acf_lags)} lag points, max lag = {acf_lags[-1]:.2f} d")

    # --- 2. gather candidates from all three methods ---
    # If you have ANY prior expectation of the period range (spectral type,
    # literature, visual inspection), pass it via method_kwargs, e.g.:
    #   method_kwargs={"lombscargle": {"min_period": 0.5, "max_period": 10}}
    # for every method you want restricted. Left blank here to demonstrate
    # the fully-blind case.
    guesses, failed_methods = gather_initial_guesses(time, flux, acf_lags, acf, n_guesses=5)
    print(f"Gathered {len(guesses)} candidate periods across "
          f"{3 - len(failed_methods)} methods (failed: {failed_methods or 'none'})")

    # --- 3. fit every candidate, let the joint fit arbitrate ---
    result = fit_rotation_period(acf_lags, acf, guesses)
    print(f"\n{result.message}")

    if result.success:
        print(f"Best period: P = {result.best_fit.P:.4f} d")
        print(f"  n_peaks_used={result.best_fit.n_peaks_used}  "
              f"redchi={result.best_fit.redchi:.4g}")

        diagnostics = assess_rotation_candidate(result.best_fit, acf)
        print("\nAcceptance diagnostics for the winner:")
        for k, v in diagnostics.items():
            print(f"  {k}: {v}")
    else:
        print("No reliable rotation period could be identified for this "
              "light curve -- see the diagnostic plot for what was tried.")

    print("\nTop 5 candidates tested (period, fitted P, n_peaks_used, redchi, passed):")
    for c in sorted(result.candidates, key=lambda c: c.fit.redchi)[:5]:
        print(f"  {c.period:8.4f} -> {c.fit.P:8.4f}  "
              f"n_used={c.fit.n_peaks_used}  redchi={c.fit.redchi:.4g}  "
              f"passed={c.passed_gates}")

    # --- 4. diagnostic plots ---
    fig, axes = plot_full_diagnostic(acf_lags, acf, guesses, result)
    fig.savefig("messy_data_diagnostic.png", dpi=120)
    print("\nSaved diagnostic figure to messy_data_diagnostic.png")
