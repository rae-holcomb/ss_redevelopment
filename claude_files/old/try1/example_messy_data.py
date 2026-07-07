"""
example_messy_data.py

Recommended workflow for real, messy light curves (gaps, weak/noisy
rotation signals) -- as opposed to example_usage.py, which demonstrates the
basic pipeline on a clean synthetic light curve.

Steps:
    1. compute_acf (acf_utils) -- gap-aware, handles NaN-filled cadences
       (e.g. TESS downlink/momentum-dump gaps) without needing to
       interpolate or discard most of the data.
    2. find_best_rotation_period (comb_fit) -- ensemble arbitration across
       all three guess_* methods: gathers a wide pool of candidate periods
       (each method's full candidate list, not just its top pick, plus
       harmonic/subharmonic expansions), fits the real joint comb model to
       every one, and lets reduced chi-squared (gated on peak count and
       physically-sane positive peak heights) pick the winner. This is
       far more robust to weak/noisy signals and harmonic confusion than
       trusting any single guess_* method's own internal heuristic.
    3. Diagnostic plots, including the candidate-ranking landscape, to
       sanity-check the result and spot genuine ambiguity.
"""

import numpy as np
from acf_utils import compute_acf
from comb_fit import find_best_rotation_period, assess_rotation_candidate
from plotting import plot_comb_fit, plot_candidate_ranking

import matplotlib.pyplot as plt


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

    # --- 2. ensemble initial guess + joint-fit arbitration ---
    # If you have ANY prior expectation of the period range (spectral type,
    # literature, visual inspection of the light curve), pass min_period /
    # max_period here -- it meaningfully speeds things up and keeps
    # physically-impossible candidates out of contention. Left blank here
    # to demonstrate the fully-blind case.
    result = find_best_rotation_period(
        time, flux, acf_lags, acf,
        n_top_peaks=15,       # how many of each method's own top candidates to pool
        min_teeth=4,          # guard against few-tooth candidates winning trivially
        min_peaks_required=4, # winning candidate must retain at least this many peaks
    )

    print(f"\nTested {result.n_candidates_tried} candidate periods across "
          f"{3 - len(result.failed_methods)} guess methods "
          f"(failed: {result.failed_methods or 'none'})")
    print(f"Best period: P = {result.best_fit.P:.4f} d")
    print(f"  n_peaks_used={result.best_fit.n_peaks_used}  "
          f"redchi={result.best_fit.redchi:.4g}")

    print("\nTop 5 candidates (period, fitted P, n_peaks_used, redchi):")
    for c in result.candidates[:5]:
        print(f"  {c.period:8.4f} -> {c.fit.P:8.4f}  "
              f"n_used={c.fit.n_peaks_used}  redchi={c.fit.redchi:.4g}")

    diagnostics = assess_rotation_candidate(result.best_fit, acf)
    print("\nAcceptance diagnostics for the winner:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    # --- 3. diagnostic plots ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_comb_fit(acf_lags, acf, result.best_fit, ax=axes[0])
    plot_candidate_ranking(result, ax=axes[1])
    fig.tight_layout()
    fig.savefig("messy_data_diagnostic.png", dpi=120)
    print("\nSaved diagnostic figure to messy_data_diagnostic.png")
