"""
example_usage.py

Minimal end-to-end demo of comb_fit.py (v2) on a synthetic quasi-periodic
light curve: build the ACF, gather candidate periods from all three
guess_* methods, fit and arbitrate between them with fit_rotation_period,
and inspect/plot the result.
"""

import numpy as np
from comb_fit import gather_initial_guesses, fit_rotation_period, assess_rotation_candidate
from plotting import plot_full_diagnostic


def compute_acf(time, flux, max_lag_frac=1 / 3):
    """Simple FFT-based autocorrelation of an evenly-sampled light curve
    (no gaps). For real data with NaN-filled gaps, use acf_utils.compute_acf
    instead -- see example_messy_data.py.
    """
    x = flux - np.mean(flux)
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf_full = np.fft.ifft(f * np.conj(f)).real[:n]
    acf_full /= acf_full[0]
    dt = np.median(np.diff(time))
    lags = np.arange(n) * dt
    mask = lags <= lags[-1] * max_lag_frac
    return lags[mask], acf_full[mask]


if __name__ == "__main__":
    # --- synthetic light curve: two spot harmonics + slow amplitude decay + noise ---
    rng = np.random.default_rng(0)
    true_P = 3.7  # days
    time = np.arange(0, 80.0, 0.02)
    phase = 2 * np.pi * time / true_P
    amp_mod = 1.0 + 0.4 * np.sin(2 * np.pi * time / (true_P * 9))
    flux = 1.0 + 0.01 * amp_mod * (np.sin(phase) + 0.4 * np.sin(2 * phase + 0.3))
    flux += rng.normal(0, 0.0015, size=time.size)
    flux /= np.median(flux)

    acf_lags, acf = compute_acf(time, flux)

    # --- 1. gather candidate periods from all three methods ---
    # Each guess_* function proposes its own top n_guesses candidates,
    # ranked by its own method-specific evidence, WITHOUT checking them
    # against the ACF's shape -- that happens next, in fit_rotation_period.
    guesses, failed_methods = gather_initial_guesses(time, flux, acf_lags, acf, n_guesses=5)
    print(f"Gathered {len(guesses)} candidate periods "
          f"(failed methods: {failed_methods or 'none'})")

    # --- 2. fit every candidate and let the joint comb fit arbitrate ---
    result = fit_rotation_period(acf_lags, acf, guesses)

    print(f"\n{result.message}")
    if not result.success:
        print("No reliable rotation period found for this light curve.")
    else:
        print(f"Fitted period: {result.best_fit.P:.4f} d  (true={true_P})")
        print(f"Peaks used: {result.best_fit.n_peaks_used}  "
              f"dropped: {result.best_fit.n_peaks_dropped}  "
              f"reduced chi^2: {result.best_fit.redchi:.4g}")

        diagnostics = assess_rotation_candidate(result.best_fit, acf)
        print("\nAcceptance diagnostics for the winner:")
        for k, v in diagnostics.items():
            print(f"  {k}: {v}")

    # --- 3. diagnostic plots ---
    fig, axes = plot_full_diagnostic(acf_lags, acf, guesses, result)
    fig.savefig("comb_fit_diagnostic.png", dpi=120)
    print("\nSaved diagnostic figure to comb_fit_diagnostic.png")
