"""
example_usage.py

Minimal end-to-end demo of comb_fit.py on a synthetic quasi-periodic light
curve: build the ACF, get an initial (P, t0) guess (swap between the three
methods freely), run the joint comb fit, and inspect the result.
"""

import numpy as np
from comb_fit import (
    guess_pairwise_histogram,
    guess_lombscargle,
    guess_acf_fft,
    fit_rotation_period,
    assess_rotation_candidate,
)
from plotting import plot_full_diagnostic


def compute_acf(time, flux, max_lag_frac=1 / 3):
    """Simple FFT-based autocorrelation of an evenly-sampled light curve."""
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

    # --- pick ONE initial-guess method (all three share the same call signature) ---
    guess = guess_pairwise_histogram(time, flux, acf_lags, acf, prominence=0.01)
    # guess = guess_lombscargle(time, flux, acf_lags, acf)
    # guess = guess_acf_fft(time, flux, acf_lags, acf)

    print(f"Initial guess ({guess.method}): P0={guess.P0:.4f}  t0={guess.t0:.4f}")

    # --- joint comb fit ---
    fit = fit_rotation_period(acf_lags, acf, guess, n_peaks=8, window_frac=0.2)
    print(f"Fitted period: {fit.P:.4f} d  (true={true_P})")
    print(f"Peaks used: {fit.n_peaks_used}  dropped: {fit.n_peaks_dropped}  "
          f"reduced chi^2: {fit.redchi:.4g}")

    # --- goodness-of-fit / acceptance diagnostics ---
    diagnostics = assess_rotation_candidate(fit, acf)
    print("\nAcceptance diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")

    # --- diagnostic plots ---
    fig, axes = plot_full_diagnostic(acf_lags, acf, guess, fit)
    fig.savefig("comb_fit_diagnostic.png", dpi=120)
    print("\nSaved diagnostic figure to comb_fit_diagnostic.png")
