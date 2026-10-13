"""Tests for the gap-aware ACF (acf_utils.compute_acf)."""

import numpy as np

from acf_utils import compute_acf
from synthetic import make_lightcurve


def brute_force_acf(flux: np.ndarray, max_lag: int) -> np.ndarray:
    """Reference ACF: for each lag, the mean product over valid pairs only,
    normalized by the lag-0 value. Slow (O(N * max_lag)) but obviously right.
    """
    ok = np.isfinite(flux)
    f = np.where(ok, flux - np.nanmean(flux), 0.0)
    out = np.empty(max_lag)
    for k in range(max_lag):
        pairs = ok[: len(f) - k] & ok[k:]
        out[k] = np.sum(f[: len(f) - k][pairs] * f[k:][pairs]) / max(pairs.sum(), 1)
    return out / out[0]


def test_matches_brute_force_on_gappy_data():
    """Masked-FFT ACF equals the direct pairwise definition, gaps included."""
    lc = make_lightcurve(1.5, amplitude=0.01, white_noise=0.002, n_sectors=1, seed=4)
    lags, acf = compute_acf(lc.time, lc.flux)
    ref = brute_force_acf(lc.flux, len(acf))
    np.testing.assert_allclose(acf, ref, atol=1e-6)


def test_lags_start_at_zero_and_acf_normalized():
    """Lag grid starts at 0 and ACF(0) == 1."""
    lc = make_lightcurve(1.5, n_sectors=1, seed=4)
    lags, acf = compute_acf(lc.time, lc.flux)
    assert lags[0] == 0.0
    assert acf[0] == np.float64(1.0) or abs(acf[0] - 1.0) < 1e-12


def test_first_peak_near_true_period(clean_lc):
    """The highest ACF value beyond half a period sits near P."""
    P = clean_lc.true_period
    sel = (clean_lc.acf_lags > 0.5 * P) & (clean_lc.acf_lags < 1.5 * P)
    lag_peak = clean_lc.acf_lags[sel][np.argmax(clean_lc.acf[sel])]
    assert abs(lag_peak - P) / P < 0.05
