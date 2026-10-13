"""Tests for the synthetic light-curve generator itself.

If the generator is wrong, every downstream test is meaningless, so these
check the injected period is genuinely present and the output format
matches what load_smarts_fits produces.
"""

import numpy as np
from astropy.timeseries import LombScargle

from synthetic import CADENCE_DAYS, make_lightcurve, make_suite


def test_reproducible():
    """Same seed -> identical light curve."""
    a = make_lightcurve(2.0, seed=5)
    b = make_lightcurve(2.0, seed=5)
    np.testing.assert_array_equal(a.flux, b.flux)


def test_format_matches_pipeline_input():
    """Even cadence, finite time, NaN gaps, zero-centered flux."""
    lc = make_lightcurve(2.0, n_sectors=2, seed=1)
    assert np.all(np.isfinite(lc.time))
    np.testing.assert_allclose(np.diff(lc.time), CADENCE_DAYS)
    gap_frac = np.mean(~np.isfinite(lc.flux))
    assert 0.03 < gap_frac < 0.15  # two ~1 d gaps per 27.4 d sector
    assert abs(np.nanmean(lc.flux)) < 1e-12


def test_injected_period_dominates_periodogram():
    """A clean equator-on single-spot star peaks at the injected period."""
    lc = make_lightcurve(4.0, amplitude=0.01, white_noise=0.0005, n_spots=1,
                         inclination_deg=90, n_sectors=2, seed=2)
    ok = np.isfinite(lc.flux)
    freq, power = LombScargle(lc.time[ok], lc.flux[ok]).autopower(minimum_frequency=1 / 20,
                                                                  maximum_frequency=5)
    assert abs(1 / freq[np.argmax(power)] - 4.0) / 4.0 < 0.05


def test_suite_period_range_and_ids():
    """Suite periods respect the requested range and IDs are unique."""
    suite = make_suite(20, seed=0, period_range=(0.5, 10.0), n_sectors=2)
    periods = [lc.true_period for lc in suite]
    assert min(periods) >= 0.5 and max(periods) <= 10.0
    assert len({lc.star_id for lc in suite}) == 20
