"""End-to-end tests of Stage 2 (fit_rotation_period) on synthetic data."""

import pytest

from comb_fit import fit_rotation_period
from evaluation import CORRECT, classify_period
from guesses import gather_initial_guesses
from synthetic import make_lightcurve
from conftest import LCWithACF


def _pipeline(lcx):
    """Default candidates -> joint comb fit. Returns (EnsembleResult, measured P or None)."""
    g, _ = gather_initial_guesses(lcx.time, lcx.flux, lcx.acf_lags, lcx.acf, n_guesses=10)
    res = fit_rotation_period(lcx.acf_lags, lcx.acf, g)
    return res, (res.best_fit.P if res.success else None)


@pytest.mark.slow
@pytest.mark.parametrize("period,n_sectors", [(0.5, 2), (3.0, 2), (8.0, 13)])
def test_recovers_period_on_clean_data(period, n_sectors):
    """Clean, strong signals across short/medium/long periods are recovered
    within REL_TOL. (Long periods need a long baseline: >= 4 ACF peaks within
    1/3 of the baseline.)"""
    lcx = LCWithACF(make_lightcurve(period, amplitude=0.01, white_noise=0.001,
                                    n_sectors=n_sectors, seed=21))
    res, P = _pipeline(lcx)
    assert classify_period(P, period) == CORRECT, res.message


@pytest.mark.slow
def test_honest_failure_on_pure_noise(noise_lc):
    """No signal -> the pipeline should decline (success=False) rather than
    force an answer. This is a core design decision (PROJECT_CONTEXT s3)."""
    res, P = _pipeline(noise_lc)
    assert not res.success, f"pipeline claimed P={P} on pure noise: {res.message}"


@pytest.mark.slow
def test_result_structure(clean_lc):
    """Successful result exposes the fields downstream code depends on."""
    res, P = _pipeline(clean_lc)
    assert res.success
    assert res.candidates and res.candidates[0].passed_gates
    assert res.best_fit.n_peaks_used >= 4
