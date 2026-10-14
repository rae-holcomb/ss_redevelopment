"""Long periods on short baselines: the comb fit should report periods up to
~1/3 of the light-curve baseline B (1-2 TESS sectors), flagged as few-peak,
without inventing periods in noise or regressing the short-period regime.

See reports/2026-10-13_acf-long-period-limit/ for the motivating diagnosis.
"""

import numpy as np
import pytest

import comb_fit
from acf_utils import compute_acf
from comb_fit import fit_rotation_period
from evaluation import CORRECT, classify_period
from guesses import gather_initial_guesses
from synthetic import make_lightcurve


def _baseline(lc) -> float:
    """Time span of a synthetic light curve, days."""
    return float(lc.time[-1] - lc.time[0])


def _run(lc):
    """compute_acf (with pair counts) -> default candidates -> comb fit."""
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags, acf, n_guesses=10)
    res = fit_rotation_period(lags, acf, g, acf_counts=counts)
    return res, (res.best_fit.P if res.success else None)


# --------------------------------------------------------------------------
# compute_acf: lag range and pair counts
# --------------------------------------------------------------------------

def test_acf_default_lag_range_reaches_well_past_one_third():
    """Default lag range is ~0.7 B (limited by pair coverage), not B/3."""
    lc = make_lightcurve(2.0, n_sectors=1, seed=4)
    lags, acf = compute_acf(lc.time, lc.flux)
    assert lags[-1] >= 0.6 * _baseline(lc)
    assert np.all(np.isfinite(acf))


def test_acf_pair_counts_match_brute_force():
    """Returned per-lag counts equal the number of valid (i, i+k) pairs, and
    respect the min_valid_frac coverage cut."""
    lc = make_lightcurve(1.5, n_sectors=1, seed=4)
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    ok = np.isfinite(lc.flux)
    for k in (0, 1, 50, len(lags) // 2, len(lags) - 1):
        assert counts[k] == np.sum(ok[: len(ok) - k] & ok[k:])
    assert counts.min() >= 0.3 * counts[0] - 1e-6


def test_acf_old_lag_range_still_available():
    """max_lag_frac=1/3 reproduces the old truncation (backward compatible)."""
    lc = make_lightcurve(2.0, n_sectors=1, seed=4)
    lags, _ = compute_acf(lc.time, lc.flux, max_lag_frac=1 / 3)
    assert lags[-1] <= _baseline(lc) / 3 + 1e-9


# --------------------------------------------------------------------------
# Weighted comb residuals: analytic Jacobian stays exact
# --------------------------------------------------------------------------

def test_weighted_comb_jacobian_matches_finite_differences():
    """With per-point weights, the analytic Jacobian equals a numerical one."""
    lags = np.linspace(0, 20, 801)
    acf = np.exp(-lags / 30) * np.cos(2 * np.pi * lags / 3.0)
    weights = np.sqrt(np.clip(1 - lags / 28.0, 0.05, None))
    windows = comb_fit._build_windows(lags, 3.0, 3.0, 5)
    resid, jac, x0 = comb_fit._comb_model_functions(windows, lags, acf, 3.0, 3.0, weights=weights)
    x = x0 + 0.01 * np.random.default_rng(0).normal(size=x0.size)
    J = jac(x)
    eps = 1e-6
    J_num = np.empty_like(J)
    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = eps
        J_num[:, j] = (resid(x + dx) - resid(x - dx)) / (2 * eps)
    np.testing.assert_allclose(J, J_num, atol=1e-5)


# --------------------------------------------------------------------------
# Factor-of-2 guard
# --------------------------------------------------------------------------

def test_half_tooth_ratio_detects_double_dip_true_period():
    """ACF with strong peaks at nP and weaker (secondary, double-dip) peaks
    at (n+1/2)P: the interleaved P/2 teeth are clearly weaker -> low ratio."""
    P = 4.0
    lags = np.linspace(0, 20, 2001)
    acf = np.exp(-lags / 60) * (0.3 * np.cos(2 * np.pi * lags / P) + 0.7 * np.cos(4 * np.pi * lags / P))
    r = comb_fit._half_tooth_height_ratio(lags, acf, P_short=P / 2, t0_short=P / 2, P_long=P, t0_long=P)
    assert r < 0.6


def test_half_tooth_ratio_near_one_for_true_short_period():
    """Pure P/2 periodicity: interleaved teeth are as tall as shared ones."""
    P = 4.0
    lags = np.linspace(0, 20, 2001)
    acf = np.exp(-lags / 60) * np.cos(4 * np.pi * lags / P)
    r = comb_fit._half_tooth_height_ratio(lags, acf, P_short=P / 2, t0_short=P / 2, P_long=P, t0_long=P)
    assert r > 0.9


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 2])
def test_one_sector_quarter_baseline_period_recovered_and_flagged(seed):
    """1 sector, P = B/4, clean signal -> CORRECT, flagged as few-peak."""
    B = _baseline(make_lightcurve(1.0, n_sectors=1, seed=seed))
    P = B / 4
    lc = make_lightcurve(P, amplitude=0.01, white_noise=0.001, n_sectors=1, seed=seed, n_spots=1)
    res, Pm = _run(lc)
    assert classify_period(Pm, P) == CORRECT, res.message
    best = res.candidates[0]
    assert best.few_peaks
    assert best.n_teeth_available < 4
    assert res.few_peaks


@pytest.mark.slow
@pytest.mark.parametrize("n_sectors,seed", [(1, 31), (1, 32), (2, 33)])
def test_pure_noise_short_baseline_declined(n_sectors, seed):
    """Pure white noise on a short baseline -> success=False, even though
    the peak requirement is now relaxed for long candidate periods."""
    lc = make_lightcurve(1.0, amplitude=0.0, white_noise=0.001, n_sectors=n_sectors, seed=seed)
    res, Pm = _run(lc)
    assert not res.success, f"claimed P={Pm} on pure noise: {res.message}"


@pytest.mark.slow
@pytest.mark.parametrize("n_sectors,frac,seed", [(1, 0.05, 41), (2, 0.05, 42), (2, 0.07, 43), (13, 0.02, 44)])
def test_short_period_regime_unchanged(n_sectors, frac, seed):
    """P/B <= 0.07: still CORRECT, with the full peak requirement (not few-peak)."""
    B = _baseline(make_lightcurve(1.0, n_sectors=n_sectors, seed=seed))
    P = frac * B
    lc = make_lightcurve(P, amplitude=0.01, white_noise=0.001, n_sectors=n_sectors, seed=seed)
    res, Pm = _run(lc)
    assert classify_period(Pm, P) == CORRECT, res.message
    assert not res.candidates[0].few_peaks
    assert res.best_fit.n_peaks_used >= 4


@pytest.mark.slow
def test_default_call_without_counts_still_works(clean_lc):
    """fit_rotation_period without acf_counts (old call signature) still
    recovers a clean period (unweighted residuals)."""
    g, _ = gather_initial_guesses(clean_lc.time, clean_lc.flux, clean_lc.acf_lags, clean_lc.acf, n_guesses=10)
    res = fit_rotation_period(clean_lc.acf_lags, clean_lc.acf, g)
    assert classify_period(res.best_fit.P if res.success else None, clean_lc.true_period) == CORRECT


@pytest.mark.slow
def test_ml_features_expose_peak_budget_and_baseline_fraction():
    """Feature table carries n_peaks_used, n_teeth_available, few_peaks and
    P / light-curve baseline (not P / ACF lag range)."""
    from ml_features import extract_candidate_features
    lc = make_lightcurve(6.0, amplitude=0.01, white_noise=0.001, n_sectors=1, seed=5, n_spots=1)
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags, acf, n_guesses=10)
    res = fit_rotation_period(lags, acf, g, acf_counts=counts)
    df = extract_candidate_features(res, g, lc.time, lc.flux, lags, acf)
    for col in ("n_peaks_used", "n_teeth_available", "n_peaks_required", "few_peaks",
                "P_over_lc_baseline", "lc_baseline_days"):
        assert col in df.columns, col
    t_ok = lc.time[np.isfinite(lc.flux)]
    span = t_ok[-1] - t_ok[0]  # baseline of the finite data, as documented
    np.testing.assert_allclose(df["lc_baseline_days"], span)
    np.testing.assert_allclose(df["P_over_lc_baseline"], df["fitted_P"] / span)


def test_trailing_decayed_teeth_are_trimmed():
    """Spot evolution: ACF peaks decay through zero at long lag. The trailing
    run of non-positive teeth is trimmed (and refit) instead of failing the
    candidate's frac_positive_heights gate; leading teeth are kept."""
    P = 2.0
    lags = np.linspace(0, 18, 1801)
    acf = 0.6 * np.cos(2 * np.pi * lags / P) + 0.4 - lags / 9.0  # peaks: 1 - nP/9 -> <= 0 from 5P on
    fit = comb_fit._fit_single_candidate(lags, acf, P, P, n_peaks=8, min_keep_after_trim=4)
    heights = [v["height"] for v in fit.per_peak.values()]
    assert fit.n_peaks_trimmed >= 3
    assert min(heights) > 0
    assert fit.n_peaks_used >= 4
    untrimmed = comb_fit._fit_single_candidate(lags, acf, P, P, n_peaks=8, trim_decayed_tail=False)
    assert untrimmed.n_peaks_trimmed == 0 and untrimmed.n_peaks_used == fit.n_peaks_used + fit.n_peaks_trimmed
