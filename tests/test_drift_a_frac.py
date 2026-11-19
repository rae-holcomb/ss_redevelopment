"""drift_a_frac ranker feature (comb_fit.comb_drift_frac), from the first P43 branch
(claude/p43-decayed-tail-22a749, merged 2026-11-19 for this feature only; its local_tol_tail /
norej / snr0 variants stay on that branch -- Rae chose local_tol_trim from the P43 clean rerun).
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from guesses import InitialGuess

def _decaying_cos_acf(P=3.0, tau=6.0, n_periods=8.6, dt=0.01, noise=0.01, seed=1):
    """A clean rotator's ACF: cos(2 pi lag / P) exp(-lag / tau) plus white noise, ACF(0) = 1."""
    lags = np.arange(0, n_periods * P, dt)
    acf = np.cos(2 * np.pi * lags / P) * np.exp(-lags / tau)
    acf[0] = 1.0
    acf = acf + np.random.default_rng(seed).normal(0, noise, len(acf))
    return lags, acf




# ---------------------------------------------------------------- drift_a_frac feature
def test_comb_drift_frac_recovers_quadratic_drift_and_nan_cases():
    """comb_drift_frac fits a / P to the free vertices (no drift term in the solver needed), is ~0 for an
    evenly spaced comb, NaN for lsq fits, and equals the solver's own a / P when the variant fits drift."""
    P, a = 3.0, 0.02
    lags = np.arange(0, 8.6 * P, 0.01)
    acf = np.full(len(lags), -1.0)
    for k in range(9):
        c = k * P + a * k * k
        mm = np.abs(lags - c) <= 0.5 * P
        acf[mm] = np.maximum(acf[mm], 0.8 - 3.2 * ((lags[mm] - c) / P) ** 2)
    fit = C._fit_single_candidate(lags, acf, P, 0.0, comb_spec=C.COMB_VARIANTS["local_tol"])
    assert C.comb_drift_frac(fit) == pytest.approx(a / fit.P, rel=0.15)
    fq = C._fit_single_candidate(lags, acf, P, 0.0, comb_spec=C.COMB_VARIANTS["local_quad"])
    assert C.comb_drift_frac(fq) == fq.local_info["drift_a"] / fq.P
    flat = C._fit_single_candidate(*_decaying_cos_acf(tau=1e9, noise=0.0), 3.0, 0.0,
                                   comb_spec=C.COMB_VARIANTS["local_tol"])
    assert abs(C.comb_drift_frac(flat)) < 1e-4
    lsq = C._fit_single_candidate(lags, acf, P, 0.0, comb_spec=C.COMB_VARIANTS["lsq"])
    assert np.isnan(C.comb_drift_frac(lsq)) and np.isnan(C.comb_drift_frac(None))
    few = C._fit_single_candidate(lags[lags < 3.6 * P], acf[lags < 3.6 * P], P, 0.0,
                                  comb_spec=C.COMB_VARIANTS["local_tol"])
    assert np.isnan(C.comb_drift_frac(few))  # < 4 vertices


def test_drift_a_frac_in_candidate_feature_table():
    """ml_features carries drift_a_frac per fitted candidate (finite under the default local_tol)."""
    import ml_features
    lags, acf = _decaying_cos_acf(tau=30.0)
    t = np.arange(0, 27.0, 2.0 / 1440)
    f = 1 + 0.01 * np.sin(2 * np.pi * t / 3.0)
    g = [InitialGuess(P0=3.0, method="test", t0=0.0)]
    res = C.fit_rotation_period(lags, acf, g, max_period=None)
    df = ml_features.extract_candidate_features(res, g, t, f, lags, acf, star_id="x")
    assert "drift_a_frac" in df.columns
    assert np.isfinite(df["drift_a_frac"]).all()
    assert df["drift_a_frac"].iloc[0] == C.comb_drift_frac(res.candidates[0].fit)

