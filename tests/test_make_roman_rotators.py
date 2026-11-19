"""Tests for tools/make_roman_rotators.py (P34 prototype). No drive, no celerite2 required (that check skips)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_roman_rotators as mr  # noqa: E402
from evaluation import CORRECT, DIFFROT, classify_period  # noqa: E402


def test_grid_fallback_matches_wg6_season():
    t = mr.roman_season_grid(1, times_path="/nonexistent.npy")
    assert t.size == mr.SEASON_POINTS
    assert np.isclose(np.median(np.diff(t)) * 1440, 12.1)
    assert 70.9 < t[-1] < 71.0


def test_sigma_curve_matches_wg6_improved_table():
    assert np.isclose(mr.sigma_f146_mmag(18.0), 1.62)
    assert np.isclose(mr.sigma_f146_mmag(20.0), 4.27)
    s = mr.sigma_f146_mmag(np.linspace(14, 24, 41))
    assert np.all(np.diff(s) >= 0)


def test_rotation_term_variance_is_sigma_squared():
    k0 = mr.rotation_term_kernel(0.0, 0.01, 5.0, 1.0, 5.0, 0.4)
    assert np.isclose(k0, 1e-4)


def test_numpy_sampler_matches_analytic_kernel():
    """Empirical autocovariance of many short numpy draws ~ the analytic RotationTerm kernel."""
    rng = np.random.default_rng(1)
    t = np.arange(400) * 0.05
    args = (1.0, 3.0, 1.0, 4.0, 0.3)
    draws = np.array([mr.sample_rotation_gp(t, *args, rng, backend="numpy") for _ in range(300)])
    for lag in (0, 10, 30, 60):
        emp = np.mean(draws[:, :-lag or None] * draws[:, lag:])
        ana = mr.rotation_term_kernel(lag * 0.05, *args)
        assert abs(emp - ana) < 0.12, (lag, emp, ana)


def test_kernel_agrees_with_celerite2_when_installed():
    celerite2 = pytest.importorskip("celerite2")
    from celerite2 import terms
    tau = np.linspace(0, 20, 50)
    k = terms.RotationTerm(sigma=0.01, period=5.0, Q0=1.0, dQ=5.0, f=0.4).get_value(tau)
    assert np.allclose(k, mr.rotation_term_kernel(tau, 0.01, 5.0, 1.0, 5.0, 0.4), rtol=1e-8, atol=1e-14)


def test_rotator_is_reproducible_and_scoreable():
    p = mr.draw_params("K", np.random.default_rng(7), p_bin=(10.0, 23.0), f146=18.0)
    t = mr.roman_season_grid(1, times_path="/nonexistent.npy")[:2000]
    a = mr.make_rotator(p, seed=11, t=t, backend="numpy")
    b = mr.make_rotator(p, seed=11, t=t, backend="numpy")
    assert np.array_equal(a.flux, b.flux)
    assert abs(np.median(a.flux)) < 1e-12
    lo, hi = a.params["period_range"]
    assert lo <= min(p["band_periods"]) + 1e-9 or lo <= p["P_eq"]
    for P_k in p["band_periods"]:  # every simulated spot band scores as a success
        assert classify_period(P_k, a.true_period, period_range=(lo, hi)) in (CORRECT, DIFFROT)
    task = mr.scorecard_task(a)
    assert task["kind"] == "synthetic" and task["lc"].params["sectors"] == [1]


def test_period_bin_outside_type_range_raises():
    with pytest.raises(ValueError):
        mr.draw_params("active", np.random.default_rng(0), p_bin=(35.5, 71.0))
