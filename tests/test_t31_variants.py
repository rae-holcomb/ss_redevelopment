"""T31 Part B: result-changing speed variants of the comb fit and the CWT.

The defaults ("lsq", "full") must be the unchanged methods; the variants must
keep the output schema downstream code reads, and the new pieces (local
per-peak comb solver, sparse Jacobian, warm start, lean / decimated GWPS)
must do what their docstrings say.
"""

from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
import guesses as G
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
from synthetic import make_suite


# ---------------------------------------------------------------- switches
def test_resolve_comb_variant_default_env_and_unknown(monkeypatch):
    """Default is lsq, the env var selects, unknown names raise."""
    monkeypatch.delenv(C.COMB_VARIANT_ENV, raising=False)
    assert C.resolve_comb_variant()[0] == "lsq"
    monkeypatch.setenv(C.COMB_VARIANT_ENV, "local")
    assert C.resolve_comb_variant()[0] == "local"
    assert C.resolve_comb_variant("lsq_tol4")[1]["tol"] == 1e-4
    with pytest.raises(ValueError):
        C.resolve_comb_variant("nope")


def test_resolve_cwt_variant_default_env_and_unknown(monkeypatch):
    """Default is full, the env var selects, unknown names raise."""
    monkeypatch.delenv(G.CWT_VARIANT_ENV, raising=False)
    assert G.resolve_cwt_variant()[0] == "full"
    monkeypatch.setenv(G.CWT_VARIANT_ENV, "dec")
    assert G.resolve_cwt_variant()[0] == "dec"
    with pytest.raises(ValueError):
        G.resolve_cwt_variant("nope")


# ---------------------------------------------------------------- comb fit
def _comb_acf(P=3.0, t0=0.0, n=8, dt=0.02, noise=0.0, seed=0):
    """A synthetic ACF made of exact downward parabolae at t0 + k P (cos-like comb)."""
    lags = np.arange(0, (n + 0.5) * P, dt)
    k = np.round((lags - t0) / P)
    acf = 0.8 - 4.0 * ((lags - t0 - k * P) / P) ** 2
    if noise:
        acf = acf + np.random.default_rng(seed).normal(0, noise, len(acf))
    return lags, acf


def test_local_solver_recovers_exact_comb():
    """On a noiseless comb of parabolae the closed-form local fit returns the true period and centers."""
    lags, acf = _comb_acf(P=3.0)
    windows = C._build_windows(lags, 3.1, 0.0, 6)
    fit = C._fit_comb_local(windows, lags, acf, 3.1, 0.0)
    assert fit["P"] == pytest.approx(3.0, rel=1e-6)
    for n, c in fit["centers"].items():
        assert c == pytest.approx(3.0 * n, abs=1e-5)
    assert all(a > 0 for a in fit["A"].values())


def test_local_solver_respects_bounds():
    """P stays inside P0 * (1 +/- P_bounds_frac) like the joint fit's bounds."""
    lags, acf = _comb_acf(P=3.0)
    windows = C._build_windows(lags, 2.0, 0.0, 4)
    fit = C._fit_comb_local(windows, lags, acf, 2.0, 0.0, P_bounds_frac=0.3)
    assert 2.0 * 0.7 <= fit["P"] <= 2.0 * 1.3


def test_sparse_jacobian_matches_dense():
    """The CSR Jacobian used by tr_solver='lsmr' equals the dense analytic one."""
    lags, acf = _comb_acf(P=3.0, noise=0.01)
    windows = C._build_windows(lags, 3.05, 0.02, 6)
    w = np.linspace(1.0, 0.4, len(lags))
    st = C._comb_model_setup(windows, lags, acf, 3.05, 0.02, weights=w)
    x = st["x0"] + 0.01
    np.testing.assert_array_equal(st["jacobian_sparse"](x).toarray(), st["jacobian"](x))


def test_warm_start_vector_is_feasible_and_round_trips():
    """A local solution maps into a feasible x0 that reproduces its centers."""
    lags, acf = _comb_acf(P=3.0, noise=0.005)
    windows = C._build_windows(lags, 3.0, 0.0, 6)
    st = C._comb_model_setup(windows, lags, acf, 3.0, 0.0)
    init = C._fit_comb_local(windows, lags, acf, 3.0, 0.0)
    x0 = C._warm_start_vector(st, init)
    assert np.all(x0 >= st["lb"]) and np.all(x0 <= st["ub"])
    np.testing.assert_allclose(st["centers"](x0), [init["centers"][n] for n in st["ns"]], atol=1e-9)


@pytest.fixture(scope="module")
def synth_star():
    """ACFs and pooled default-method guesses for one synthetic star."""
    lc = make_suite(4, seed=0)[1]
    t, f = lc.time, lc.flux
    lags, acf, cnt = compute_acf(t, f, return_counts=True)
    l1, a1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    pooled = []
    for m in ("pairwise_histogram", "lombscargle", "acf_fft"):
        pooled += G.two_view_guesses(getattr(G, "guess_" + m), t, f, (l1, a1), (lags, acf), n_guesses=10)
    return lc, lags, acf, cnt, pooled


def test_default_variant_is_explicit_lsq(synth_star, monkeypatch):
    """No variant given == comb_variant='lsq': same period bytes and per-peak values."""
    monkeypatch.delenv(C.COMB_VARIANT_ENV, raising=False)
    _, lags, acf, cnt, pooled = synth_star
    a = C.fit_rotation_period(lags, acf, pooled, acf_counts=cnt)
    b = C.fit_rotation_period(lags, acf, pooled, acf_counts=cnt, comb_variant="lsq")
    assert [c.fit.P for c in a.candidates] == [c.fit.P for c in b.candidates]
    assert [c.fit.per_peak for c in a.candidates] == [c.fit.per_peak for c in b.candidates]


@pytest.mark.parametrize("variant", [v for v in C.COMB_VARIANTS if v != "lsq"])
def test_every_variant_keeps_the_result_schema(synth_star, variant):
    """Each variant fills the CombFitResult fields that gates and ml_features read."""
    lc, lags, acf, cnt, pooled = synth_star
    res = C.fit_rotation_period(lags, acf, pooled, acf_counts=cnt, comb_variant=variant)
    assert res.candidates
    for c in res.candidates:
        fit = c.fit
        assert np.isfinite(fit.P) and np.isfinite(fit.t0)
        assert fit.P_err is None or np.isfinite(fit.P_err) or np.isnan(fit.P_err)
        for pk in fit.per_peak.values():
            assert set(pk) == {"center", "height", "curvature", "residual_rms", "n_points", "vertex_in_window"}
            assert pk["curvature"] >= 0
    if res.success:
        assert res.best_fit.P == pytest.approx(lc.true_period, rel=0.1)


# ---------------------------------------------------------------- CWT
@pytest.fixture(scope="module")
def sine_lc():
    """Evenly sampled noisy sinusoid (P = 4.3 d, 30-min cadence, 54 d)."""
    t = np.arange(0, 54.0, 1 / 48)
    f = np.sin(2 * np.pi * t / 4.3) + np.random.default_rng(3).normal(0, 0.3, len(t))
    return t, f


def test_lean_gwps_bytes_match_full(sine_lc):
    """'lean' accumulates the same GWPS bytes as the full 2D transform and drops wavelet_power."""
    t, f = sine_lc
    full = G.guess_wavelet(t, f, None, None, n_guesses=5, cwt_variant="full")
    lean = G.guess_wavelet(t, f, None, None, n_guesses=5, cwt_variant="lean")
    assert full[0].info["gwps"].tobytes() == lean[0].info["gwps"].tobytes()
    assert [g.P0 for g in full] == [g.P0 for g in lean]
    assert "wavelet_power" in full[0].info and "wavelet_power" not in lean[0].info


@pytest.mark.parametrize("variant", ["dec", "n100", "dec_n100"])
def test_cheaper_cwt_variants_find_the_period(sine_lc, variant):
    """The decimated / coarser-grid GWPS still puts its top peak at the injected period."""
    t, f = sine_lc
    g = G.guess_wavelet(t, f, None, None, n_guesses=5, cwt_variant=variant)
    assert g[0].P0 == pytest.approx(4.3, rel=0.05)


def test_decimated_gwps_close_to_full_at_long_periods(sine_lc):
    """Block-averaged long scales reproduce the full GWPS to a few percent where the power is."""
    t, f = sine_lc
    y = f - f.mean()
    dt = np.median(np.diff(t))
    periods = np.geomspace(4 * dt, (t[-1] - t[0]) / 2, 200)
    scales = G._period_to_morlet_scale(periods, dt, 6.0)
    full = G._gwps_lean(y, scales, 6.0)
    dec = G._gwps_morlet(y, scales, 6.0, periods, dt, decimate=True)
    near = (periods > 3.0) & (periods < 6.0)
    np.testing.assert_allclose(dec[near], full[near], rtol=0.05)
    short = periods < 16 * 2 * dt  # never decimated: equal up to the group's FFT-length rounding
    np.testing.assert_allclose(dec[short], full[short], rtol=1e-12)


@pytest.mark.parametrize("variant", ["local_screen1", "local_screen2"])
def test_local_screen_selects_the_lsq_fit(synth_star, variant):
    """When the screen lets main's answer through, local_screen returns exactly the lsq fit of it."""
    _, lags, acf, cnt, pooled = synth_star
    ref = C.fit_rotation_period(lags, acf, pooled, acf_counts=cnt, comb_variant="lsq")
    scr = C.fit_rotation_period(lags, acf, pooled, acf_counts=cnt, comb_variant=variant)
    assert ref.success and scr.success
    assert scr.best_fit.P == ref.best_fit.P
    assert scr.best_fit.per_peak == ref.best_fit.per_peak
    # candidates that were refit carry lsq's scipy result; screened-out ones the local record
    kinds = {type(c.fit.fit_result).__name__ for c in scr.candidates}
    assert "OptimizeResult" in kinds
