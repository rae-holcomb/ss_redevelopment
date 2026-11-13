"""T32: drift-robust variants of the closed-form "local" comb solver.

Each of Rae's five ideas is a COMB_VARIANTS switch; these tests check that
each one does what its docstring says on small synthetic ACFs, that the
untouched "local" variant is unchanged, and that the new gates only appear for
the variants that enable them.
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from guesses import InitialGuess


def _drift_acf(P=3.0, a=0.0, n=9, dt=0.01, noise=0.0, seed=0, jitter=None, heights=None):
    """ACF of downward parabolae at c_k = k P + a k^2 (a cos-like comb whose peaks drift quadratically).

    jitter: optional per-tooth offsets added to c_k; heights: optional per-tooth peak heights.
    """
    lags = np.arange(0, (n + 0.5) * P, dt)
    acf = np.full(len(lags), -1.0)
    for k in range(n + 1):
        c = k * P + a * k * k + (0.0 if jitter is None else jitter[k])
        h = 0.8 if heights is None else heights[k]
        m = np.abs(lags - c) <= 0.5 * P
        acf[m] = np.maximum(acf[m], h - 4.0 * h * ((lags[m] - c) / P) ** 2)
    if noise:
        acf = acf + np.random.default_rng(seed).normal(0, noise, len(acf))
    return lags, acf


# ---------------------------------------------------------------- switches
def test_t32_variants_registered_and_default_unchanged():
    """Every T32 switch resolves, and the default comb variant is still local_screen2 (Rae's call)."""
    for name in ("local_track", "local_tol", "local_tol_lin", "local_tol_width", "local_quad", "local_snrw",
                 "local_scatter", "local_track_tol", "local_track_scatter", "local_robust"):
        assert C.resolve_comb_variant(name)[1]["solver"] == "local"
    assert C.DEFAULT_COMB_VARIANT == "local_screen2"


def test_local_variant_unchanged_by_t32_options():
    """'local' with the T32 options at their defaults gives the T31 local solution bit for bit."""
    lags, acf = _drift_acf(P=3.0, noise=0.03)
    windows = C._build_windows(lags, 3.05, 0.0, 7)
    a = C._fit_comb_local(windows, lags, acf, 3.05, 0.0)
    b = C._fit_comb_local(windows, lags, acf, 3.05, 0.0, weight_mode="none", drift=False)
    assert a["P"] == b["P"] and a["t0"] == b["t0"] and a["centers"] == b["centers"]
    assert a["local_info"]["drift_a"] == 0.0 and not a["local_info"]["drift_fit"]


# ---------------------------------------------------------------- idea 1: tracking
def test_tracked_windows_follow_a_drifting_comb():
    """With P0 6% off, frozen windows lose the late peaks; tracked windows stay centred on them."""
    P, P0 = 3.0, 3.18
    lags, acf = _drift_acf(P=P)
    frozen = C._build_windows(lags, P0, 0.0, 9)
    tracked = C._build_windows_tracked(lags, acf, P0, 0.0, 9)
    off_frozen = [abs(0.5 * (w.lag_lo + w.lag_hi) - w.n * P) / P for w in frozen]
    # (windows clipped by either end of the lag range are not centred by construction)
    off_tracked = [abs(0.5 * (w.lag_lo + w.lag_hi) - w.n * P) / P for w in tracked
                   if lags[0] < w.lag_lo and w.lag_hi < lags[-1]]
    assert max(off_frozen) > 0.25        # the last frozen windows have walked off their peaks
    # n = 1, 2 are placed with P_est = P0 (fewer than two vertices yet): 0.06 P off; later ones are centred
    assert max(off_tracked) < 0.07
    assert max(o for o, w in zip(off_tracked, [w for w in tracked if lags[0] < w.lag_lo and w.lag_hi < lags[-1]])
               if w.n >= 3) < 0.01


def test_local_track_rescues_vertex_gate_on_drifting_comb():
    """End to end: an 8%-off seed on a clean comb is declined on the vertex gate by lsq and local
    (frozen windows), but local_track follows the peaks and recovers P with every tooth."""
    lags, acf = _drift_acf(P=3.0, n=9, noise=0.01)
    g = [InitialGuess(P0=3.24, method="test", t0=0.0)]
    kw = dict(max_period=None, half_lag_gate=False)
    for frozen in ("lsq", "local"):
        r = C.fit_rotation_period(lags, acf, g, comb_variant=frozen, **kw)
        assert not r.success and "frac_vertex_in_window" in r.candidates[0].failed_gates
    r_trk = C.fit_rotation_period(lags, acf, g, comb_variant="local_track", **kw)
    assert r_trk.success and r_trk.best_fit.P == pytest.approx(3.0, rel=0.005)
    assert r_trk.best_fit.n_peaks_used == 8


# ---------------------------------------------------------------- idea 2: growing tolerance
@pytest.mark.parametrize("growth", ["sqrt", "linear", "width"])
def test_tolerance_grows_with_n(growth):
    """A fixed 0.1 P offset on every tooth fails tol at n = 1 but passes at large n (sqrt/linear)."""
    lags, acf = _drift_acf(P=3.0, n=9)
    windows = C._build_windows(lags, 3.0, 0.0, 9)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local"])
    li = fit.local_info
    # move every free vertex ~0.1 P off the comb (just inside 0.05 P sqrt(4) and 0.05 P * 2)
    li["free_vertex"] = {n: li["model_t0"] + li["model_P"] * n + 0.29 for n in li["free_vertex"]}
    spec = dict(tol_growth=growth, tol_base=0.05, tol_k=0.5)
    st = C._vertex_stats(fit, spec)
    if growth == "sqrt":   # 0.1 <= 0.05 sqrt(n) for n >= 4
        assert st["frac_vertex_tol"] == pytest.approx(5 / 8)
    elif growth == "linear":  # 0.1 <= 0.05 n for n >= 2
        assert st["frac_vertex_tol"] == pytest.approx(7 / 8)
    else:  # HWHM of h - 4h(x/P)^2 is P/sqrt(8) ~ 0.354 P; tol = 0.177 P > 0.1 P
        assert st["frac_vertex_tol"] == pytest.approx(1.0)


# ---------------------------------------------------------------- idea 3: quadratic drift
def test_quadratic_drift_coefficient_recovered():
    """local_quad reports the injected drift a (c_n = n P + a n^2) and leaves a ~ 0 on an undrifted comb."""
    lags, acf = _drift_acf(P=3.0, a=0.02, n=9)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local_quad"])
    assert fit.local_info["drift_fit"]
    assert fit.local_info["drift_a"] == pytest.approx(0.02, abs=0.002)
    lags0, acf0 = _drift_acf(P=3.0, a=0.0, n=9)
    fit0 = C._fit_single_candidate(lags0, acf0, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local_quad"])
    assert abs(fit0.local_info["drift_a"]) < 1e-3


def test_drift_a_reported_in_diagnostics():
    """drift_a / drift_a_frac reach CandidateResult.diagnostics for local-family variants."""
    lags, acf = _drift_acf(P=3.0, a=0.02, n=9)
    r = C.fit_rotation_period(lags, acf, [InitialGuess(P0=3.0, method="t", t0=0.0)], comb_variant="local_quad",
                              max_period=None, half_lag_gate=False)
    d = r.candidates[0].diagnostics
    assert d["drift_a"] == pytest.approx(0.02, abs=0.002)
    assert d["drift_a_frac"] == pytest.approx(d["drift_a"] / r.candidates[0].fit.P)


# ---------------------------------------------------------------- idea 4: height weighting
def test_height_weighting_downweights_low_teeth():
    """A low, displaced late tooth pulls the unweighted P more than the height-weighted P."""
    P = 3.0
    jit = np.zeros(10)
    jit[6] = 0.12 * P
    hts = np.full(10, 0.8)
    hts[6] = 0.08
    lags, acf = _drift_acf(P=P, n=9, jitter=jit, heights=hts)
    windows = C._build_windows(lags, P, 0.0, 8)
    plain = C._fit_comb_local(windows, lags, acf, P, 0.0, n_irls=1)
    hw = C._fit_comb_local(windows, lags, acf, P, 0.0, n_irls=1, weight_mode="height")
    assert abs(hw["P"] - P) < abs(plain["P"] - P)


def test_height_weighting_unknown_mode_raises():
    """A typo in the weight mode fails loudly."""
    lags, acf = _drift_acf()
    with pytest.raises(ValueError):
        C._fit_comb_local(C._build_windows(lags, 3.0, 0.0, 4), lags, acf, 3.0, 0.0, weight_mode="snr2")


# ---------------------------------------------------------------- idea 5: scatter gate
def test_scatter_small_for_clean_comb_large_for_jittered():
    """vertex_scatter ~ 0 on a clean comb and ~ the injected RMS jitter on a randomly jittered one."""
    lags, acf = _drift_acf(P=3.0, n=9)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local"])
    assert C._vertex_stats(fit, {})["vertex_scatter"] < 0.005
    jit = 0.08 * 3.0 * (-1.0) ** np.arange(10)  # alternating, so no single tooth is a RANSAC outlier
    jit[0] = 0.0
    lagsj, acfj = _drift_acf(P=3.0, n=9, jitter=jit)
    fitj = C._fit_single_candidate(lagsj, acfj, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local_track"])
    sj = C._vertex_stats(fitj, dict(min_free=0.5))["vertex_scatter"]
    assert sj > 0.04


def test_scatter_inf_when_too_few_vertices():
    """Fewer free vertices than min_free of the teeth -> vertex_scatter = inf (gate fails)."""
    lags, acf = _drift_acf(P=3.0, n=9)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["local"])
    for n in list(fit.local_info["free_vertex"])[:6]:
        fit.local_info["free_vertex"][n] = float("nan")
    assert C._vertex_stats(fit, dict(min_free=0.8))["vertex_scatter"] == float("inf")


def test_scatter_gate_replaces_vertex_window_gate():
    """local_scatter: frac_vertex_in_window is disabled and vertex_scatter is the enabled vertex gate;
    a jittered (noise-comb-like) ACF is declined on it, a clean comb passes."""
    jit = 0.08 * 3.0 * (-1.0) ** np.arange(10)
    jit[0] = 0.0
    lags, acf = _drift_acf(P=3.0, n=9, jitter=jit)
    g = [InitialGuess(P0=3.0, method="t", t0=0.0)]
    kw = dict(max_period=None, half_lag_gate=False)
    r = C.fit_rotation_period(lags, acf, g, comb_variant="local_track_scatter", **kw)
    gr = r.candidates[0].gate_results
    assert gr["frac_vertex_in_window"]["enabled"] is False
    assert gr["vertex_scatter"]["enabled"] and gr["vertex_scatter"]["op"] == "<="
    assert not r.success and r.decline_reason["gates"] == ["vertex_scatter"]
    lags0, acf0 = _drift_acf(P=3.0, n=9)
    assert C.fit_rotation_period(lags0, acf0, g, comb_variant="local_track_scatter", **kw).success


def test_extra_gates_absent_for_default_variants():
    """Default-schema variants carry exactly GATE_NAMES in gate_results (no T32 extras)."""
    lags, acf = _drift_acf(P=3.0, n=9, noise=0.01)
    g = [InitialGuess(P0=3.0, method="t", t0=0.0)]
    for v in ("lsq", "local", "local_screen2", "local_track", "local_quad", "local_snrw"):
        r = C.fit_rotation_period(lags, acf, g, comb_variant=v, max_period=None)
        assert set(r.candidates[0].gate_results) == set(C.GATE_NAMES)
    r = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol", max_period=None)
    assert "frac_vertex_tol" in r.candidates[0].gate_results


def test_vertex_stats_nan_for_lsq_fits():
    """lsq fits have no local_info, so the T32 statistics are NaN (and never gated)."""
    lags, acf = _drift_acf(P=3.0, n=9)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=9, comb_spec=C.COMB_VARIANTS["lsq"])
    st = C._vertex_stats(fit, {})
    assert fit.local_info is None and all(np.isnan(v) for v in st.values())
