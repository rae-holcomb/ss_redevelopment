"""P43: variants of local_tol that try to recover its decayed-tail losses, plus the drift_a_frac feature.

Checks that each new COMB_VARIANTS key does what its comment says on small
synthetic ACFs, that local_tol itself is unchanged, that no threshold moved,
and that the drift_a_frac ranker feature is computed (NaN where it should be).
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from guesses import InitialGuess

P43_VARIANTS = ("local_tol_pos", "local_tol_tail", "local_tol_norej", "local_tol_norej_tail",
                "local_tol_norej_pos", "local_tol_tail_rh", "local_tol_norej_rh")


def _decaying_cos_acf(P=3.0, tau=6.0, n_periods=8.6, dt=0.01, noise=0.01, seed=1):
    """A clean rotator's ACF: cos(2 pi lag / P) exp(-lag / tau) plus white noise, ACF(0) = 1."""
    lags = np.arange(0, n_periods * P, dt)
    acf = np.cos(2 * np.pi * lags / P) * np.exp(-lags / tau)
    acf[0] = 1.0
    acf = acf + np.random.default_rng(seed).normal(0, noise, len(acf))
    return lags, acf


def _vertexless_tail_acf(P=3.0, n_good=3, n_periods=8.6, dt=0.01):
    """Parabolic peaks at k P for k <= n_good, then a gently rising, peakless positive tail.

    The tail's windows have no downward vertex (the ACF is linear there), but their comb-tied
    heights are positive, so the decayed-tail trim (non-positive heights) never fires.
    """
    lags = np.arange(0, n_periods * P, dt)
    acf = np.full(len(lags), -0.2)
    for k in range(n_good + 1):
        h = 0.9 * 0.75 ** k
        m = np.abs(lags - k * P) <= 0.5 * P
        acf[m] = np.maximum(acf[m], h - 4.0 * (h + 0.2) * ((lags[m] - k * P) / P) ** 2)
    tail = lags >= (n_good + 0.5) * P
    acf[tail] = 0.05 + 0.002 * (lags[tail] - (n_good + 0.5) * P) / P
    return lags, acf


# ---------------------------------------------------------------- schema
def test_p43_variants_registered_with_local_tol_thresholds():
    """Every P43 variant resolves, is local_tol plus only its own new keys, and the default is unchanged."""
    base = C.COMB_VARIANTS["local_tol"]
    assert base == dict(solver="local", vertex_gate="tol", tol_growth="sqrt", tol_base=0.05, min_frac_tol=0.6)
    new_keys = {"tol_count", "tol_tail", "reject", "height_fit"}
    for name in P43_VARIANTS:
        spec = C.resolve_comb_variant(name)[1]
        assert {k: v for k, v in spec.items() if k not in new_keys} == base, name
        assert set(spec) - set(base) <= new_keys, name
    assert C.DEFAULT_COMB_VARIANT == "local_tol"


@pytest.mark.parametrize("variant", P43_VARIANTS)
def test_p43_gate_schema(variant):
    """P43 variants record the same gates as local_tol: GATE_NAMES plus frac_vertex_tol (enabled),
    frac_vertex_in_window disabled, and the diagnostics carry frac_vertex_tol_all."""
    lags, acf = _decaying_cos_acf(tau=12.0)
    r = C.fit_rotation_period(lags, acf, [InitialGuess(P0=3.0, method="t", t0=0.0)], comb_variant=variant,
                              max_period=None)
    c = r.candidates[0]
    assert set(c.gate_results) == set(C.GATE_NAMES) | {"frac_vertex_tol"}
    assert c.gate_results["frac_vertex_tol"]["enabled"] and c.gate_results["frac_vertex_tol"]["threshold"] == 0.6
    assert not c.gate_results["frac_vertex_in_window"]["enabled"]
    assert 0.0 <= c.diagnostics["frac_vertex_tol_all"] <= 1.0


def test_local_tol_unchanged_by_p43_keys():
    """local_tol's fit and gate values are bit-identical with the P43 keys spelled out at their defaults."""
    lags, acf = _decaying_cos_acf(tau=12.0, noise=0.03)
    g = [InitialGuess(P0=3.05, method="t", t0=0.0)]
    a = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol", max_period=None).candidates[0]
    spec = dict(C.COMB_VARIANTS["local_tol"], tol_count="all", tol_tail=False, reject=True, height_fit="ls")
    C.COMB_VARIANTS["_p43_test_explicit"] = spec
    try:
        b = C.fit_rotation_period(lags, acf, g, comb_variant="_p43_test_explicit", max_period=None).candidates[0]
    finally:
        del C.COMB_VARIANTS["_p43_test_explicit"]
    assert a.fit.P == b.fit.P and a.fit.per_peak == b.fit.per_peak
    assert a.gate_results == b.gate_results
    assert a.diagnostics["frac_vertex_tol"] == a.diagnostics["frac_vertex_tol_all"]


# ---------------------------------------------------------------- (b) tol_tail
def test_tol_tail_excuses_a_vertexless_tail_that_local_tol_rejects():
    """Three clean teeth then a peakless positive tail: local_tol fails frac_vertex_tol (3 of 7 teeth),
    local_tol_tail excuses the trailing run (keeping n_required = 4 windows) and passes."""
    lags, acf = _vertexless_tail_acf()
    g = [InitialGuess(P0=3.0, method="t", t0=0.0)]
    kw = dict(max_period=None, half_lag_gate=False)
    base = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol", **kw)
    c = base.candidates[0]
    assert not base.success and c.failed_gates == ["frac_vertex_tol"]
    assert c.fit.n_peaks_trimmed == 0 and min(pk["height"] for pk in c.fit.per_peak.values()) > 0
    assert c.diagnostics["frac_vertex_tol"] == pytest.approx(3 / 7)
    tail = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol_tail", **kw)
    assert tail.success and tail.best_fit.P == pytest.approx(3.0, rel=0.01)
    d = tail.candidates[0].diagnostics
    assert d["frac_vertex_tol"] == 1.0 and d["frac_vertex_tol_all"] == pytest.approx(3 / 7)


def test_tol_tail_never_excuses_leading_misses():
    """Only a TRAILING run is excused: a comb whose early teeth miss gets no help."""
    fit = C._fit_single_candidate(*_vertexless_tail_acf(), 3.0, 0.0, comb_spec=C.COMB_VARIANTS["local_tol"])
    li = fit.local_info
    # move the first two teeth's vertices far off the comb (leading misses), keep the rest
    li["free_vertex"][1] += 0.5
    li["free_vertex"][2] += 0.5
    s_all = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol"], n_keep=4)
    s_tail = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_tail"], n_keep=4)
    # teeth 1..7: v = . . v . . . .  -> the tail run 4..7 is excused, the two leading misses still count
    assert s_all["frac_vertex_tol"] == pytest.approx(1 / 7)
    assert s_tail["frac_vertex_tol"] == pytest.approx(1 / 3)


def test_tol_tail_keeps_at_least_n_keep_windows():
    """The excused run stops once n_keep windows (lag-0 window included) would remain."""
    fit = C._fit_single_candidate(*_vertexless_tail_acf(n_good=1), 3.0, 0.0, comb_spec=C.COMB_VARIANTS["local_tol"])
    # windows 0..7: lag-0, one good tooth, six vertexless ones
    s = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_tail"], n_keep=4)
    assert s["frac_vertex_tol"] == pytest.approx(1 / 3)       # teeth 1, 2, 3 counted (4 windows with lag-0)
    s2 = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_tail"], n_keep=None)
    assert s2["frac_vertex_tol"] == 1.0                        # no floor: only tooth 1 left


# ---------------------------------------------------------------- (a) tol_count="positive"
def test_tol_count_positive_skips_non_positive_teeth():
    """frac_vertex_tol over positive-height teeth only: a non-positive tooth without a vertex no longer
    counts against the vertex test (it still counts in frac_positive_heights)."""
    fit = C._fit_single_candidate(*_decaying_cos_acf(tau=30.0, noise=0.0), 3.0, 0.0,
                                  comb_spec=C.COMB_VARIANTS["local_tol"])
    for n in (3, 5):  # make teeth 3 and 5 decayed: no vertex, negative fitted height
        fit.local_info["free_vertex"][n] = float("nan")
        fit.per_peak[n]["height"] = -0.05
    n_teeth = len([n for n in fit.per_peak if n > 0])
    s_all = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol"])
    s_pos = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_pos"])
    assert s_all["frac_vertex_tol"] == pytest.approx((n_teeth - 2) / n_teeth)
    assert s_pos["frac_vertex_tol"] == 1.0
    # every tooth non-positive -> nothing counted -> 0 (fails), never a vacuous pass
    for pk in fit.per_peak.values():
        pk["height"] = -0.05
    assert C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_pos"])["frac_vertex_tol"] == 0.0


# ---------------------------------------------------------------- reject=False
def test_rejection_cascade_on_decaying_acf_and_norej_fix():
    """A plain decaying cosine ACF (coherence time 2 P): the MAD residual-RMS rejection under the local
    solver drops the two TALLEST teeth (residual RMS scales with peak height), so mean height SNR falls
    below 1 and local_tol declines a clean rotator. reject=False keeps every tooth and recovers P."""
    lags, acf = _decaying_cos_acf(tau=6.0)
    g = [InitialGuess(P0=3.0, method="t", t0=0.0)]
    base = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol", max_period=None)
    c = base.candidates[0]
    assert not base.success and "mean_height_snr" in c.failed_gates
    assert c.fit.n_peaks_dropped == 2 and min(c.fit.per_peak) == 2
    norej = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol_norej", max_period=None)
    assert norej.success and norej.best_fit.P == pytest.approx(3.0, rel=0.01)
    assert norej.candidates[0].fit.n_peaks_dropped == 0 and min(norej.candidates[0].fit.per_peak) == 0


def test_norej_still_trims_decayed_tail():
    """reject=False switches off only the MAD loop; the non-positive-height tail trim still runs."""
    lags, acf = _vertexless_tail_acf()
    tail = lags >= 3.5 * 3.0
    acf[tail] = -0.05 - 0.002 * (lags[tail] - 3.5 * 3.0) / 3.0  # decayed below zero: negative tied heights
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, comb_spec=C.COMB_VARIANTS["local_tol_norej"],
                                  min_keep_after_trim=4)
    assert fit.n_peaks_dropped == 0 and fit.n_peaks_trimmed == 4 and sorted(fit.per_peak) == [0, 1, 2, 3]


# ---------------------------------------------------------------- height_fit="robust"
def test_robust_height_fit_resists_a_glitch():
    """One glitched stretch inside a tooth's window pulls the plain-LS height down; soft-L1 IRLS much less."""
    lags, acf = _decaying_cos_acf(tau=1e9, noise=0.005)
    m = (lags > 2 * 3.0 - 0.15) & (lags < 2 * 3.0 - 0.05)
    acf = acf.copy()
    acf[m] -= 0.8
    windows = C._build_windows(lags, 3.0, 0.0, 5)
    ls = C._fit_comb_local(windows, lags, acf, 3.0, 0.0)
    rb = C._fit_comb_local(windows, lags, acf, 3.0, 0.0, height_fit="robust")
    assert abs(rb["h"][2] - 1.0) < 0.5 * abs(ls["h"][2] - 1.0)
    assert ls["P"] == rb["P"] and ls["centers"] == rb["centers"]  # heights only
    with pytest.raises(ValueError):
        C._fit_comb_local(windows, lags, acf, 3.0, 0.0, height_fit="nope")


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
