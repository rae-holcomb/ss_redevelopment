"""P59: the lag-0 tooth out of the remaining comb gates (n_peaks, frac_positive_heights, frac_vertex_in_window).

Rae's OK 2026-11-20 (reports/2026-11-20_p59-lag0-gates/). Switch: comb-spec key lag0_gates=True, on top of
P44's lag0_tooth="exclude"; variants may carry their own min_peaks_required / min_frac_positive_heights.
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from test_p43_decayed_tail import _comb_acf

HEIGHTS = (1.0, 0.5, -0.1, 0.4, -0.1, 0.3, 0.2, 0.15)  # interior non-positive teeth, positive tail (no trim)


def _cand(variant, lags, acf, P=3.0, t0=0.0, **kw):
    res = C.fit_rotation_period(lags, acf, [C.InitialGuess(P0=P, t0=t0, method="test")], comb_variant=variant,
                                max_period=None, **kw)
    return res, res.candidates[0]


# ---------------------------------------------------------------- registration
def test_l0x2_variants_registered_default_unchanged():
    spec = C.resolve_comb_variant("local_tol_trim_l0x2")[1]
    assert spec == dict(C.COMB_VARIANTS["local_tol_trim_l0x"], lag0_gates=True,
                        min_mean_height_snr=C.P59_SNR_CUTOFF, min_peaks_required=C.P59_MIN_PEAKS,
                        min_frac_positive_heights=C.P59_MIN_FRAC_POS)
    assert C.resolve_comb_variant("lsq_l0x2")[1]["lag0_gates"] is True
    for name, sp in C.COMB_VARIANTS.items():
        if not name.endswith("_l0x2"):
            assert "lag0_gates" not in sp and "min_peaks_required" not in sp, name
    assert C.DEFAULT_COMB_VARIANT == "local_tol_trim_l0x"
    assert (C.DEFAULT_MIN_PEAKS_REQUIRED, C.DEFAULT_MIN_FRAC_POSITIVE_HEIGHTS) == (4, 0.8)


# ---------------------------------------------------------------- lag0_gate_stats
def test_lag0_gate_stats_counts_real_teeth_only():
    lags, acf = _comb_acf(P=3.0, heights=HEIGHTS, noise=0.003)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=8, min_peaks_required=2,
                                  comb_spec=C.COMB_VARIANTS["local_tol_trim_l0x2"])
    assert 0 in fit.per_peak and C._is_lag0_tooth(fit.per_peak[0]["center"], fit.P)
    st = C.lag0_gate_stats(fit)
    real = [p for n, p in fit.per_peak.items() if n != 0]
    assert st["n_peaks_gate"] == fit.n_peaks_used - 1 == len(real)
    assert st["frac_positive_gate"] == pytest.approx(np.mean([p["height"] > 0 for p in real]))
    assert st["frac_vertex_in_window_gate"] == pytest.approx(np.mean([p["vertex_in_window"] for p in real]))
    allpos = np.mean([p["height"] > 0 for p in fit.per_peak.values()])
    assert st["frac_positive_gate"] < allpos  # the lag-0 tooth was a guaranteed positive


def test_lag0_gate_stats_without_lag0_tooth_equals_plain_counts():
    lags, acf = _comb_acf(P=3.0, noise=0.01, seed=3)
    fit = C._fit_single_candidate(lags, acf, 3.0, 3.0, n_peaks=6, comb_spec=C.COMB_VARIANTS["local_tol_trim_l0x2"])
    st = C.lag0_gate_stats(fit)
    assert st["n_peaks_gate"] == fit.n_peaks_used
    assert st["frac_positive_gate"] == np.mean([p["height"] > 0 for p in fit.per_peak.values()])


# ---------------------------------------------------------------- gates in fit_rotation_period
def test_gate_values_are_lag0_free_under_l0x2_only():
    lags, acf = _comb_acf(P=3.0, heights=HEIGHTS, noise=0.003)
    _, new = _cand("local_tol_trim_l0x2", lags, acf)
    _, old = _cand("local_tol_trim_l0x", lags, acf)
    g_new, g_old = new.gate_results, old.gate_results
    st = C.lag0_gate_stats(new.fit)
    assert g_new["n_peaks"]["value"] == st["n_peaks_gate"] == new.fit.n_peaks_used - 1
    assert g_new["frac_positive_heights"]["value"] == pytest.approx(st["frac_positive_gate"])
    # the P44 variant keeps the old definitions
    assert g_old["n_peaks"]["value"] == old.fit.n_peaks_used
    assert g_old["frac_positive_heights"]["value"] == pytest.approx(old.diagnostics["frac_positive_heights"])


def test_frac_positive_gate_can_flip_on_the_lag0_tooth():
    """A threshold between the with- and without-lag-0 fractions passes the old definition and fails the new."""
    lags, acf = _comb_acf(P=3.0, heights=HEIGHTS, noise=0.003)
    _, c0 = _cand("local_tol_trim_l0x", lags, acf)
    _, c2 = _cand("local_tol_trim_l0x2", lags, acf)
    a = c0.gate_results["frac_positive_heights"]["value"]
    b = c2.gate_results["frac_positive_heights"]["value"]
    assert b < a
    thr = 0.5 * (a + b)
    _, c0 = _cand("local_tol_trim_l0x", lags, acf, min_frac_positive_heights=thr)
    _, c2 = _cand("local_tol_trim_l0x2", lags, acf, min_frac_positive_heights=thr)
    assert c0.gate_results["frac_positive_heights"]["passed"]
    assert not c2.gate_results["frac_positive_heights"]["passed"]


def test_teeth_available_excludes_lag0_tooth():
    """A short ACF with 4 windows from t0 = 0: the lag-0 tooth leaves the available count under l0x2."""
    lags, acf = _comb_acf(P=3.0, heights=(1.0, 0.6, 0.5, 0.4), noise=0.003)
    _, c0 = _cand("local_tol_trim_l0x", lags, acf)
    _, c2 = _cand("local_tol_trim_l0x2", lags, acf)
    assert c2.n_teeth_available == c0.n_teeth_available - 1
    assert c2.n_peaks_required == min(c2.n_teeth_available, 4)


def test_window_vertex_gate_lag0_free_for_lsq():
    lags, acf = _comb_acf(P=3.0, noise=0.003)
    _, c = _cand("lsq_l0x2", lags, acf)
    st = C.lag0_gate_stats(c.fit)
    assert c.gate_results["frac_vertex_in_window"]["value"] == pytest.approx(st["frac_vertex_in_window_gate"])


def test_threshold_precedence():
    lags, acf = _comb_acf(P=3.0, noise=0.003)
    _, c = _cand("local_tol_trim_l0x2", lags, acf)
    assert c.gate_results["frac_positive_heights"]["threshold"] == C.P59_MIN_FRAC_POS
    assert c.gate_results["mean_height_snr"]["threshold"] == C.P59_SNR_CUTOFF
    _, c = _cand("local_tol_trim_l0x2", lags, acf, min_frac_positive_heights=0.33, min_peaks_required=7)
    assert c.gate_results["frac_positive_heights"]["threshold"] == 0.33
    assert c.n_peaks_required == 7
    _, c = _cand("local_tol_trim_l0x", lags, acf)
    assert c.gate_results["frac_positive_heights"]["threshold"] == 0.8 and c.n_peaks_required == 4
