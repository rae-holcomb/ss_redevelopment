"""P44: leave the lag-0 tooth out of the comb fit's outlier rejection and the mean_height_snr gate.

Rae's OK 2026-11-19 (reports/2026-11-20_p44-lag0-out/). The switch is the comb-spec key
lag0_tooth="exclude"; the old behaviour ("include", the default) stays reachable for before/after.
The lag-0 tooth is the one whose comb centre lies below P/2 (the rule _vertex_stats already uses).
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from test_p43_decayed_tail import _comb_acf


# ---------------------------------------------------------------- registration
def test_l0x_variants_registered_and_default_unchanged():
    """The P44 variants add lag0_tooth="exclude" plus their own cutoff; no older variant changes."""
    spec = C.resolve_comb_variant("local_tol_trim_l0x")[1]
    assert spec == dict(C.COMB_VARIANTS["local_tol_trim"], lag0_tooth="exclude",
                        min_mean_height_snr=C.P44_SNR_CUTOFF)
    assert C.resolve_comb_variant("lsq_l0x")[1] == dict(solver="lsq", lag0_tooth="exclude",
                                                        min_mean_height_snr=C.P44_SNR_CUTOFF)
    for name, sp in C.COMB_VARIANTS.items():
        if not name.endswith("_l0x"):
            assert "lag0_tooth" not in sp and "min_mean_height_snr" not in sp, name
    assert C.DEFAULT_COMB_VARIANT == "local_tol_trim"
    assert C.DEFAULT_MIN_MEAN_HEIGHT_SNR == 1.0


def test_lag0_mode_rejects_unknown_value():
    assert C._lag0_mode(None) is False
    assert C._lag0_mode(dict(lag0_tooth="include")) is False
    assert C._lag0_mode(dict(lag0_tooth="exclude")) is True
    with pytest.raises(ValueError):
        C._lag0_mode(dict(lag0_tooth="drop"))


def test_is_lag0_tooth_rule():
    assert C._is_lag0_tooth(0.0, 3.0) and C._is_lag0_tooth(1.49, 3.0)
    assert not C._is_lag0_tooth(1.5, 3.0) and not C._is_lag0_tooth(3.0, 3.0)
    assert not C._is_lag0_tooth(float("nan"), 3.0)


# ---------------------------------------------------------------- outlier rejection
def test_local_solver_no_longer_drops_lag0_tooth():
    """The P43 mechanism: with the later teeth offset by 5% P, the pinned lag-0 parabola fits badly and
    MAD rejection drops it under local_tol_trim. With lag0_tooth="exclude" it survives and nothing is dropped."""
    lags, acf = _comb_acf(P=3.0, noise=0.005, offset=0.15)
    kw = dict(n_peaks=8, min_peaks_required=2)
    old = C._fit_single_candidate(lags, acf, 3.0, 0.15, comb_spec=C.COMB_VARIANTS["local_tol_trim"], **kw)
    new = C._fit_single_candidate(lags, acf, 3.0, 0.15, comb_spec=C.COMB_VARIANTS["local_tol_trim_l0x"], **kw)
    assert 0 not in old.per_peak and old.n_peaks_dropped == 1
    assert 0 in new.per_peak and new.n_peaks_dropped == 0
    assert new.P == pytest.approx(3.0, rel=0.01) and old.P == pytest.approx(3.0, rel=0.01)


@pytest.mark.parametrize("variant", ["local_tol_trim_l0x", "lsq_l0x"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lag0_tooth_never_dropped_when_excluded(variant, seed):
    """Whatever the noise, lag0_tooth="exclude" never rejects the lag-0 tooth, for either solver."""
    lags, acf = _comb_acf(P=3.0, noise=0.02, seed=seed, offset=0.15)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.15, n_peaks=8, min_peaks_required=2,
                                  comb_spec=C.COMB_VARIANTS[variant])
    assert 0 in fit.per_peak


def test_fit_without_lag0_tooth_is_unchanged():
    """A comb seeded at t0 = P has no lag-0 tooth, so the switch changes nothing."""
    lags, acf = _comb_acf(P=3.0, noise=0.01, seed=3)
    a = C._fit_single_candidate(lags, acf, 3.0, 3.0, n_peaks=6, comb_spec=C.COMB_VARIANTS["local_tol_trim"])
    b = C._fit_single_candidate(lags, acf, 3.0, 3.0, n_peaks=6, comb_spec=C.COMB_VARIANTS["local_tol_trim_l0x"])
    assert not any(C._is_lag0_tooth(pk["center"], a.P) for pk in a.per_peak.values())
    assert a.P == b.P and a.t0 == b.t0 and a.n_peaks_dropped == b.n_peaks_dropped
    assert {n: pk["height"] for n, pk in a.per_peak.items()} == {n: pk["height"] for n, pk in b.per_peak.items()}


# ---------------------------------------------------------------- the SNR gate
def test_gate_mean_height_snr_leaves_out_lag0():
    """include: the mean over every surviving tooth (the old gate); exclude: over teeth n >= 1 only,
    which on a decaying ACF is lower because ACF(0) ~ 1 is the tallest tooth."""
    lags, acf = _comb_acf(P=3.0, noise=0.005, offset=0.0)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=8, min_peaks_required=2,
                                  comb_spec=C.COMB_VARIANTS["local_tol_trim_l0x"])
    diag = C.assess_rotation_candidate(fit, acf)
    assert 0 in fit.per_peak
    old = C.gate_mean_height_snr(fit, diag, C.COMB_VARIANTS["local_tol_trim"])
    new = C.gate_mean_height_snr(fit, diag, C.COMB_VARIANTS["local_tol_trim_l0x"])
    snr = np.asarray(diag["height_snr"])
    assert old == pytest.approx(np.nanmean(snr))
    assert new == pytest.approx(np.mean(snr[1:]))
    assert new < old


def test_fit_rotation_period_uses_variant_cutoff_and_lag0_free_mean():
    """min_mean_height_snr=None takes the variant's cutoff; an explicit value wins; the gate's recorded
    value is the lag-0-free mean."""
    lags, acf = _comb_acf(P=3.0, noise=0.005)
    g = [C.InitialGuess(P0=3.0, t0=0.0, method="test")]
    res = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol_trim_l0x", max_period=None)
    cand = res.candidates[0]
    gr = cand.gate_results["mean_height_snr"]
    assert gr["threshold"] == C.P44_SNR_CUTOFF
    assert gr["value"] == pytest.approx(C.gate_mean_height_snr(cand.fit, cand.diagnostics,
                                                               C.COMB_VARIANTS["local_tol_trim_l0x"]))
    res2 = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol_trim_l0x", max_period=None,
                                 min_mean_height_snr=123.0)
    assert res2.candidates[0].gate_results["mean_height_snr"]["threshold"] == 123.0
    assert not res2.success
    res3 = C.fit_rotation_period(lags, acf, g, comb_variant="local_tol_trim", max_period=None)
    assert res3.candidates[0].gate_results["mean_height_snr"]["threshold"] == 1.0
