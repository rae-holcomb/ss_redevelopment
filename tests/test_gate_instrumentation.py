"""T16b: per-candidate gate instrumentation and structured decline reasons.

fit_rotation_period records, for every fitted candidate, each gate's
measured value, threshold and pass/fail (CandidateResult.gate_results /
.failed_gates), and a declined EnsembleResult names the gates that rejected
its best-fitting candidates (decline_reason, also appended to message).
These are inspection-only: the tests check that the recorded gates agree
exactly with the pass/fail decisions the function actually made.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
from comb_fit import (GATE_NAMES, InitialGuess, failed_gate_names, fit_rotation_period,
                      format_decline_reason)
from guesses import gather_initial_guesses
from synthetic import make_lightcurve

BASE_GATES = ("n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window")


def _pipeline(lc, **fit_kwargs):
    """Scorecard-equivalent pipeline (two-view Stage 1, pair-count-weighted fit)."""
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    lags1, acf1 = compute_acf(lc.time, lc.flux, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    g, _ = gather_initial_guesses(lc.time, lc.flux, lags1, acf1, n_guesses=10, long_view=(lags, acf))
    return fit_rotation_period(lags, acf, g, acf_counts=counts, **fit_kwargs)


@pytest.fixture(scope="module")
def red_noise_1s():
    """Single-sector red-noise light curve (pb_scan style); the pipeline declines it."""
    return make_lightcurve(1.0, amplitude=0.0, white_noise=0.001, n_sectors=1, seed=901001,
                           red_noise_sigma=0.001, red_noise_tau_days=2.0)


@pytest.fixture(scope="module")
def clean_1s():
    """Clean single-sector 2.5 d rotator; the pipeline measures it."""
    return make_lightcurve(2.5, amplitude=0.01, white_noise=0.001, n_sectors=1, seed=3)


def _check_consistent(res, half_lag_gate=True):
    """Every fitted candidate's recorded gates reproduce its pass/fail flags."""
    assert res.candidates
    for r in res.candidates:
        gr = r.gate_results
        assert set(gr) == set(GATE_NAMES)
        for name, d in gr.items():
            assert {"passed", "value", "threshold", "op", "enabled"} <= set(d)
        assert r.passed_base_gates == all(gr[g]["passed"] for g in BASE_GATES)
        assert r.passed_gates == (len(r.failed_gates) == 0)
        assert r.failed_gates == failed_gate_names(gr)
        assert gr["half_lag_consistency"]["enabled"] == half_lag_gate
        assert gr["few_peaks_height"]["enabled"] == bool(r.few_peaks)
        # values match the diagnostics the gates were computed from
        assert gr["n_peaks"]["value"] == r.fit.n_peaks_used
        assert gr["n_peaks"]["threshold"] == r.n_peaks_required
        assert gr["mean_height_snr"]["value"] == pytest.approx(np.nanmean(r.diagnostics["height_snr"]))
        assert gr["half_lag_consistency"]["contrast"] == r.diagnostics["acf_half_lag_contrast"] or (
            np.isnan(gr["half_lag_consistency"]["contrast"]))


def test_declined_star_names_rejecting_gates(red_noise_1s):
    """A decline carries a 'gates' decline_reason naming the best candidates' failing gates."""
    res = _pipeline(red_noise_1s)
    assert not res.success
    reason = res.decline_reason
    assert reason["kind"] == "gates"
    assert reason["gates"], "a gate decline must name at least one gate"
    assert set(reason["gates"]) <= set(GATE_NAMES)
    best = reason["best_candidates"]
    assert 1 <= len(best) <= 3
    for b, r in zip(best, res.candidates):  # best_candidates follow the redchi order
        assert b["P"] == r.fit.P and b["failed_gates"] == r.failed_gates and b["failed_gates"]
    assert set(reason["gates"]) == {g for b in best for g in b["failed_gates"]}
    assert sum(reason["gate_fail_counts"].values()) == sum(len(r.failed_gates) for r in res.candidates)
    assert "Rejecting gates for the best-fitting candidates" in res.message
    assert format_decline_reason(reason) in res.message
    _check_consistent(res)


def test_success_has_no_decline_reason(clean_1s):
    """A measured star has decline_reason None and a selected candidate with no failed gates."""
    res = _pipeline(clean_1s)
    assert res.success and res.decline_reason is None
    assert res.candidates[0].failed_gates == []
    _check_consistent(res)


def test_fallback_tier_marks_half_lag_override(red_noise_1s, clean_1s):
    """Option D (half_lag_fallback_min=0): admitted candidates have no failed gates."""
    for lc in (red_noise_1s, clean_1s):
        res = _pipeline(lc, half_lag_fallback_min=0.0)
        for r in res.candidates:
            hl = r.gate_results["half_lag_consistency"]
            if hl.get("fallback"):
                assert not hl["passed"] and r.passed_gates
                assert "half_lag_consistency" not in r.failed_gates
        _check_consistent(res)


def test_disabled_half_lag_gate_not_reported(red_noise_1s):
    """half_lag_gate=False: the gate is recorded but disabled, never named as failing."""
    res = _pipeline(red_noise_1s, half_lag_gate=False)
    for r in res.candidates:
        assert "half_lag_consistency" not in r.failed_gates
    _check_consistent(res, half_lag_gate=False)


def test_instrumentation_does_not_change_selection(red_noise_1s, clean_1s):
    """Recomputing pass/fail from gate_results alone reproduces passed_gates and the winner."""
    for lc in (red_noise_1s, clean_1s):
        res = _pipeline(lc)
        passing = [r for r in res.candidates if not r.failed_gates]
        assert res.success == bool(passing)
        if res.success:
            assert res.candidates[0] in passing


def test_trivial_declines_have_kinds():
    """Non-gate declines report their kind: no guesses / nothing sane."""
    lags = np.arange(0, 10, 0.01)
    acf = np.cos(2 * np.pi * lags / 1.0)
    assert fit_rotation_period(lags, acf, []).decline_reason == {"kind": "no_guesses"}
    res = fit_rotation_period(lags, acf, [InitialGuess(P0=50.0, method="x")])
    assert res.decline_reason == {"kind": "none_sane"}


def test_scorecard_gate_descriptors(red_noise_1s, clean_1s):
    """scorecard per-star gate columns and per-candidate rows are consistent."""
    import scorecard
    from evaluation import REL_TOL
    for lc, P_true in ((red_noise_1s, 3.0), (clean_1s, 2.5)):
        lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
        res = _pipeline(lc)
        d = scorecard._gate_descriptors(res, lags, counts, P_true, REL_TOL, None)
        assert d["n_fit_candidates"] == len(res.candidates)
        assert len(d["candidates"]) == len(res.candidates) + len(res.failed_candidates)
        assert sum(c["selected"] for c in d["candidates"]) == int(res.success)
        if res.success:
            assert d["decline_kind"] == "" and d["decline_gates"] == ""
        else:
            assert d["decline_gates"].split(";") == res.decline_reason["gates"]
        assert d["n_success_candidates"] == sum(c["outcome"] in ("correct", "diffrot") for c in d["candidates"])
        df = scorecard.flatten_candidates([dict(star_id="x", true_period=P_true, **d)])
        assert len(df) == len(d["candidates"])
        star, _ = scorecard.flatten([dict(star_id="x", **d)])
        assert "candidates" not in star.columns
