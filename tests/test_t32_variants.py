"""T32: `local` comb-solver variants robust to drifting ACF peaks.

Synthetic ACFs here are sums of Gaussian bumps at chosen tooth positions, so
each test controls exactly where the peaks sit (a seed-period error, one
displaced tooth, a quadratic drift) and checks that the new code paths
(tracked windows, vertex gates, quadratic line, SNR weights) do what their
docstrings say. The default `local` / `local_screen2` paths are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C
from guesses import InitialGuess

P = 1.0
LAGS = np.linspace(0.0, 9.5, 9501)


def bump_acf(centers, heights=None, sigma=0.12 * P):
    """ACF made of Gaussian teeth at `centers` (heights default 1, decaying gently)."""
    heights = np.ones(len(centers)) if heights is None else np.asarray(heights)
    y = np.zeros_like(LAGS)
    for c, h in zip(centers, heights):
        y += h * np.exp(-0.5 * ((LAGS - c) / sigma) ** 2)
    return y + 1.0 * np.exp(-0.5 * (LAGS / sigma) ** 2)  # lag-0 spike


def test_variants_registered_and_local_only():
    """Every T32 variant is a local-solver spec; T32 keys on an lsq spec raise."""
    t32 = [k for k in C.COMB_VARIANTS if k.startswith("local_") and k not in
           ("local_warm", "local_warm_tol4", "local_screen1", "local_screen2")]
    assert {"local_seq", "local_tol1", "local_tolsqrt", "local_toln", "local_quad", "local_snrw",
            "local_scatter", "local_seq_tol1"} <= set(t32)
    assert all(C.COMB_VARIANTS[k]["solver"] == "local" for k in t32)
    acf = bump_acf(np.arange(1, 9) * P)
    w = C._build_windows(LAGS, P, P, 4)
    with pytest.raises(ValueError):
        C._solve_comb(dict(solver="lsq", vgate="free"), w, LAGS, acf, P, P)


def test_tracked_windows_follow_a_wrong_seed():
    """Seed P0 4% short: frozen windows lose the late teeth, tracked windows stay on them."""
    acf = bump_acf(np.arange(1, 9) * P)
    P0 = 0.96 * P
    frozen = C._build_windows(LAGS, P0, P0, 8)
    tracked = C._build_windows_tracked(LAGS, acf, P0, P0, 8)
    def n_accepted(ws):
        return sum(C._free_vertex(LAGS[w.mask], acf[w.mask], np.ones(w.mask.sum()), w.lag_lo, w.lag_hi)[1]
                   for w in ws)
    assert n_accepted(tracked) == 8
    assert n_accepted(frozen) <= 6
    mids = np.array([0.5 * (w.lag_lo + w.lag_hi) for w in tracked])
    assert np.max(np.abs(mids - np.arange(1, 9) * P)) < 0.06 * P  # windows sit on the true teeth


def test_local_seq_recovers_period_from_wrong_seed():
    """With tracking the local fit's P is the true one and every vertex flag passes under vgate=free."""
    acf = bump_acf(np.arange(1, 9) * P)
    fit = C._fit_single_candidate(LAGS, acf, 0.96 * P, 0.96 * P, comb_spec=C.COMB_VARIANTS["local_seq_free"])
    assert abs(fit.P - P) < 0.005
    assert all(p["vertex_in_window"] for p in fit.per_peak.values())
    fit_frozen = C._fit_single_candidate(LAGS, acf, 0.96 * P, 0.96 * P, comb_spec=C.COMB_VARIANTS["local_free"])
    assert np.mean([p["vertex_in_window"] for p in fit_frozen.per_peak.values()]) < 1.0


def test_tolerance_gates_flag_a_displaced_tooth():
    """Tooth 6 displaced by 0.12 P: free and toln accept it, tol1 / tolsqrt reject it."""
    c = np.arange(1, 9) * P
    c[5] += 0.12 * P  # n = 5 (window 5 when t0 = P)
    acf = bump_acf(c)
    flags = {}
    for v in ("local_free", "local_tol1", "local_tolsqrt", "local_toln"):
        fit = C._fit_single_candidate(LAGS, acf, P, P, comb_spec=C.COMB_VARIANTS[v], max_reject_iters=0)
        flags[v] = fit.per_peak[5]["vertex_in_window"]
        assert all(fit.per_peak[n]["vertex_in_window"] for n in fit.per_peak if n != 5)
    # tol_n = 0.05 P0 * g(5): const 0.05, sqrt 0.112, n 0.25
    assert flags == {"local_free": True, "local_tol1": False, "local_tolsqrt": False, "local_toln": True}


def test_scatter_statistic():
    """Clean comb: scatter ~0 and the gate passes; < 3 accepted vertices -> NaN (fails)."""
    acf = bump_acf(np.arange(1, 9) * P)
    fit = C._fit_single_candidate(LAGS, acf, P, P, comb_spec=C.COMB_VARIANTS["local_scatter"])
    assert fit.vertex_scatter is not None and fit.vertex_scatter < 0.005
    flat = 0.01 * np.cos(2 * np.pi * LAGS / 0.37)  # no comb at P
    fit2 = C._fit_single_candidate(LAGS, flat, P, P, comb_spec=C.COMB_VARIANTS["local_scatter"], n_peaks=3,
                                   min_peaks_required=2)
    assert fit2.vertex_scatter is None or not fit2.vertex_scatter <= 0.05


def test_quadratic_line_absorbs_drift():
    """Teeth at t0 + nP + a n^2: quad records drift_a ~ a/P and leaves much smaller vertex residuals than the line.

    Tracked windows, so each free vertex is fit on a window centred on its tooth (a free quadratic over an
    off-centre window is biased toward the window centre, which would blur the drift being measured).
    """
    a = 0.006 * P  # tooth 7 sits 0.29 P late
    n = np.arange(8)
    acf = bump_acf(P + n * P + a * n**2)
    fq = C._fit_single_candidate(LAGS, acf, P, P, comb_spec=C.COMB_VARIANTS["local_seq_quad"], max_reject_iters=0,
                                 trim_decayed_tail=False)
    fl = C._fit_single_candidate(LAGS, acf, P, P, comb_spec=C.COMB_VARIANTS["local_seq_tol1"], max_reject_iters=0,
                                 trim_decayed_tail=False)
    assert abs(fq.fit_result.drift_a - a / fq.P) < 0.3 * a / fq.P
    assert np.isnan(fl.fit_result.drift_a)
    def worst(f):
        r = f.fit_result
        return max(abs(r.c_free[k] - r.line(k)) for k in r.c_free if r.accepted[k])
    assert worst(fq) < 0.3 * worst(fl)  # the drift lives in the quadratic, not in the residuals


def test_snr_weights_downweight_a_low_offset_tooth():
    """A low tooth shifted late pulls P less with peak_weight='snr'."""
    c = np.arange(1, 9) * P
    c[7] += 0.08 * P
    h = np.ones(8)
    h[7] = 0.15
    acf = bump_acf(c, h)
    w = C._build_windows(LAGS, P, P, 8)
    plain = C._fit_comb_local(w, LAGS, acf, P, P)
    snr = C._fit_comb_local(w, LAGS, acf, P, P, peak_weight="snr")
    assert abs(snr["P"] - P) < abs(plain["P"] - P)


def test_default_local_unchanged_outputs():
    """Default options: no drift term, frame vertex flags, and raw carries the new vertex record."""
    acf = bump_acf(np.arange(1, 9) * P)
    w = C._build_windows(LAGS, P, P, 8)
    out = C._fit_comb_local(w, LAGS, acf, P, P)
    assert np.isnan(out["raw"].drift_a)
    assert set(out["raw"].accepted) == {x.n for x in w}
    assert abs(out["raw"].line(3) - (out["t0"] + 3 * out["P"])) < 1e-12


@pytest.mark.parametrize("variant", ["local_seq", "local_tol1", "local_tolsqrt", "local_toln", "local_quad",
                                     "local_snrw", "local_scatter", "local_seq_tol1", "local_seq_scatter"])
def test_fit_rotation_period_runs_every_variant(variant):
    """End to end on a clean comb: every T32 variant selects the true period."""
    acf = bump_acf(np.arange(1, 9) * P, heights=np.linspace(0.8, 0.4, 8))
    res = C.fit_rotation_period(LAGS, acf, [InitialGuess(P0=(0.97 if "seq" in variant else 0.995) * P, method="test")], comb_variant=variant,
                                max_period=None)
    assert res.success, res.message
    assert abs(res.best_fit.P - P) < 0.02
