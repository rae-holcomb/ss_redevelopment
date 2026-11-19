"""P43: local_tol variants for decaying ACFs (reports/2026-11-16_p43-clean-rerun/).

Three switches on top of local_tol, none adding a threshold:
  tol_teeth="positive"  frac_vertex_tol over teeth with fitted height > 0 only;
  trim_novertex         the decayed-tail trim also drops trailing teeth without a free vertex;
  center_search         teeth without a free vertex are centred where they fit best within the existing
                        jitter bound ("t0": the lag-0 tooth within lsq's t0 bound, cut at the first lag).
"""
from __future__ import annotations

import numpy as np
import pytest

import comb_fit as C


def _comb_acf(P=3.0, heights=(1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05, 0.02), dt=0.01, floor=-0.05,
              noise=0.0, seed=0, flat_from=None, offset=0.0):
    """ACF of downward parabolae at k P with the given heights, on a constant floor.

    offset: teeth k >= 1 sit at k P + offset (the lag-0 peak stays at 0), so the vertex line's intercept
    is not the lag-0 peak. flat_from: teeth k >= flat_from are replaced by a gently falling line (no local maximum), the way a
    decayed ACF's late teeth look once spot evolution has erased the signal.
    """
    n = len(heights)
    lags = np.arange(0, (n - 0.5) * P, dt)
    acf = np.full(len(lags), float(floor))
    for k, h in enumerate(heights):
        if flat_from is not None and k >= flat_from:
            continue
        c = k * P + (offset if k else 0.0)
        m = np.abs(lags - c) <= 0.5 * P
        acf[m] = np.maximum(acf[m], h - 4.0 * (h - floor) * ((lags[m] - c) / P) ** 2)
    if flat_from is not None:
        start = (flat_from - 0.5) * P
        m = lags >= start
        acf[m] = 0.03 - 0.01 * (lags[m] - start) / P
    if noise:
        acf = acf + np.random.default_rng(seed).normal(0, noise, len(acf))
    return lags, acf


# ---------------------------------------------------------------- registration
def test_p43_variants_registered_and_default_unchanged():
    """The P43 variants are local_tol plus one or two switches; the default stays local_tol."""
    base = C.COMB_VARIANTS["local_tol"]
    for name, extra in [("local_tol_pos", dict(tol_teeth="positive")),
                        ("local_tol_trim", dict(trim_novertex=True)),
                        ("local_tol_trim_pos", dict(trim_novertex=True, tol_teeth="positive")),
                        ("local_tol_cs", dict(center_search=True)),
                        ("local_tol_cs_trim", dict(center_search=True, trim_novertex=True)),
                        ("local_tol_cs0", dict(center_search="t0")),
                        ("local_tol_cs0_trim", dict(center_search="t0", trim_novertex=True))]:
        assert C.resolve_comb_variant(name)[1] == dict(base, **extra)
    assert C.DEFAULT_COMB_VARIANT == "local_tol_trim"  # Rae, 2026-11-19


def test_local_fit_unchanged_without_center_search():
    """center_search=False reproduces the plain local solution bit for bit."""
    lags, acf = _comb_acf(noise=0.02)
    windows = C._build_windows(lags, 3.05, 0.0, 8)
    a = C._fit_comb_local(windows, lags, acf, 3.05, 0.0)
    b = C._fit_comb_local(windows, lags, acf, 3.05, 0.0, center_search=False)
    assert a["P"] == b["P"] and a["centers"] == b["centers"] and a["h"] == b["h"] and a["redchi"] == b["redchi"]


# ---------------------------------------------------------------- _best_center_in_bounds
def test_best_center_finds_parabola_vertex_inside_bounds():
    """On an exact parabola the bounded search returns its vertex to well under the grid step."""
    lag = np.linspace(9.0, 11.0, 81)
    y = 0.7 - 0.3 * (lag - 10.13) ** 2
    c = C._best_center_in_bounds(lag, y, np.ones_like(lag), 9.8, 10.4)
    assert c == pytest.approx(10.13, abs=1e-3)


def test_best_center_respects_bounds():
    """A vertex outside [lo, hi] gives the nearer bound; a degenerate interval returns lo."""
    lag = np.linspace(9.0, 11.0, 81)
    y = 0.7 - 0.3 * (lag - 10.6) ** 2
    assert C._best_center_in_bounds(lag, y, np.ones_like(lag), 9.8, 10.3) == pytest.approx(10.3)
    assert C._best_center_in_bounds(lag, y, np.ones_like(lag), 10.0, 10.0) == 10.0


# ---------------------------------------------------------------- center_search
def test_center_search_improves_vertexless_lag0_tooth_and_keeps_P():
    """The lag-0 tooth never has a free vertex. With the later teeth offset by 3% P, the vertex line's
    intercept (where the pinned lag-0 parabola sits) is 3% P from the lag-0 peak, and the searched centre
    fits that tooth better. P and t0 come from the free vertices either way."""
    lags, acf = _comb_acf(P=3.0, noise=0.0, offset=0.09)
    windows = C._build_windows(lags, 3.0, 0.09, 8)
    pinned = C._fit_comb_local(windows, lags, acf, 3.0, 0.09)
    searched = C._fit_comb_local(windows, lags, acf, 3.0, 0.09, center_search=True)
    assert pinned["t0"] == pytest.approx(0.09, abs=1e-6)
    assert np.isnan(pinned["local_info"]["free_vertex"][0])
    assert searched["P"] == pinned["P"] and searched["t0"] == pinned["t0"]

    def rms0(fit):
        w = windows[0]
        lag, y = lags[w.mask], acf[w.mask]
        return np.sqrt(np.mean((fit["h"][0] - fit["A"][0] * (lag - fit["centers"][0]) ** 2 - y) ** 2))

    assert rms0(searched) < 0.5 * rms0(pinned)
    # teeth WITH a free vertex are placed exactly as before
    for n in pinned["centers"]:
        if n > 0 and np.isfinite(pinned["local_info"]["free_vertex"][n]):
            assert searched["centers"][n] == pinned["centers"][n]


def test_center_search_t0_keeps_lag0_tooth_inside_acf():
    """center_search="t0": a vertex-less lag-0 tooth is never centred before the ACF's first lag, even when
    the vertex line's intercept is negative (center_search=True would follow it there)."""
    lags, acf = _comb_acf(P=3.0, offset=-0.09)
    windows = C._build_windows(lags, 3.0, -0.09, 8)
    fit = C._fit_comb_local(windows, lags, acf, 3.0, -0.09, center_search="t0")
    assert np.isnan(fit["local_info"]["free_vertex"][0]) and fit["t0"] < 0
    assert fit["centers"][0] >= lags[0]
    assert fit["h"][0] <= 1.05  # no extrapolated lag-0 peak above the ACF's own maximum


# ---------------------------------------------------------------- tol_teeth="positive"
def test_tol_teeth_positive_ignores_negative_teeth():
    """Teeth with fitted height <= 0 are left out of frac_vertex_tol under tol_teeth="positive"."""
    lags, acf = _comb_acf(P=3.0, heights=(1.0, 0.8, 0.6, 0.4, -0.2, -0.3, 0.3, -0.25), floor=-0.4)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=8, trim_decayed_tail=False,
                                  max_reject_iters=0, comb_spec=C.COMB_VARIANTS["local_tol"])
    neg = [n for n, pk in fit.per_peak.items() if n > 0 and pk["height"] <= 0]
    assert neg, "the test comb needs teeth with non-positive fitted height"
    all_ = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol"])
    pos = C._vertex_stats(fit, C.COMB_VARIANTS["local_tol_pos"])
    tol_ok = {n: np.isfinite(fit.local_info["free_vertex"][n])
              and abs(fit.local_info["free_vertex"][n] - (fit.local_info["model_t0"] + fit.local_info["model_P"] * n))
              <= 0.05 * fit.P * np.sqrt(n) for n in fit.per_peak if n > 0}
    keep = [n for n in tol_ok if fit.per_peak[n]["height"] > 0]
    assert all_["frac_vertex_tol"] == pytest.approx(np.mean(list(tol_ok.values())))
    assert pos["frac_vertex_tol"] == pytest.approx(np.mean([tol_ok[n] for n in keep]))
    with pytest.raises(ValueError):
        C._vertex_stats(fit, dict(C.COMB_VARIANTS["local_tol"], tol_teeth="bogus"))


# ---------------------------------------------------------------- trim_novertex
def test_trim_novertex_drops_trailing_flat_teeth():
    """Late teeth that decayed into a falling line keep small positive heights (no h <= 0 trim), have no
    free vertex and fail the tolerance test; trim_novertex removes that trailing run and the candidate
    then passes frac_vertex_tol."""
    lags, acf = _comb_acf(P=3.0, flat_from=5, noise=0.002)
    kw = dict(n_peaks=8, min_peaks_required=2, min_keep_after_trim=4)
    plain = C._fit_single_candidate(lags, acf, 3.0, 0.0, comb_spec=C.COMB_VARIANTS["local_tol"], **kw)
    trim = C._fit_single_candidate(lags, acf, 3.0, 0.0, comb_spec=C.COMB_VARIANTS["local_tol_trim"], **kw)
    late = [n for n in plain.per_peak if n >= 5]
    assert late and all(np.isnan(plain.local_info["free_vertex"][n]) for n in late)
    assert plain.n_peaks_trimmed == 0
    assert trim.n_peaks_trimmed == len(late) and max(trim.per_peak) == 4
    assert C._vertex_stats(plain, C.COMB_VARIANTS["local_tol"])["frac_vertex_tol"] < 0.6
    assert C._vertex_stats(trim, C.COMB_VARIANTS["local_tol"])["frac_vertex_tol"] == 1.0


def test_trim_novertex_keeps_required_teeth():
    """The trim never goes below min_keep_after_trim, so an early-dead comb is not rescued by trimming."""
    lags, acf = _comb_acf(P=3.0, flat_from=2, noise=0.002)
    fit = C._fit_single_candidate(lags, acf, 3.0, 0.0, n_peaks=8, min_peaks_required=2, min_keep_after_trim=4,
                                  comb_spec=C.COMB_VARIANTS["local_tol_trim"])
    assert fit.n_peaks_used == 4


def test_trim_novertex_ignored_for_lsq():
    """trim_novertex is a local-solver switch: an lsq spec with it set fits exactly like lsq."""
    lags, acf = _comb_acf(P=3.0, flat_from=5, noise=0.002)
    kw = dict(n_peaks=8, min_peaks_required=2, min_keep_after_trim=4)
    a = C._fit_single_candidate(lags, acf, 3.0, 0.0, comb_spec=C.COMB_VARIANTS["lsq"], **kw)
    b = C._fit_single_candidate(lags, acf, 3.0, 0.0, comb_spec=dict(C.COMB_VARIANTS["lsq"], trim_novertex=True), **kw)
    assert a.P == b.P and a.n_peaks_trimmed == b.n_peaks_trimmed
