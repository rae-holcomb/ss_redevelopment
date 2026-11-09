"""Tests for the T16c searchable range (decision D4): coverage.searchable_range,
the comb fit's beyond_reach skip, and the P_over_reach ML feature."""

import numpy as np
import pytest

from acf_utils import compute_acf
from comb_fit import fit_rotation_period
from coverage import reach_from_pair_counts, searchable_range
from guesses import InitialGuess
from ml_features import _reach_features
from synthetic import make_lightcurve

CADENCE = 1.0 / 48  # 30-min FFI cadence (days)


def _grid(days: float, gaps=()) -> tuple:
    """A regular time grid with flux 1, NaN inside each (start, end) gap."""
    t = np.arange(0, days, CADENCE)
    f = np.ones_like(t)
    for a, b in gaps:
        f[(t >= a) & (t < b)] = np.nan
    return t, f


def _reference_reach(time: np.ndarray, flux: np.ndarray, frac: float = 0.3,
                     max_lag_frac: float = 0.7) -> float:
    """Independent brute-force reference for the D4 / T23 reach definition.

    R = the last lag k * dt before the number of finite (i, i+k) pairs first
    drops below frac x the number of finite cadences, capped at the last grid
    lag <= max_lag_frac x the full array's baseline (as compute_acf and
    evaluation.acf_reach do). Direct pair counting (no FFT), on the light
    curve's own regular grid.
    """
    m = np.isfinite(flux)
    dt = np.median(np.diff(time))
    n0 = m.sum()
    k_cap = int(np.floor(max_lag_frac * (len(m) - 1) + 1e-9))
    for k in range(1, len(m)):
        if np.count_nonzero(m[:-k] & m[k:]) < frac * n0:
            return min(k - 1, k_cap) * dt
    return min(len(m) - 1, k_cap) * dt


SAMPLINGS = {
    "contiguous_1sector": _grid(27.4),
    "1sector_orbit_gap": _grid(27.4, [(13.2, 14.2)]),
    "consecutive_2": _grid(54.8, [(13.2, 14.2), (26.9, 27.9), (40.6, 41.6)]),
    "gapped_2": _grid(82.2, [(13.2, 14.2), (27.4, 54.8), (68.0, 69.0)]),
    "1-2-13": _grid(356.2, [(54.8, 328.8)]),
}


@pytest.mark.parametrize("name", list(SAMPLINGS))
def test_reach_matches_reference_definition(name):
    """searchable_range's R equals the brute-force reference definition
    (valid-pair fraction 0.3) to the cadence. Cross-check against
    evaluation.acf_reach once T23 merges."""
    t, f = SAMPLINGS[name]
    r = searchable_range(t, f)
    assert r["reach"] == pytest.approx(_reference_reach(t, f), abs=1e-9)
    assert r["p_max"] == pytest.approx(0.5 * r["reach"])
    assert r["p_min"] == pytest.approx(4 * CADENCE)  # PMIN_CADENCES = 4 (Rae, 2026-11-09)


def test_contiguous_reach_is_07_baseline_and_does_not_bridge_gaps():
    """Contiguous: R ~ 0.7 B. Two sectors 2 sectors apart: R is the single
    sector's (the reach does not bridge the gap)."""
    t, f = SAMPLINGS["contiguous_1sector"]
    assert searchable_range(t, f)["reach"] == pytest.approx(0.7 * 27.4, abs=2 * CADENCE)
    t2, f2 = _grid(82.2, [(27.4, 54.8)])
    assert searchable_range(t2, f2)["reach"] == pytest.approx(searchable_range(t, f)["reach"], abs=2 * CADENCE)


def test_gridded_and_compressed_inputs_agree_and_match_compute_acf():
    """NaN-gapped and gap-removed arrays give the same range, and the reach
    equals the one from compute_acf's own pair counts (untruncated)."""
    t, f = SAMPLINGS["gapped_2"]
    keep = np.isfinite(f)
    a, b = searchable_range(t, f), searchable_range(t[keep], f[keep])
    assert a["reach"] == b["reach"] and a["p_min"] == b["p_min"]
    rng = np.random.default_rng(0)
    fl = np.where(keep, rng.normal(size=len(t)), np.nan)
    lags, _, counts = compute_acf(t, fl, max_lag_frac=1.0, min_valid_frac=0.0, return_counts=True)
    assert reach_from_pair_counts(lags, counts) == pytest.approx(a["reach"])


def test_per_batch_and_degenerate_input():
    """A batch on its own gives that batch's range; < 2 finite points -> NaN."""
    t, f = SAMPLINGS["1-2-13"]
    late = t > 300
    assert searchable_range(t[late], f[late])["reach"] == pytest.approx(0.7 * 27.4, abs=2 * CADENCE)
    r = searchable_range(np.arange(5.0), np.array([1, np.nan, np.nan, np.nan, np.nan]))
    assert np.isnan(r["reach"]) and np.isnan(r["p_max"])


def _fit_inputs(period=3.0, n_sectors=1, seed=1):
    """A clean synthetic star and its Stage 2 ACF (with counts)."""
    lc = make_lightcurve(period, n_sectors=n_sectors, seed=seed)
    lags, acf, counts = compute_acf(lc.time, lc.flux, return_counts=True)
    return lc, lags, acf, counts


def test_auto_reach_matches_searchable_range():
    """max_period='auto' derives R from compute_acf's default counts; it
    equals searchable_range's R to within one cadence."""
    lc, lags, acf, counts = _fit_inputs()
    res = fit_rotation_period(lags, acf, [InitialGuess(P0=3.0, method="test")], acf_counts=counts)
    assert res.searchable_range["source"] == "acf_counts"
    assert res.searchable_range["reach"] == pytest.approx(searchable_range(lc.time, lc.flux)["reach"],
                                                          abs=1.01 * CADENCE)


def test_beyond_reach_candidate_skipped_but_recorded():
    """A candidate with P > R/2 is not fit but is kept in
    beyond_reach_candidates; in-reach candidates fit exactly as with no range."""
    lc, lags, acf, counts = _fit_inputs()
    p_max = 0.5 * reach_from_pair_counts(lags, counts)
    far = 0.8 * reach_from_pair_counts(lags, counts)
    guesses = [InitialGuess(P0=3.0, method="a", strength=1.0), InitialGuess(P0=far, method="b", strength=1.0)]
    res = fit_rotation_period(lags, acf, guesses, acf_counts=counts)
    old = fit_rotation_period(lags, acf, guesses, acf_counts=counts, max_period=None)
    assert [c.period for c in res.beyond_reach_candidates] == [far]
    assert res.beyond_reach_candidates[0].beyond_reach and res.beyond_reach_candidates[0].fit is None
    assert all(c.period <= p_max for c in res.candidates + res.failed_candidates)
    assert any(c.period == far for c in old.candidates + old.failed_candidates)
    assert res.n_candidates_tried == old.n_candidates_tried - 1
    assert res.success and res.best_fit.P == old.best_fit.P  # bit-identical in-reach fit
    assert old.searchable_range is None and old.beyond_reach_candidates == []


def test_all_beyond_reach_declines_and_explicit_ranges():
    """Only beyond-reach candidates -> decline kind 'all_beyond_reach'; a
    float or a searchable_range dict can set the range explicitly."""
    lc, lags, acf, counts = _fit_inputs()
    far = 0.8 * reach_from_pair_counts(lags, counts)
    res = fit_rotation_period(lags, acf, [InitialGuess(P0=far, method="b")], acf_counts=counts)
    assert not res.success and res.decline_reason["kind"] == "all_beyond_reach"
    assert len(res.beyond_reach_candidates) == 1
    res = fit_rotation_period(lags, acf, [InitialGuess(P0=3.0, method="a")], acf_counts=counts, max_period=2.0)
    assert res.searchable_range["source"] == "explicit" and res.decline_reason["kind"] == "all_beyond_reach"
    sr = searchable_range(lc.time, lc.flux)
    res = fit_rotation_period(lags, acf, [InitialGuess(P0=3.0, method="a")], max_period=sr)
    assert res.searchable_range["source"] == "searchable_range" and res.searchable_range["p_max"] == sr["p_max"]
    # no counts and 'auto' -> no range (pre-T16c behavior for count-less callers)
    assert fit_rotation_period(lags, acf, [InitialGuess(P0=far, method="b")]).searchable_range is None


def test_reach_feature():
    """P_over_reach = P / R; NaN when undefined."""
    assert _reach_features(5.0, 20.0)["P_over_reach"] == pytest.approx(0.25)
    assert np.isnan(_reach_features(5.0, np.nan)["P_over_reach"])
    assert np.isnan(_reach_features(np.nan, 20.0)["P_over_reach"])


@pytest.mark.parametrize("name", sorted(SAMPLINGS))
def test_reach_matches_frozen_evaluation_reference(name):
    """The pipeline's searchable_range R equals the scorecard's frozen
    evaluation.acf_reach on the same sampling (P22a; both define R by a 0.3
    valid-pair fraction, D4). If this fails, the pipeline and the scoring
    disagree about which stars are searchable."""
    from evaluation import acf_reach
    t, f = SAMPLINGS[name]
    assert searchable_range(t, f)["reach"] == pytest.approx(acf_reach(t, f), abs=1e-9)
