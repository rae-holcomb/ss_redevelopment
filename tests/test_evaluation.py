"""Tests for evaluation.py -- the project-wide scoring convention."""

import math

import pytest

import numpy as np

from evaluation import (CORRECT, DIFFROT, DOUBLE, HALF, NONE, REL_TOL, WRONG,
                        active_band_period_range, best_outcome, classify_period,
                        latitude_period, outcome_fractions, period_bin)


def test_rel_tol_convention():
    """Project convention (2026-10-13): 10% relative tolerance."""
    assert REL_TOL == 0.10


@pytest.mark.parametrize("measured,expected", [
    (10.0, CORRECT), (10.9, CORRECT), (9.1, CORRECT),
    (11.5, WRONG), (8.5, WRONG),
    (5.0, HALF), (5.4, HALF), (4.6, HALF),
    (20.0, DOUBLE), (21.5, DOUBLE), (18.5, DOUBLE),
    (3.33, WRONG), (30.0, WRONG),
    (None, NONE), (float("nan"), NONE),
])
def test_classify_period(measured, expected):
    """Each window around P, P/2 and 2P maps to its own category."""
    assert classify_period(measured, 10.0) == expected


def test_classify_rejects_overlapping_tolerance():
    """A tolerance wide enough to make the windows overlap is an error."""
    with pytest.raises(ValueError):
        classify_period(10.0, 10.0, rel_tol=0.4)


def test_classify_rejects_bad_truth():
    """True period must be finite and positive."""
    with pytest.raises(ValueError):
        classify_period(1.0, 0.0)


def test_best_outcome_prefers_correct_over_alias():
    """CORRECT wins even when an alias appears at a better rank."""
    assert best_outcome([5.0, 7.0, 10.2], 10.0) == (CORRECT, 3)


def test_best_outcome_alias_and_empty():
    """Alias beats wrong; an empty list is NONE."""
    assert best_outcome([3.0, 19.9], 10.0) == (DOUBLE, 2)
    assert best_outcome([3.0, 7.0], 10.0) == (WRONG, None)
    assert best_outcome([], 10.0) == (NONE, None)


def test_period_bin_edges():
    """Bin labels at representative periods, including the edges."""
    assert period_bin(0.5) == "<1d"
    assert period_bin(1.0) == "1-3d"
    assert period_bin(9.99) == "3-10d"
    assert period_bin(10.0) == "10-30d"
    assert period_bin(45.0) == ">30d"


def test_outcome_fractions():
    """alias = half + double; success = correct + diffrot."""
    f = outcome_fractions([CORRECT, DIFFROT, HALF, DOUBLE, WRONG])
    assert f["n"] == 5
    assert f["correct"] == pytest.approx(0.2)
    assert f["success"] == pytest.approx(0.4)
    assert f["alias"] == pytest.approx(0.4)
    assert math.isnan(outcome_fractions([])["correct"])


# --- differential rotation (butterpy / SMARTS convention) ---

def test_latitude_period_sign_convention():
    """butterpy: omega = omega_eq (1 - shear sin^2 lat). Positive shear is
    solar-like (slower at high latitude => longer period). Pinned here because
    Claytor+2022's printed Eq. 1 has the opposite sign (docs/SMARTS_REFERENCE.md)."""
    assert latitude_period(10.0, 0.5, 0.0) == pytest.approx(10.0)
    assert latitude_period(10.0, 0.5, 90.0) == pytest.approx(20.0)
    assert latitude_period(10.0, -1.0, 90.0) == pytest.approx(5.0)
    assert np.isinf(latitude_period(10.0, 1.0, 90.0))


def test_active_band_range_ordering():
    """Range is (lo, hi) for either sign of shear; zero shear collapses to P_eq."""
    lo, hi = active_band_period_range(10.0, 0.5, 10.0, 60.0)
    assert lo < hi and lo > 10.0
    lo, hi = active_band_period_range(10.0, -0.5, 10.0, 60.0)
    assert lo < hi < 10.0
    assert active_band_period_range(10.0, 0.0, 10.0, 60.0) == pytest.approx((10.0, 10.0))


def test_diffrot_outcome():
    """A spot-latitude period outside REL_TOL of P_eq but inside the band is DIFFROT;
    without band truth it stays WRONG; CORRECT still takes precedence."""
    band = active_band_period_range(10.0, 0.8, 30.0, 60.0)   # ~ (12.5, 25)
    assert classify_period(15.0, 10.0, period_range=band) == DIFFROT
    assert classify_period(15.0, 10.0) == WRONG
    assert classify_period(10.5, 10.0, period_range=band) == CORRECT
    assert classify_period(7.5, 10.0, period_range=band) == HALF      # half of a band period
    assert classify_period(40.0, 10.0, period_range=band) == DOUBLE   # double of a band period
    assert classify_period(3.0, 10.0, period_range=band) == WRONG


def test_best_outcome_ranks_diffrot_below_correct():
    """CORRECT beats DIFFROT beats alias in a candidate list."""
    band = (12.0, 18.0)
    assert best_outcome([15.0, 10.1], 10.0, period_range=band) == (CORRECT, 2)
    assert best_outcome([5.0, 15.0], 10.0, period_range=band) == (DIFFROT, 2)
