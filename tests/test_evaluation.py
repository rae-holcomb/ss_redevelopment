"""Tests for evaluation.py -- the project-wide scoring convention."""

import math

import pytest

from evaluation import (CORRECT, DOUBLE, HALF, NONE, REL_TOL, WRONG, best_outcome,
                        classify_period, outcome_fractions, period_bin)


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
    """Fractions sum to 1 and alias = half + double."""
    f = outcome_fractions([CORRECT, CORRECT, HALF, DOUBLE, WRONG])
    assert f["n"] == 5
    assert f["correct"] == pytest.approx(0.4)
    assert f["alias"] == pytest.approx(0.4)
    assert math.isnan(outcome_fractions([])["correct"])
