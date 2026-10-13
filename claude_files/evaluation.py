"""
evaluation.py

Single source of truth for "was this period measurement right?".

Every script that scores a measured or candidate period against a known
(injected) period should import from here rather than re-implementing its
own tolerance check, so that changing the convention is a one-line edit
and every report in the project stays comparable.

Conventions (set by Rae, 2026-10-13)
------------------------------------
- REL_TOL = 0.10: a measurement is CORRECT if |P_meas - P_true| / P_true
  <= 0.10. (Previously 0.15 across the batch scripts.)
- The factor-of-two aliases are their own outcome category, NOT lumped
  in with generic failures: HALF if P_meas is within REL_TOL of P_true/2,
  DOUBLE if within REL_TOL of 2*P_true. These are the dominant, physically
  meaningful failure modes for ACF-based rotation measurements (double-dip
  light curves from spots on opposite hemispheres -> P/2; ACF peaks that
  skip a cycle -> 2P), and the ML ranking stage needs to distinguish them
  from "just wrong".
- WRONG: a finite measurement that matches none of the above.
- NONE: no measurement (NaN/None, or the pipeline declined to answer).

Note on tolerances: with REL_TOL=0.10 the CORRECT, HALF and DOUBLE windows
([0.9, 1.1], [0.45, 0.55], [1.8, 2.2] in units of P_true) never overlap,
so the classification is unambiguous. Any REL_TOL >= 1/3 would make the
windows overlap; classify_period raises in that case.

The ML training label tolerance (ml_features.extract_candidate_features
`label_rel_tol`, default 0.03 on the *refined, fitted* period) is a
separate, deliberately tighter knob and is not changed by this module.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

REL_TOL = 0.10

CORRECT = "correct"
HALF = "half"        # measured ~ P_true / 2
DOUBLE = "double"    # measured ~ 2 * P_true
WRONG = "wrong"
NONE = "none"

OUTCOMES = (CORRECT, HALF, DOUBLE, WRONG, NONE)
ALIAS_OUTCOMES = (HALF, DOUBLE)

# Period bins (days) used for all per-regime reporting. The <10 d regime is
# split finely because that's where recovery is currently weakest (see
# docs/PROJECT_CONTEXT.md open issue 1).
PERIOD_BIN_EDGES = (0.0, 1.0, 3.0, 10.0, 30.0, np.inf)
PERIOD_BIN_LABELS = ("<1d", "1-3d", "3-10d", "10-30d", ">30d")


def _is_missing(x) -> bool:
    """Return True if `x` is None or a non-finite number."""
    return x is None or not math.isfinite(float(x))


def classify_period(measured, true_period, rel_tol: float = REL_TOL) -> str:
    """Classify one period measurement against the known true period.

    Parameters
    ----------
    measured : float or None
        Measured/candidate period (days). None or NaN means "no answer".
    true_period : float
        Known (injected) period, days. Must be finite and > 0.
    rel_tol : float
        Relative tolerance applied to each target (P, P/2, 2P).

    Returns
    -------
    str
        One of CORRECT, HALF, DOUBLE, WRONG, NONE.
    """
    if rel_tol >= 1.0 / 3.0:
        raise ValueError(f"rel_tol={rel_tol} makes the correct/alias windows overlap; use < 1/3.")
    if _is_missing(true_period) or float(true_period) <= 0:
        raise ValueError(f"true_period must be finite and positive, got {true_period!r}.")
    if _is_missing(measured):
        return NONE
    m, p = float(measured), float(true_period)
    if abs(m - p) / p <= rel_tol:
        return CORRECT
    if abs(m - p / 2) / (p / 2) <= rel_tol:
        return HALF
    if abs(m - 2 * p) / (2 * p) <= rel_tol:
        return DOUBLE
    return WRONG


def best_outcome(candidates: Iterable, true_period, rel_tol: float = REL_TOL) -> tuple:
    """Score a ranked candidate list: did the true period (or an alias) appear?

    CORRECT beats any alias, and an alias beats WRONG, regardless of rank;
    within the winning category the best (lowest) rank is reported.

    Parameters
    ----------
    candidates : iterable of float
        Candidate periods, best-first (index 0 = rank 1).
    true_period : float
        Known (injected) period, days.
    rel_tol : float
        Passed to classify_period.

    Returns
    -------
    (outcome, rank) : (str, int or None)
        outcome is CORRECT, HALF, DOUBLE, WRONG (non-empty list, no match)
        or NONE (empty list). rank is 1-based, or None if no match.
    """
    cands = [c for c in candidates if not _is_missing(c)]
    if not cands:
        return NONE, None
    priority = {CORRECT: 0, HALF: 1, DOUBLE: 1}
    best = (WRONG, None)
    best_key = (2, math.inf)
    for i, c in enumerate(cands, start=1):
        out = classify_period(c, true_period, rel_tol)
        if out in priority:
            key = (priority[out], i)
            if key < best_key:
                best, best_key = (out, i), key
    return best


def period_bin(true_period: float) -> str:
    """Return the PERIOD_BIN_LABELS label for `true_period` (days)."""
    idx = int(np.searchsorted(PERIOD_BIN_EDGES, float(true_period), side="right")) - 1
    idx = min(max(idx, 0), len(PERIOD_BIN_LABELS) - 1)
    return PERIOD_BIN_LABELS[idx]


def outcome_fractions(outcomes: Iterable[str]) -> dict:
    """Fraction of each outcome category in `outcomes` (plus 'n' and 'alias').

    Parameters
    ----------
    outcomes : iterable of str
        Values from OUTCOMES.

    Returns
    -------
    dict
        {'n': int, 'correct': f, 'half': f, 'double': f, 'alias': f,
         'wrong': f, 'none': f}. Fractions are NaN when n == 0.
    """
    outs = list(outcomes)
    n = len(outs)
    res: dict = {"n": n}
    for o in OUTCOMES:
        res[o] = outs.count(o) / n if n else float("nan")
    res["alias"] = (res[HALF] + res[DOUBLE]) if n else float("nan")
    return res
