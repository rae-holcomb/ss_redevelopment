"""
evaluation.py

Single source of truth for "was this period measurement right?".

Every script that scores a measured or candidate period against a known
(injected) period should import from here rather than re-implementing its
own tolerance check, so that changing the convention is a one-line edit
and every report in the project stays comparable.

Conventions (set by Rae, 2026-10-13)
------------------------------------
- REL_TOL = 0.10: a measurement is CORRECT if |P_meas - P_eq| / P_eq
  <= 0.10, where P_eq is the injected *equatorial* period (SMARTS
  `PERIOD`). (Previously 0.15 across the batch scripts.)
- DIFFROT (also a success): with latitudinal differential rotation, spots
  at latitude rotate with P(lat) = P_eq / (1 - shear sin^2 lat) (butterpy
  convention; see docs/SMARTS_REFERENCE.md). A measurement within REL_TOL
  of the range of periods spanned by the star's active latitude band is a
  genuine physical rotation period and counts as success, but is tracked
  separately from CORRECT so we can see how often it happens. We only know
  the band on simulated data; the point is to make the tool robust to
  differentially rotating stars, not to require P_eq specifically.
- The factor-of-two aliases are their own outcome category, NOT lumped
  in with generic failures: HALF if P_meas is within REL_TOL of P_eq/2 (or
  of half the active-band period range), DOUBLE likewise for 2x. These are
  the dominant, physically meaningful failure modes for ACF-based rotation
  measurements (double-dip light curves -> P/2; ACF peaks that skip a
  cycle -> 2P), and the ML ranking stage needs to distinguish them.
- WRONG: a finite measurement that matches none of the above.
- NONE: no measurement (NaN/None, or the pipeline declined to answer).
- "success" = CORRECT + DIFFROT. Goals are set on success.

Precedence when windows overlap (possible for large shear, where the
active-band range can reach 2x P_eq): CORRECT > DIFFROT > HALF/DOUBLE >
WRONG. Without a band, the CORRECT/HALF/DOUBLE windows never overlap for
REL_TOL < 1/3; classify_period raises for larger tolerances.

The ML training label tolerance (ml_features.extract_candidate_features
`label_rel_tol`, default 0.03 on the *refined, fitted* period) is a
separate, deliberately tighter knob and is not changed by this module.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

REL_TOL = 0.10

CORRECT = "correct"
DIFFROT = "diffrot"  # within REL_TOL of the active-latitude-band period range
HALF = "half"        # measured ~ P / 2
DOUBLE = "double"    # measured ~ 2 * P
WRONG = "wrong"
NONE = "none"

OUTCOMES = (CORRECT, DIFFROT, HALF, DOUBLE, WRONG, NONE)
SUCCESS_OUTCOMES = (CORRECT, DIFFROT)
ALIAS_OUTCOMES = (HALF, DOUBLE)

# Period bins (days) used for all per-regime reporting. The <10 d regime is
# split finely because that's where recovery is currently weakest (see
# docs/PROJECT_CONTEXT.md open issue 1).
PERIOD_BIN_EDGES = (0.0, 1.0, 3.0, 10.0, 30.0, np.inf)
PERIOD_BIN_LABELS = ("<1d", "1-3d", "3-10d", "10-30d", ">30d")


def _is_missing(x) -> bool:
    """Return True if `x` is None or a non-finite number."""
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def latitude_period(p_eq: float, shear: float, lat_deg) -> np.ndarray:
    """Rotation period at latitude `lat_deg` under butterpy's sin^2 law.

    P(lat) = P_eq / (1 - shear * sin^2(lat)); shear > 0 is solar-like
    (higher latitudes rotate slower). Returns inf where the denominator is
    <= 0 (shear >= 1 at the pole: that latitude doesn't rotate).

    Parameters
    ----------
    p_eq : equatorial period, days.
    shear : SMARTS `DIFFROT` (delta_omega / omega_eq).
    lat_deg : latitude(s) in degrees (sign irrelevant).

    Returns
    -------
    np.ndarray of periods (days), same shape as lat_deg.
    """
    s2 = np.sin(np.radians(np.asarray(lat_deg, dtype=float))) ** 2
    denom = 1.0 - float(shear) * s2
    with np.errstate(divide="ignore"):
        return np.where(denom > 0, float(p_eq) / np.where(denom > 0, denom, 1.0), np.inf)


def active_band_period_range(p_eq: float, shear: float, minlat_deg: float, maxlat_deg: float) -> tuple:
    """(P_lo, P_hi): range of rotation periods across the active latitude band.

    Uses the SMARTS `MINLAT`/`MAXLAT` average-emergence-latitude bounds
    directly (not butterpy's +/-7 deg spillover), so the success window is
    where most spot flux lives rather than its rare extremes. Since P(lat)
    is monotonic in |lat|, the extremes are at the band edges.

    Parameters
    ----------
    p_eq : equatorial period (SMARTS `PERIOD`), days.
    shear : SMARTS `DIFFROT`.
    minlat_deg, maxlat_deg : SMARTS `MINLAT`, `MAXLAT`, degrees.

    Returns
    -------
    (P_lo, P_hi) with P_lo <= P_hi; P_hi may be inf for extreme shear.
    """
    p = latitude_period(p_eq, shear, [minlat_deg, maxlat_deg])
    return float(np.min(p)), float(np.max(p))


def _in_window(m: float, center: float, rel_tol: float) -> bool:
    """True if m is within rel_tol (relative) of center."""
    return abs(m - center) / center <= rel_tol


def _in_range(m: float, lo: float, hi: float, rel_tol: float) -> bool:
    """True if m lies in [lo * (1 - rel_tol), hi * (1 + rel_tol)]."""
    return lo * (1 - rel_tol) <= m <= hi * (1 + rel_tol)


def classify_period(measured, true_period, rel_tol: float = REL_TOL,
                    period_range: Optional[tuple] = None) -> str:
    """Classify one period measurement against the known true period.

    Parameters
    ----------
    measured : float or None
        Measured/candidate period (days). None or NaN means "no answer".
    true_period : float
        Known equatorial period P_eq, days. Must be finite and > 0.
    rel_tol : float
        Relative tolerance applied to each target window.
    period_range : (P_lo, P_hi) or None
        Active-latitude-band period range from active_band_period_range.
        None (or non-finite) disables the DIFFROT category, e.g. for
        solid-body stars or data without that truth.

    Returns
    -------
    str
        One of CORRECT, DIFFROT, HALF, DOUBLE, WRONG, NONE (see module
        docstring for precedence).
    """
    if rel_tol >= 1.0 / 3.0:
        raise ValueError(f"rel_tol={rel_tol} makes the correct/alias windows overlap; use < 1/3.")
    if _is_missing(true_period) or float(true_period) <= 0:
        raise ValueError(f"true_period must be finite and positive, got {true_period!r}.")
    if _is_missing(measured):
        return NONE
    m, p = float(measured), float(true_period)
    has_range = period_range is not None and not _is_missing(period_range[0])
    if has_range:
        lo, hi = float(period_range[0]), float(period_range[1])
    if _in_window(m, p, rel_tol):
        return CORRECT
    if has_range and _in_range(m, lo, hi, rel_tol):
        return DIFFROT
    if _in_window(m, p / 2, rel_tol) or (has_range and _in_range(m, lo / 2, hi / 2, rel_tol)):
        return HALF
    if _in_window(m, 2 * p, rel_tol) or (has_range and _in_range(m, 2 * lo, 2 * hi, rel_tol)):
        return DOUBLE
    return WRONG


def best_outcome(candidates: Iterable, true_period, rel_tol: float = REL_TOL,
                 period_range: Optional[tuple] = None) -> tuple:
    """Score a ranked candidate list: did the true period (or an alias) appear?

    CORRECT beats DIFFROT beats any alias beats WRONG, regardless of rank;
    within the winning category the best (lowest) rank is reported.

    Parameters
    ----------
    candidates : iterable of float
        Candidate periods, best-first (index 0 = rank 1).
    true_period : float
        Known equatorial period, days.
    rel_tol : float
        Passed to classify_period.
    period_range : (P_lo, P_hi) or None
        Passed to classify_period.

    Returns
    -------
    (outcome, rank) : (str, int or None)
        outcome is CORRECT, DIFFROT, HALF, DOUBLE, WRONG (non-empty list,
        no match) or NONE (empty list). rank is 1-based, or None if no match.
    """
    cands = [c for c in candidates if not _is_missing(c)]
    if not cands:
        return NONE, None
    priority = {CORRECT: 0, DIFFROT: 1, HALF: 2, DOUBLE: 2}
    best = (WRONG, None)
    best_key = (3, math.inf)
    for i, c in enumerate(cands, start=1):
        out = classify_period(c, true_period, rel_tol, period_range)
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
    """Fraction of each outcome category in `outcomes`, plus 'n', 'success'
    (correct + diffrot) and 'alias' (half + double).

    Parameters
    ----------
    outcomes : iterable of str
        Values from OUTCOMES.

    Returns
    -------
    dict
        {'n': int, 'success': f, 'correct': f, 'diffrot': f, 'half': f,
         'double': f, 'alias': f, 'wrong': f, 'none': f}. Fractions are NaN
        when n == 0.
    """
    outs = list(outcomes)
    n = len(outs)
    res: dict = {"n": n}
    for o in OUTCOMES:
        res[o] = outs.count(o) / n if n else float("nan")
    res["success"] = (res[CORRECT] + res[DIFFROT]) if n else float("nan")
    res["alias"] = (res[HALF] + res[DOUBLE]) if n else float("nan")
    return res
