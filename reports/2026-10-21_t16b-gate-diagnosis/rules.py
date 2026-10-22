"""
rules.py -- the alternative gate rules replayed offline for the T16b
trade-off table (none of them is a pipeline default; see README).

Each rule is (overrides, fallback_min, short_only):
    overrides : {gate: fn(candidates) -> bool Series} replacing that gate's test
    fallback_min : emulate half_lag_fallback_min (option D) or None
    short_only : apply only to candidates of stars with <= 2 sectors
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _contrast(c: pd.DataFrame) -> pd.Series:
    """Mean half-lag contrast (acf_half_lag_contrast), NaN -> -inf."""
    return c.half_lag_contrast.fillna(-np.inf)


def _strong(c: pd.DataFrame, x: float = 0.1) -> pd.Series:
    """Structured-ACF test: every half-lag term > 0 or mean contrast >= x."""
    return (c.half_lag_consistency_value.fillna(-np.inf) > 0) | (_contrast(c) >= x)


def few_height(th: float):
    """few_peaks_height with a different threshold."""
    return {"few_peaks_height": lambda c: c.few_peaks_height_value.fillna(-np.inf) >= th}


def few_or_contrast(x: float):
    """few_peaks_height passes at the current 0.4, OR when mean contrast >= x."""
    return {"few_peaks_height": lambda c: (c.few_peaks_height_value.fillna(-np.inf) >= c.few_peaks_height_threshold)
            | (_contrast(c) >= x)}


def _many(c: pd.DataFrame, many_only: bool) -> pd.Series:
    """True where a contrast-conditional relaxation may apply: every candidate,
    or (many_only) only candidates with >= 4 teeth (not few_peaks), since with
    2-3 teeth the mean contrast has 1-2 terms and red noise reaches ~0.4."""
    return ~c.few_peaks.astype(bool) if many_only else pd.Series(True, index=c.index)


def snr_if_contrast(th: float, x: float = 0.1, many_only: bool = False):
    """mean_height_snr >= 1 as now, or >= th when mean contrast >= x."""
    return {"mean_height_snr": lambda c: (c.mean_height_snr_value >= c.mean_height_snr_threshold)
            | ((c.mean_height_snr_value >= th) & (_contrast(c) >= x) & _many(c, many_only))}


def vertex_if_contrast(th: float, x: float = 0.1, many_only: bool = False):
    """frac_vertex_in_window >= 0.8 as now, or >= th when mean contrast >= x."""
    return {"frac_vertex_in_window": lambda c: (c.frac_vertex_in_window_value >= c.frac_vertex_in_window_threshold)
            | ((c.frac_vertex_in_window_value >= th) & (_contrast(c) >= x) & _many(c, many_only))}


def npeaks_if_contrast(x: float = 0.1):
    """n_peaks as now, or one peak short (never below 2) when mean contrast >= x."""
    return {"n_peaks": lambda c: (c.n_peaks_value >= c.n_peaks_threshold)
            | ((c.n_peaks_value >= np.maximum(c.n_peaks_threshold - 1, 2)) & (_contrast(c) >= x))}


def contrast_override(x: float):
    """Waive every comb-shape gate and the few-peak guard when mean contrast >= x
    and every half-lag term > 0 (the strongest ACF-structure evidence)."""
    def ok(c):
        return (c.half_lag_consistency_value.fillna(-np.inf) > 0) & (_contrast(c) >= x)
    out = {}
    for g in ("n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window", "few_peaks_height"):
        out[g] = (lambda gg: (lambda c: c[f"{gg}_pass"] | ok(c)))(g)
    return out


def merge(*dicts) -> dict:
    """Combine override dicts (later ones win per gate)."""
    out = {}
    for d in dicts:
        out.update(d)
    return out


PROPOSED = merge(few_or_contrast(0.1), snr_if_contrast(0.5), vertex_if_contrast(0.5), npeaks_if_contrast())
# Refined after inspecting the noise false positives PROPOSED admits (all red
# noise, 2-3 teeth): lower the few-peak height floor to 0.25 instead of
# letting contrast override it, and restrict contrast-conditional
# relaxations to candidates with >= 4 teeth.
PROPOSED2 = merge(few_height(0.25), snr_if_contrast(0.5, many_only=True), vertex_if_contrast(0.5, many_only=True))

RULES = {
    "current": ({}, None, False),
    "option D (fallback contrast>0)": ({}, 0.0, False),
    "few height >= 0.25": (few_height(0.25), None, False),
    "few height >= 0.25 (short only)": (few_height(0.25), None, True),
    "few: height>=0.4 or contrast>=0.1": (few_or_contrast(0.1), None, False),
    "snr>=0.5 if contrast>=0.1": (snr_if_contrast(0.5), None, False),
    "vertex>=0.5 if contrast>=0.1": (vertex_if_contrast(0.5), None, False),
    "n_peaks-1 if contrast>=0.1": (npeaks_if_contrast(), None, False),
    "override: all terms>0 & contrast>=0.1": (contrast_override(0.1), None, False),
    "override: all terms>0 & contrast>=0.2": (contrast_override(0.2), None, False),
    "PROPOSED (4 contrast-conditional relaxations)": (PROPOSED, None, False),
    "PROPOSED (short only)": (PROPOSED, None, True),
    "snr>=0.5 if contrast>=0.1 & >=4 teeth": (snr_if_contrast(0.5, many_only=True), None, False),
    "vertex>=0.5 if contrast>=0.1 & >=4 teeth": (vertex_if_contrast(0.5, many_only=True), None, False),
    "PROPOSED2 (few>=0.25; snr/vertex if contrast & >=4 teeth)": (PROPOSED2, None, False),
    "PROPOSED2 (short only)": (PROPOSED2, None, True),
    "snr>=0.5 everywhere (no contrast cond.)": ({"mean_height_snr": lambda c: c.mean_height_snr_value >= 0.5}, None, False),
    "half-lag: contrast>=0.0 (min term ignored)": ({"half_lag_consistency": lambda c: _contrast(c) >= 0.0}, None, False),
}
