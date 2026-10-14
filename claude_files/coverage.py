"""
coverage.py

Sector-coverage patterns for patchy-coverage studies (task T16).

Real TESS stars rarely get 13 consecutive sectors: of the 3,438 stars with
Cycle-1 data in `target_df_shortform.csv`, 74% have a single sector. This
module holds everything about *which sectors a star was observed in*, kept
separate from the pipeline itself:

- patterns: a pattern is a sorted tuple of 1-indexed sector numbers within
  the 13-sector SMARTS year, e.g. (1, 2, 13). Written as "1-2-13" in CSVs and
  dataset strings (`format_pattern` / `parse_pattern`).
- classes: `coverage_class` buckets a pattern into one of COVERAGE_CLASSES.
- library: `build_library` counts how many real stars share each pattern
  (tools/build_coverage_library.py writes it to
  datasets/coverage/cycle1_patterns.csv; `load_library` reads it back).
- sampling: `sample_patterns` draws one pattern per star for the scorecard's
  `--coverage` option.
- descriptors: `coverage_descriptors` measures what a star's (time, flux)
  arrays actually cover (baseline, longest contiguous stretch, largest gap,
  duty cycle).

The same pattern means the same thing for synthetic stars
(`synthetic.make_lightcurve(sectors=...)`) and SMARTS FITS files
(`preprocessing.load_smarts_fits(sectors=...)`): sector s is orbits 2s-1 and
2s of the year, unobserved sectors are dropped, and the time grid starts at
the first kept cadence.
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

N_SECTORS_YEAR = 13  # sectors in one SMARTS year (= TESS Cycle 1, sectors 1-13)

# Coverage classes, in display order. See coverage_class for definitions.
COVERAGE_CLASSES = ("single", "consecutive-2", "gapped-2", "3-5", "6-12", "full")

# Gaps longer than this split the light curve into separate "contiguous"
# stretches (coverage_descriptors). TESS orbit/sector downlink gaps are
# ~1 d (a few Cycle-1 interruptions reach ~2-4 d); a missing sector is
# >= ~27 d. 5 d sits safely between the two.
GAP_SPLIT_DAYS = 5.0

DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "datasets" / "coverage" / "cycle1_patterns.csv"

# --coverage specs that are not a single class
FULL = "full"
REAL_WEIGHTED = "real-weighted"


def normalize_pattern(sectors: Iterable[int], n_total: int = N_SECTORS_YEAR) -> tuple:
    """Return `sectors` as a sorted tuple of unique ints, validated to lie in 1..n_total.

    Parameters
    ----------
    sectors : iterable of sector numbers (1-indexed).
    n_total : number of sectors in the year.

    Returns
    -------
    tuple of int

    Raises
    ------
    ValueError if empty or out of range.
    """
    pat = tuple(sorted(set(int(s) for s in sectors)))
    if not pat:
        raise ValueError("empty sector pattern")
    if pat[0] < 1 or pat[-1] > n_total:
        raise ValueError(f"sector pattern {pat} outside 1..{n_total}")
    return pat


def format_pattern(sectors: Iterable[int]) -> str:
    """Pattern -> canonical string, e.g. (1, 2, 13) -> "1-2-13"."""
    return "-".join(str(s) for s in normalize_pattern(sectors, n_total=10**6))


def parse_pattern(text) -> tuple:
    """Parse a pattern string ("1-2-13", "1,2,13", "1 2 13" or "[1, 2, 13]") to a tuple.

    Also accepts an iterable of ints, so callers can pass either form.
    """
    if isinstance(text, (list, tuple, np.ndarray)):
        return normalize_pattern(text, n_total=10**6)
    s = str(text).strip().strip("[]()")
    for sep in (",", " "):
        s = s.replace(sep, "-")
    return normalize_pattern([int(x) for x in s.split("-") if x], n_total=10**6)


def is_consecutive(sectors: Sequence[int]) -> bool:
    """True if the sector numbers form one unbroken run (a single sector counts)."""
    pat = normalize_pattern(sectors, n_total=10**6)
    return pat[-1] - pat[0] == len(pat) - 1


def coverage_class(sectors: Sequence[int], n_total: int = N_SECTORS_YEAR) -> str:
    """Bucket a sector pattern into one of COVERAGE_CLASSES.

    - single         : 1 sector
    - consecutive-2  : 2 adjacent sectors (s, s+1)
    - gapped-2       : 2 non-adjacent sectors. Note (1, 13) is the commonest
                       gapped pair in Cycle 1: adjacent on the sky, ~11 months
                       apart in time.
    - 3-5            : 3-5 sectors, consecutive or not
    - 6-12           : 6-12 sectors (in Cycle 1 mostly near-full years
                       missing one or two sectors)
    - full           : all n_total sectors

    Parameters
    ----------
    sectors : the pattern.
    n_total : sectors in the year (13).

    Returns
    -------
    str
    """
    pat = normalize_pattern(sectors, n_total)
    n = len(pat)
    if n == n_total:
        return "full"
    if n == 1:
        return "single"
    if n == 2:
        return "consecutive-2" if pat[1] - pat[0] == 1 else "gapped-2"
    if n <= 5:
        return "3-5"
    return "6-12"


def cycle1_patterns_from_targets(target_df: pd.DataFrame, n_total: int = N_SECTORS_YEAR) -> pd.Series:
    """Per-star Cycle-1 pattern from a target table (as ss_claude_sandbox.ipynb cell 10 did).

    Parameters
    ----------
    target_df : DataFrame with a `sec` column holding each star's observed
        sector list (a list, or its string repr as in target_df_shortform.csv).
    n_total : keep sectors <= n_total (Cycle 1 = sectors 1-13).

    Returns
    -------
    pd.Series of tuples (empty tuple for stars with no Cycle-1 sector),
    indexed like target_df.
    """
    sec = target_df["sec"].apply(lambda x: literal_eval(x) if isinstance(x, str) else list(x))
    return sec.apply(lambda s: tuple(sorted(set(int(v) for v in s if 1 <= int(v) <= n_total))))


def build_library(target_df: pd.DataFrame, n_total: int = N_SECTORS_YEAR) -> pd.DataFrame:
    """Count how many stars share each Cycle-1 sector pattern.

    Stars with no Cycle-1 sector are dropped (they were observed only later).

    Parameters
    ----------
    target_df : see cycle1_patterns_from_targets.
    n_total : sectors in the year.

    Returns
    -------
    DataFrame with columns pattern (str, "1-2-13"), n_stars, frac_stars,
    n_sectors, consecutive, class, first_sector, last_sector, span_sectors,
    max_missing_run; sorted by class order, then n_stars descending, then
    pattern.
    """
    pats = cycle1_patterns_from_targets(target_df, n_total)
    pats = pats[pats.str.len() > 0]
    counts = pats.value_counts()
    rows = []
    for pat, n in counts.items():
        rows.append(dict(
            pattern=format_pattern(pat), n_stars=int(n), n_sectors=len(pat),
            consecutive=is_consecutive(pat), **{"class": coverage_class(pat, n_total)},
            first_sector=pat[0], last_sector=pat[-1], span_sectors=pat[-1] - pat[0] + 1,
            max_missing_run=int(max(np.diff(pat)) - 1) if len(pat) > 1 else 0,
        ))
    lib = pd.DataFrame(rows)
    lib.insert(2, "frac_stars", lib["n_stars"] / lib["n_stars"].sum())
    order = {c: i for i, c in enumerate(COVERAGE_CLASSES)}
    lib = (lib.assign(_c=lib["class"].map(order))
              .sort_values(["_c", "n_stars", "pattern"], ascending=[True, False, True])
              .drop(columns="_c").reset_index(drop=True))
    return lib


def load_library(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the committed coverage library (default datasets/coverage/cycle1_patterns.csv)."""
    lib = pd.read_csv(path or DEFAULT_LIBRARY, comment="#")
    lib["pattern"] = lib["pattern"].astype(str)
    return lib


def valid_coverage_specs() -> list:
    """Every value accepted by sample_patterns' `spec` (besides 'pattern:<p>')."""
    return [FULL, REAL_WEIGHTED] + [c for c in COVERAGE_CLASSES if c != FULL]


def sample_patterns(spec: str, n: int, seed: int = 0, library: Optional[pd.DataFrame] = None,
                    weighting: str = "uniform") -> list:
    """Draw one sector pattern per star.

    Parameters
    ----------
    spec : what to draw from:
        - 'full'           : every star gets None (= all sectors, the legacy default);
        - 'real-weighted'  : any library pattern, weighted by its n_stars;
        - a class name     : a pattern of that class (see coverage_class);
        - 'pattern:1-2-13' : that exact pattern for every star.
    n : number of stars.
    seed : RNG seed (independent of the synthetic-suite seed, so the same
        stars can be re-run under different coverage draws).
    library : coverage library (default: load_library()).
    weighting : for a class spec, 'uniform' (each pattern in the class equally
        likely) or 'frequency' (weighted by n_stars). Ignored otherwise.

    Returns
    -------
    list of length n: a tuple per star, or None for 'full'.
    """
    if spec == FULL:
        return [None] * n
    if spec.startswith("pattern:"):
        return [parse_pattern(spec[len("pattern:"):])] * n
    lib = load_library() if library is None else library
    if spec == REAL_WEIGHTED:
        pool, w = lib, lib["n_stars"].to_numpy(float)
    elif spec in COVERAGE_CLASSES:
        pool = lib[lib["class"] == spec]
        if pool.empty:
            raise ValueError(f"no library patterns in class {spec!r}")
        if weighting == "uniform":
            w = np.ones(len(pool))
        elif weighting == "frequency":
            w = pool["n_stars"].to_numpy(float)
        else:
            raise ValueError(f"weighting must be 'uniform' or 'frequency', got {weighting!r}")
    else:
        raise ValueError(f"unknown coverage spec {spec!r}; expected one of "
                         f"{valid_coverage_specs()} or 'pattern:<sectors>'")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, p=w / w.sum())
    pats = pool["pattern"].to_numpy()
    return [parse_pattern(pats[i]) for i in idx]


def coverage_descriptors(time: np.ndarray, flux: np.ndarray, sectors: Sequence[int],
                         gap_split_days: float = GAP_SPLIT_DAYS) -> dict:
    """Describe what a light curve actually covers.

    Parameters
    ----------
    time : evenly sampled times (days), as from load_smarts_fits / synthetic.
    flux : flux with NaN at missing cadences.
    sectors : the sectors present (pattern), used for n_sectors / sectors /
        coverage_class.
    gap_split_days : gaps longer than this end a contiguous stretch.

    Returns
    -------
    dict with
        n_sectors, sectors ("1-2-13"), coverage_class,
        total_baseline_days      : last minus first finite cadence;
        longest_contiguous_days  : longest stretch with no gap > gap_split_days
                                   (first to last finite cadence of the stretch);
        largest_gap_days         : largest difference between consecutive finite
                                   cadences (0 if < 2 points);
        duty_cycle               : finite cadences / cadences spanned by the
                                   baseline (first to last finite cadence, inclusive).
    """
    pat = normalize_pattern(sectors, n_total=10**6)
    t = np.asarray(time, float)[np.isfinite(np.asarray(flux, float))]
    out = dict(n_sectors=len(pat), sectors=format_pattern(pat),
               coverage_class=coverage_class(pat) if pat[-1] <= N_SECTORS_YEAR else None)
    if len(t) < 2:
        return out | dict(total_baseline_days=0.0, longest_contiguous_days=0.0,
                          largest_gap_days=0.0, duty_cycle=float(len(t) > 0))
    dt = np.diff(t)
    cadence = float(np.median(dt))
    breaks = np.flatnonzero(dt > gap_split_days)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t) - 1]
    baseline = float(t[-1] - t[0])
    return out | dict(
        total_baseline_days=baseline,
        longest_contiguous_days=float(np.max(t[ends] - t[starts])),
        largest_gap_days=float(dt.max()),
        duty_cycle=float(len(t) / (round(baseline / cadence) + 1)),
    )
