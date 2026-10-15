"""
coverage.py

Sector-coverage patterns for patchy-coverage studies (task T16).

Real TESS stars rarely get 13 consecutive sectors: of the 3,438 stars with
Cycle-1 data in `target_df_shortform.csv`, 74% have a single sector. This
module holds everything about *which sectors a star was observed in*, kept
separate from the pipeline itself:

- cycles: TESS observing years 1-7 (CYCLE_SECTORS). Each star's sectors within
  one cycle form one "star-season" pattern, expressed relative to the cycle's
  first sector and fitted into the 13-sector SMARTS year (`fit_to_year`):
  cycles 4-6 have 14-16 sectors, so late patterns are shifted earlier and
  patterns spanning > 13 sectors are truncated to their best 13-sector window.

- patterns: a pattern is a sorted tuple of 1-indexed sector numbers within
  the 13-sector SMARTS year, e.g. (1, 2, 13). Written as "1-2-13" in CSVs and
  dataset strings (`format_pattern` / `parse_pattern`).
- classes: `coverage_class` buckets a pattern into one of COVERAGE_CLASSES.
- library: `build_library` counts how many real stars share each pattern
  per cycle, and `combine_libraries` pools cycles (tools/build_coverage_library.py
  writes datasets/coverage/cycle<N>_patterns.csv and all_cycles_patterns.csv;
  `load_library` reads them back; all_cycles is the default).
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
COVERAGE_CLASSES = ("single", "consecutive-2", "gapped-2", "consecutive-3-5", "gapped-3-5",
                    "consecutive-6-12", "gapped-6-12", "full")
# Coarser groups accepted by sample_patterns (uniform over all their patterns).
CLASS_GROUPS = {"3-5": ("consecutive-3-5", "gapped-3-5"), "6-12": ("consecutive-6-12", "gapped-6-12")}

# TESS observing cycles (years) as inclusive sector ranges, as in tess-point
# (tess_stars2px 0.9.2: Year 5 = S56-69, Year 6 = S70-83, Year 7 = S84-96).
# Cycle 4 has 16 sectors and cycles 5-6 have 14, more than a SMARTS year holds.
CYCLE_SECTORS = {1: (1, 13), 2: (14, 26), 3: (27, 39), 4: (40, 55), 5: (56, 69), 6: (70, 83), 7: (84, 96)}

# Gaps longer than this split the light curve into separate "contiguous"
# stretches (coverage_descriptors). TESS orbit/sector downlink gaps are
# ~1 d (a few Cycle-1 interruptions reach ~2-4 d); a missing sector is
# >= ~27 d. 5 d sits safely between the two.
GAP_SPLIT_DAYS = 5.0

LIBRARY_DIR = Path(__file__).resolve().parents[1] / "datasets" / "coverage"
# Default: patterns pooled over TESS cycles 1-7 (Rae, 2026-10-15: don't tune to Year 1's quirks).
DEFAULT_LIBRARY = LIBRARY_DIR / "all_cycles_patterns.csv"

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

    - single              : 1 sector
    - consecutive-2       : 2 adjacent sectors (s, s+1)
    - gapped-2            : 2 non-adjacent sectors. Note (1, 13) is the commonest
                            gapped pair in Cycle 1: adjacent on the sky, ~11
                            months apart in time.
    - consecutive-3-5     : 3-5 sectors in one unbroken run
    - gapped-3-5          : 3-5 sectors with at least one missing in between
    - consecutive-6-12    : 6-12 sectors in one unbroken run
    - gapped-6-12         : 6-12 sectors with gaps (in Cycle 1 mostly near-full
                            years missing one or two sectors)
    - full                : all n_total sectors

    The consecutive/gapped split of 3-5 and 6-12 is Rae's decision (2026-10-15).

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
    kind = "consecutive" if pat[-1] - pat[0] == n - 1 else "gapped"
    if n == 2:
        return f"{kind}-2"
    if n <= 5:
        return f"{kind}-3-5"
    return f"{kind}-6-12"


def cycle_patterns_from_targets(target_df: pd.DataFrame, cycle: int = 1) -> pd.Series:
    """Per-star raw pattern within one TESS cycle, relative to the cycle's first sector.

    Parameters
    ----------
    target_df : DataFrame with a `sec` column holding each star's observed
        sector list (a list, or its string repr as in target_df_shortform.csv).
    cycle : TESS cycle (key of CYCLE_SECTORS). Cycle 1 reproduces
        ss_claude_sandbox.ipynb cell 10 (sectors 1-13 kept as-is).

    Returns
    -------
    pd.Series of tuples of relative sector numbers (1 = the cycle's first
    sector; may exceed 13 for cycles 4-6), empty for stars not observed in
    the cycle; indexed like target_df.
    """
    first, last = CYCLE_SECTORS[cycle]
    sec = target_df["sec"].apply(lambda x: literal_eval(x) if isinstance(x, str) else list(x))
    return sec.apply(lambda s: tuple(sorted(set(int(v) - first + 1 for v in s if first <= int(v) <= last))))


def cycle1_patterns_from_targets(target_df: pd.DataFrame) -> pd.Series:
    """Per-star Cycle-1 pattern (shorthand for cycle_patterns_from_targets(df, 1))."""
    return cycle_patterns_from_targets(target_df, 1)


def fit_to_year(rel_pattern: Sequence[int], n_total: int = N_SECTORS_YEAR) -> tuple:
    """Fit a within-cycle pattern into the n_total-sector SMARTS year.

    - Patterns already inside 1..n_total are kept as they are ("none").
    - Patterns spanning <= n_total sectors but ending after n_total (cycles with
      14-16 sectors) are shifted earlier so they end at n_total, keeping their
      spacing ("shifted").
    - Patterns spanning > n_total sectors keep only the n_total-sector window
      holding the most observed sectors (earliest window on ties), re-based to
      start at 1 ("truncated"). Only cycles 4 and 6 have these.

    Parameters
    ----------
    rel_pattern : sector numbers relative to the cycle start (>= 1).
    n_total : sectors in the SMARTS year.

    Returns
    -------
    (pattern tuple, action str)
    """
    pat = tuple(sorted(set(int(s) for s in rel_pattern)))
    if pat[-1] <= n_total:
        return pat, "none"
    if pat[-1] - pat[0] + 1 <= n_total:
        shift = pat[-1] - n_total
        return tuple(s - shift for s in pat), "shifted"
    arr = np.array(pat)
    best = max(range(pat[0], pat[-1] - n_total + 2),
               key=lambda a: (np.sum((arr >= a) & (arr < a + n_total)), -a))
    kept = arr[(arr >= best) & (arr < best + n_total)]
    return tuple(int(s - best + 1) for s in kept), "truncated"


def _library_rows(patterns: pd.Series, n_total: int) -> pd.DataFrame:
    """Shared table construction: count `patterns` (tuples) and add the descriptive columns."""
    rows = []
    for pat, n in patterns.value_counts().items():
        rows.append(dict(
            pattern=format_pattern(pat), n_stars=int(n), n_sectors=len(pat),
            consecutive=is_consecutive(pat), **{"class": coverage_class(pat, n_total)},
            first_sector=pat[0], last_sector=pat[-1], span_sectors=pat[-1] - pat[0] + 1,
            max_missing_run=int(max(np.diff(pat)) - 1) if len(pat) > 1 else 0,
        ))
    lib = pd.DataFrame(rows)
    lib.insert(2, "frac_stars", lib["n_stars"] / lib["n_stars"].sum())
    return lib


def _sort_library(lib: pd.DataFrame) -> pd.DataFrame:
    """Sort by class order, then n_stars descending, then pattern."""
    order = {c: i for i, c in enumerate(COVERAGE_CLASSES)}
    return (lib.assign(_c=lib["class"].map(order))
               .sort_values(["_c", "n_stars", "pattern"], ascending=[True, False, True])
               .drop(columns="_c").reset_index(drop=True))


def build_library(target_df: pd.DataFrame, cycle: int = 1, n_total: int = N_SECTORS_YEAR) -> pd.DataFrame:
    """Count how many stars share each sector pattern within one TESS cycle.

    Stars not observed in the cycle are dropped. Patterns are relative to the
    cycle start and fitted into the SMARTS year with fit_to_year.

    Parameters
    ----------
    target_df : see cycle_patterns_from_targets.
    cycle : TESS cycle (1-7).
    n_total : sectors in the SMARTS year.

    Returns
    -------
    DataFrame with columns pattern (str, "1-2-13"), n_stars, frac_stars,
    n_sectors, consecutive, class, first_sector, last_sector, span_sectors,
    max_missing_run, n_shifted, n_truncated (stars whose raw pattern needed
    fit_to_year's shift / truncation), cycle; sorted by class order, then
    n_stars descending, then pattern.
    """
    raw = cycle_patterns_from_targets(target_df, cycle)
    raw = raw[raw.str.len() > 0]
    fitted = raw.apply(lambda p: fit_to_year(p, n_total))
    pats, actions = fitted.str[0], fitted.str[1]
    lib = _library_rows(pats, n_total)
    act = pd.crosstab(pats.apply(format_pattern), actions)
    for a in ("shifted", "truncated"):
        lib[f"n_{a}"] = lib["pattern"].map(act[a] if a in act else {}).fillna(0).astype(int)
    lib["cycle"] = cycle
    return _sort_library(lib)


def combine_libraries(libs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Pool per-cycle libraries into one.

    Parameters
    ----------
    libs : per-cycle libraries from build_library.

    Returns
    -------
    DataFrame with the build_library columns, where n_stars / frac_stars count
    star-seasons (a star observed in two cycles counts once per cycle), plus
    n_cycles (cycles in which the pattern occurs) and cycles ("1 2 5"); no
    `cycle` column. Sorted like build_library.
    """
    allc = pd.concat(libs, ignore_index=True)
    desc = ["n_sectors", "consecutive", "class", "first_sector", "last_sector", "span_sectors", "max_missing_run"]
    g = allc.groupby("pattern", sort=False)
    lib = g[desc].first()
    lib["n_stars"] = g["n_stars"].sum()
    lib["n_shifted"] = g["n_shifted"].sum()
    lib["n_truncated"] = g["n_truncated"].sum()
    lib["n_cycles"] = g["cycle"].nunique()
    lib["cycles"] = g["cycle"].apply(lambda c: " ".join(str(v) for v in sorted(c)))
    lib = lib.reset_index()
    lib.insert(2, "frac_stars", lib["n_stars"] / lib["n_stars"].sum())
    lib = lib[["pattern", "n_stars", "frac_stars", *desc, "n_shifted", "n_truncated", "n_cycles", "cycles"]]
    return _sort_library(lib)


def load_library(path: Optional[Path] = None) -> pd.DataFrame:
    """Read a committed coverage library (default datasets/coverage/all_cycles_patterns.csv)."""
    lib = pd.read_csv(path or DEFAULT_LIBRARY, comment="#")
    lib["pattern"] = lib["pattern"].astype(str)
    return lib


def valid_coverage_specs() -> list:
    """Every value accepted by sample_patterns' `spec` (besides 'pattern:<p>')."""
    return [FULL, REAL_WEIGHTED] + [c for c in COVERAGE_CLASSES if c != FULL] + list(CLASS_GROUPS)


def sample_patterns(spec: str, n: int, seed: int = 0, library: Optional[pd.DataFrame] = None,
                    weighting: str = "uniform") -> list:
    """Draw one sector pattern per star.

    Parameters
    ----------
    spec : what to draw from:
        - 'full'           : every star gets None (= all sectors, the legacy default);
        - 'real-weighted'  : any library pattern, weighted by its n_stars;
        - a class name     : a pattern of that class (see coverage_class);
        - '3-5' or '6-12'  : a pattern of either the consecutive or the gapped
                             class of that size (CLASS_GROUPS);
        - 'pattern:1-2-13' : that exact pattern for every star.
    n : number of stars.
    seed : RNG seed (independent of the synthetic-suite seed, so the same
        stars can be re-run under different coverage draws).
    library : coverage library (default: load_library()).
    weighting : for a class spec, 'uniform' (each pattern in the class equally
        likely; the default, Rae 2026-10-15, to avoid tuning to one year's mix)
        or 'frequency' (weighted by n_stars). Ignored otherwise.

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
    elif spec in COVERAGE_CLASSES or spec in CLASS_GROUPS:
        pool = lib[lib["class"].isin(CLASS_GROUPS.get(spec, (spec,)))]
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
