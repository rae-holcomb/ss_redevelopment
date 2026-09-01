"""Phase-coverage curves for TESS window functions.

Recreates Fig. 4 of Rodel et al. (2024, MNRAS 529, 3739; doi:
10.1093/mnras/stae474) -- the fraction of orbital phase covered by TESS
observations as a function of a simulated planet's orbital period -- but
using current TESS pointings rather than the paper's Cycle-1-only,
13-consecutive-Sector picture.

Two things change relative to the paper:

1. The real window function is taken from a star observed in *45* sectors
   spread over seven years, so the curves can be extended well past N=13.
2. Because modern targets are almost never observed in one unbroken run,
   the extension figure contrasts N *consecutive* sectors against the N
   sectors a star was *actually* observed in, which are clumped into
   per-cycle blocks separated by year-long gaps.

Coverage is computed exactly, not by binning: each contiguous run of
observed cadences is mapped to an arc on the [0, 1) phase circle and the
arcs are unioned. This is what produces the fine oscillatory structure in
the real-window-function curves -- resonances between the period and the
repeating orbit/downlink gap pattern -- rather than sampling noise.
"""

import argparse
import os

import numpy as np

SECTOR_LENGTH = 27.4  # days; nominal TESS Sector, two ~13.7 d spacecraft orbits


def cadences_to_intervals(t, dt, gap_tol=1.5):
    """Collapse a cadence time array into contiguous observing intervals.

    Consecutive cadences separated by more than `gap_tol * dt` are treated
    as bracketing a gap. Each returned interval is padded by half a cadence
    at both ends so that it spans the actual exposure coverage rather than
    the cadence mid-times.

    Parameters
    ----------
    t : ndarray
        Sorted cadence mid-times, in days.
    dt : float
        Nominal cadence spacing, in days.
    gap_tol : float
        Multiple of `dt` above which a spacing counts as a gap.

    Returns
    -------
    ndarray, shape (n_intervals, 2)
        Interval start and stop times, in the same units as `t`.
    """
    t = np.asarray(t, dtype=float)
    breaks = np.flatnonzero(np.diff(t) > gap_tol * dt)
    starts = np.concatenate(([t[0]], t[breaks + 1])) - 0.5 * dt
    stops = np.concatenate((t[breaks], [t[-1]])) + 0.5 * dt
    return np.column_stack([starts, stops])


def phase_coverage(intervals, periods):
    """Fraction of orbital phase covered by a set of observing intervals.

    For each trial period the intervals are wrapped onto the phase circle
    and their union measured. An interval at least one period long covers
    all phases by itself.

    Parameters
    ----------
    intervals : ndarray, shape (n, 2)
        Contiguous observing intervals (start, stop) in days.
    periods : ndarray
        Trial orbital periods in days.

    Returns
    -------
    ndarray
        Covered phase fraction in [0, 1], one entry per trial period.
    """
    intervals = np.atleast_2d(np.asarray(intervals, dtype=float))
    starts = intervals[:, 0]
    durations = intervals[:, 1] - intervals[:, 0]
    longest = durations.max()
    n = len(starts)

    out = np.empty(len(periods), dtype=float)
    for i, period in enumerate(periods):
        if longest >= period:
            out[i] = 1.0
            continue
        a = np.mod(starts / period, 1.0)
        b = a + durations / period
        # Split arcs that run past phase 1 into two non-wrapping arcs. The
        # non-wrapping half is kept in place; the wrapped remainder is
        # appended as an arc starting at phase 0. Arcs that do not wrap get
        # a zero-length filler so the arrays stay a fixed size.
        wraps = b > 1.0
        lo = np.concatenate([a, np.zeros(n)])
        hi = np.concatenate([np.where(wraps, 1.0, b), np.where(wraps, b - 1.0, 0.0)])

        order = np.argsort(lo, kind="stable")
        lo, hi = lo[order], hi[order]
        # Sorted by arc start, the already-covered phase above lo[i] is
        # exactly [lo[i], max(hi[:i])], so each arc contributes only the
        # part of itself beyond the running maximum of previous ends.
        running = np.empty_like(hi)
        running[0] = -np.inf
        np.maximum.accumulate(hi[:-1], out=running[1:])
        out[i] = min(np.maximum(hi - np.maximum(lo, running), 0.0).sum(), 1.0)
    return out


def idealised_intervals(n_sectors, sector_length=SECTOR_LENGTH):
    """Window function for `n_sectors` perfectly contiguous, gap-free Sectors.

    This is the paper's scenario (1): one unbroken block of observations
    `n_sectors * sector_length` days long, which covers all phases for any
    period shorter than that block.

    Parameters
    ----------
    n_sectors : int
        Number of Sectors in the block.
    sector_length : float
        Length of one Sector in days.

    Returns
    -------
    ndarray, shape (1, 2)
        The single contiguous interval.
    """
    return np.array([[0.0, n_sectors * sector_length]])


def load_window(path):
    """Load the cached per-sector cadence times written by `fetch_window.py`.

    Parameters
    ----------
    path : str
        Path to the .npz cache.

    Returns
    -------
    (dict, str)
        Mapping of sector number to its (n_intervals, 2) interval array,
        and the target identifier the cache was built from.
    """
    z = np.load(path, allow_pickle=True)
    sectors = [int(s) for s in z["sectors"]]
    windows = {}
    for s in sectors:
        windows[s] = cadences_to_intervals(z[f"t_{s}"], float(z[f"dt_{s}"]))
    return windows, str(z["tic"])


def stacked_intervals(windows, sectors):
    """Concatenate the observing intervals of several Sectors.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array, as returned by `load_window`.
    sectors : sequence of int
        Sectors to combine.

    Returns
    -------
    ndarray, shape (n, 2)
        All intervals, sorted by start time.
    """
    stack = np.vstack([windows[s] for s in sectors])
    return stack[np.argsort(stack[:, 0])]


def consecutive_runs(sectors):
    """Split a sorted sector list into runs of consecutive sector numbers.

    Parameters
    ----------
    sectors : sequence of int
        Sector numbers, ascending.

    Returns
    -------
    list of list of int
        One list per unbroken run.
    """
    runs, run = [], [sectors[0]]
    for s in sectors[1:]:
        if s == run[-1] + 1:
            run.append(s)
        else:
            runs.append(run)
            run = [s]
    runs.append(run)
    return runs


def describe(windows):
    """Print a short summary of the loaded window function.

    Parameters
    ----------
    windows : dict
        Sector number -> interval array.

    Returns
    -------
    None
    """
    sectors = sorted(windows)
    runs = consecutive_runs(sectors)
    print(f"{len(sectors)} sectors: {sectors}")
    print("consecutive runs: " + ", ".join(
        f"{r[0]}-{r[-1]} (n={len(r)})" if len(r) > 1 else f"{r[0]} (n=1)" for r in runs))
    for s in sectors:
        iv = windows[s]
        span = iv[-1, 1] - iv[0, 0]
        on = (iv[:, 1] - iv[:, 0]).sum()
        print(f"  S{s:3d}: {len(iv):3d} intervals  span={span:6.2f}d  "
              f"on-sky={on:6.2f}d  duty={on / span:.3f}")
