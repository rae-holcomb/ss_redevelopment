"""
preprocessing.py

Turns raw SMARTS-format FITS light curves into the (time, flux) arrays the
rest of this pipeline expects: evenly-sampled at a fixed cadence, with NaN
at missing/removed cadences.

Why this is needed: the raw SMARTS light curves ship with their per-orbit
data gaps already linearly interpolated over (so there are no NaNs to work
with directly), which makes them look artificially cleaner than real TESS
data. This module reverses that: it re-detects where the original gaps were
(from the telltale drop in local flux scatter left behind by the
interpolation), splits the light curve back into individual orbits, then
lets you recombine only the orbits/sectors you want -- including
deliberately keeping only a subset of sectors to degrade a long, mostly-
complete SMARTS baseline down to a realistic, gappy TESS-like observing
pattern -- before regridding onto a fixed cadence.

Pipeline (see load_smarts_fits for the one-call version):
    1. Read time/flux (and the injected PERIOD) from a SMARTS FITS file.
    2. split_lightcurve: re-locate the original per-orbit gaps and break
       the timeseries back into one segment per orbit, flagging orbits
       with anomalous scatter (contaminated/over-processed) along the way.
    3. recombine_segments: stitch segments back into one (time, flux)
       array, optionally dropping the identified gap points, flagged
       (bad) orbits, and/or restricting to hand-picked sectors.
    4. to_even_cadence: map onto a fixed-cadence grid, NaN at any cadence
       with no data -- the format acf_utils.compute_acf expects.

The orbit-splitting/recombination functions (find_candidate_gaps,
split_lightcurve, plot_split_diagnostics, segments_to_lightkurve,
recombine_segments) are unchanged from the project's existing
split_smarts_lc.py; see their docstrings for full detail on the gap
re-detection algorithm and flagging logic. to_even_cadence and bin_lc are
likewise unchanged from their original implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits


# ==========================================================================
# Cadence regridding / binning
# ==========================================================================

def to_even_cadence(time, flux, flux_err=None, timestep=1800):
    """
    Maps a light curve onto an evenly-spaced time grid at the given cadence.
    Each original point is placed at its nearest grid index (no averaging);
    timesteps with no matching data point are filled with NaN.

    Args:
        time (array): time stamps, in days.
        flux (array): flux values.
        flux_err (array): flux_err values, if available.
        timestep (float): cadence of the output grid, in seconds (default 1800 s).

    Returns:
        time_even (array): evenly-spaced time stamps, in days.
        flux_even (array): flux values on the even grid, with NaN for gaps.
        flux_err (array): flux_err values on the even grid, if provided.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    dt = timestep / 86400.0  # seconds -> days

    # sanity check: does the requested timestep match the data's actual cadence?
    diffs = np.round(np.diff(time) * 86400).astype(int)
    values, counts = np.unique(diffs, return_counts=True)
    mode_timestep = values[np.argmax(counts)]
    if not np.isclose(timestep, mode_timestep, rtol=0.01):
        warnings.warn(
            f"Requested timestep ({timestep} s) does not match the mode "
            f"cadence of the input time array ({mode_timestep} s). "
            "The wrong timestep may have been provided."
        )

    n_bins = int(round((time[-1] - time[0]) / dt)) + 1
    time_even = time[0] + np.arange(n_bins) * dt

    idx = np.round((time - time[0]) / dt).astype(int)
    valid = (idx >= 0) & (idx < n_bins)

    flux_even = np.full(n_bins, np.nan)
    flux_even[idx[valid]] = flux[valid]

    if flux_err is not None:
        flux_err_even = np.full(n_bins, np.nan)
        flux_err_even[idx[valid]] = flux_err[valid]
        return time_even, flux_even, flux_err_even

    return time_even, flux_even


def bin_lc(time, flux, flux_err=None, binsize=1800):
    """
    Bins a timeseries to the desired cadence. Works much faster than Lightkurve's built in binning function.

    Args:
        time (:obj:`array`): time stamps of the lightcurve.
        flux (:obj:`array`): flux values of the lightcurve.
        flux_err (:obj:`array`): flux error values of the lightcurve.
        binsize (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
    Two parameters, or three if a flux_err is also provided.

        - time_bin (:obj:`array`): binned time stamps of the lightcurve.
        - flux_bin (:obj:`array`): binned flux values of the lightcurve.
        - flux_err_bin (:obj:`array`): binned flux errors values of the lightcurve. Only returned if an array is passed to flux_err.
    """
    time_bin = np.arange(time[0], time[-1], binsize / 86400.0)  # seconds -> days
    flux_bin = stats.binned_statistic(time, flux, bins=time_bin)[0]

    # If flux_err is populated, assume the errors combine as the root-mean-square
    if flux_err is not None:
        # define a function to calculate the root mean square error of each bin
        rmse_func = (lambda x: np.sqrt(np.nansum(np.square(x))) / len(np.atleast_1d(x))
                     if np.any(np.isfinite(x))
                     else np.nan)
        flux_err_bin = stats.binned_statistic(time, flux_err, statistic=rmse_func, bins=time_bin)[0]
        return time_bin[:-1], flux_bin, flux_err_bin

    return time_bin[:-1], flux_bin


# ==========================================================================
# Orbit splitting / recombination (from split_smarts_lc.py, unchanged)
# ==========================================================================

def _rolling_std(flux, window):
    return pd.Series(flux).rolling(window, center=True, min_periods=1).std().to_numpy()


def _orbit_idxs_for_sector(sector_num, n_orbits):
    """
    0-based indices into the orbit-segment list belonging to TESS-like
    sector `sector_num` (1-indexed), i.e. orbits `2*sector_num - 1` and
    `2*sector_num` in 1-indexed terms. Only indices that actually exist
    (< n_orbits) are returned, so a sector entirely beyond the end of the
    timeseries yields an empty list.
    """
    idx0 = 2 * (sector_num - 1)
    return [i for i in (idx0, idx0 + 1) if 0 <= i < n_orbits]


def find_candidate_gaps(time, flux, roll_window=9, baseline_window_days=10.0,
                         thresh=0.2, min_width_pts=10):
    """
    Locate candidate interpolated-gap regions in a flux timeseries.

    A point is flagged as part of a gap if a rolling standard deviation of
    the flux drops below `thresh` times a slower-varying local baseline
    (a rolling median of the rolling std), and flagged runs shorter than
    `min_width_pts` points are discarded as noise rather than real gaps.

    Parameters
    ----------
    time : array_like
        1D array of time values (assumed sorted, roughly uniform cadence).
    flux : array_like
        1D array of flux values, same length as `time`.
    roll_window : int, optional
        Window size (in points) for the rolling std used to spot gaps.
        Default 9.
    baseline_window_days : float, optional
        Width, in days, of the window used to compute the local "typical
        scatter" baseline that the rolling std is compared against.
        Default 10.
    thresh : float, optional
        A point is flagged as part of a gap if its rolling std is below
        `thresh` times the local baseline. Default 0.2.
    min_width_pts : int, optional
        Minimum number of consecutive flagged points required for a dip
        to be treated as a real gap rather than noise. Default 10.

    Returns
    -------
    candidates : list of dict
        One dict per candidate gap, with keys 'start_idx', 'end_idx'
        (inclusive index range into `time`/`flux`), 'center_idx',
        'center_time', 'width_pts', and 'mean_ratio' (mean rolling-std /
        baseline ratio within the gap; lower means a more convincing gap).
    roll : np.ndarray
        The rolling std array used for detection.
    baseline : np.ndarray
        The local baseline the rolling std was compared against.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    cadence = np.median(np.diff(time))
    roll = _rolling_std(flux, roll_window)
    baseline_pts = max(3, int(round(baseline_window_days / cadence)))
    if baseline_pts % 2 == 0:
        baseline_pts += 1
    baseline = pd.Series(roll).rolling(baseline_pts, center=True, min_periods=baseline_pts // 4).median().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = roll / baseline
    flagged = ratio < thresh
    idx = np.where(flagged)[0]
    if len(idx) == 0:
        return [], roll, baseline
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    groups = [g for g in groups if len(g) >= min_width_pts]
    candidates = []
    for g in groups:
        candidates.append(dict(
            start_idx=int(g[0]),
            end_idx=int(g[-1]),
            center_idx=int(g[len(g) // 2]),
            center_time=time[g].mean(),
            width_pts=len(g),
            mean_ratio=float(np.nanmean(ratio[g])),
        ))
    return candidates, roll, baseline


def split_lightcurve(time, flux, n_segments=27, roll_window=9,
                      baseline_window_days=10.0, thresh=0.2, min_width_pts=10,
                      tol_frac=0.4, remove_gap_points=False, flag_n_mad=5.0):
    """
    Split a long, gap-interpolated light curve back into its individual
    orbits.

    Long timeseries such as SMARTS/TESS light curves are sometimes
    distributed with the original per-orbit data gaps linearly
    interpolated over. This function relocates those gaps and uses them
    to break the timeseries back into pieces. Each returned segment
    corresponds to a single orbit -- for TESS-style data, two consecutive
    segments make up one sector (see `segments_to_lightkurve`, which
    recombines pairs of segments back into per-sector light curves).

    Detection strategy
    -------------------
    Interpolated gaps are flat, so a rolling standard deviation of the
    flux drops sharply and briefly right where a gap was filled in.
    Because different orbits can have very different characteristic
    scatter, the rolling std is normalized by a much more slowly-varying
    *local* baseline (a rolling median of the rolling std over
    `baseline_window_days`) before thresholding, so a low-scatter orbit
    isn't mistaken for a gap and a high-scatter orbit's real gaps are not
    missed. Candidate gaps are required to persist for at least
    `min_width_pts` consecutive points, which rejects momentary noise
    dips that are not wide enough to be a real interpolated gap.

    Because orbits are not exactly equal in length, breaks are not
    placed at `duration / n_segments` exactly. Instead, each nominal
    (evenly-spaced) boundary is used only as a rough guess, and the
    function looks for the nearest detected gap candidate within
    `tol_frac` of a nominal segment length. If no candidate gap is found
    near an expected boundary (for example, because a flare or other
    high-amplitude event masks the interpolation signature there), the
    function falls back to splitting at the nominal boundary itself and
    flags that break as unmatched/low-confidence in the diagnostics.

    After splitting, each segment's flux scatter (computed excluding any
    identified gap points) is compared to the other segments, and
    segments with anomalously high *or low* scatter are flagged -- useful
    for spotting orbits contaminated by flares, systematics, etc. (high
    scatter), or ones affected by an over-aggressive detrending or
    clipping step upstream (low scatter). Separately, each segment's
    *median rolling std* (the same point-to-point statistic used for gap
    detection, computed over every point in the segment, including any
    gap points) is compared across segments and flagged if anomalously
    low. This catches segments that are mostly or entirely one long
    interpolated gap even when the gap-candidate search above missed
    them (e.g. a long interpolated stretch can drift across a wide flux
    range, which gives it an overall `segment_std` similar to a real
    orbit even though its point-to-point scatter is ~0 throughout).

    Parameters
    ----------
    time : array_like
        1D array of time values (assumed sorted, roughly uniform cadence).
    flux : array_like
        1D array of flux values, same length as `time`.
    n_segments : int, optional
        Number of segments the full baseline is expected to span. Each
        segment corresponds to a single *orbit*, not a full sector (a
        TESS-like sector is made up of two orbits) -- so this should be
        set to the expected number of orbits, not sectors. Default 27,
        matching a ~year-long baseline of two-orbit, ~13.5-day-long
        TESS-like sectors.
    roll_window : int, optional
        Window size (in points) for the rolling std used to spot gaps.
        Default 9.
    baseline_window_days : float, optional
        Width, in days, of the window used to compute the local
        "typical scatter" baseline that the rolling std is compared
        against. Should be a few times longer than a gap but much
        shorter than an orbit. Default 10.
    thresh : float, optional
        A point is flagged as part of a gap if its rolling std is below
        `thresh` times the local baseline. Default 0.2.
    min_width_pts : int, optional
        Minimum number of consecutive flagged points required for a
        dip to be treated as a real interpolated gap rather than noise.
        Default 10.
    tol_frac : float, optional
        Fraction of the nominal segment length within which a detected
        gap candidate is accepted as the break for a given expected
        boundary. Default 0.4.
    remove_gap_points : bool, optional
        If True, points that fall within an identified (matched) gap
        region are dropped from the returned segments. Points near an
        unmatched/fallback break are not removed, since no gap region
        was actually located there. Default False.
    flag_n_mad : float, optional
        Segments are flagged as anomalous if a statistic deviates from
        the median of that statistic across all segments by more than
        `flag_n_mad * 1.4826 * MAD` (1.4826 * MAD approximates one
        standard deviation for normally-distributed data). Applied
        two-sided to each segment's flux std (catching both anomalously
        high- and low-scatter segments) and one-sided (low only) to each
        segment's median rolling std (catching mostly-interpolated
        segments). Default 5.0.

    Returns
    -------
    segments : list of (time_segment, flux_segment)
        One tuple per orbit, in time order. If `remove_gap_points=True`,
        identified gap points are excluded.
    diagnostics : dict
        Keys:
        - 'roll' : the rolling std array used for detection
        - 'baseline' : the local baseline used to normalize the rolling std
        - 'candidates' : all candidate gaps found anywhere in the data
        - 'break_time' : time of each break actually used to split
        - 'matched' : bool array, True if a break was matched to a real
          detected gap candidate, False if it had to fall back to the
          nominal (uniform-spacing) guess
        - 'confidence' : mean_ratio of the matched candidate at each
          break (nan for unmatched/fallback breaks)
        - 'gap_mask' : boolean array, same length as `time`, True for
          points identified as falling inside an interpolated gap
        - 'segment_std' : flux std of each segment (gap points excluded)
        - 'segment_median_roll_std' : median rolling std of each segment
          (all points included, so a mostly-interpolated segment reads
          very low even if its overall flux std does not)
        - 'segment_flagged' : bool array, True for segments flagged by
          either the `segment_std` or `segment_median_roll_std` check
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    candidates, roll, baseline = find_candidate_gaps(
        time, flux, roll_window=roll_window,
        baseline_window_days=baseline_window_days,
        thresh=thresh, min_width_pts=min_width_pts)

    duration = time[-1] - time[0]
    seg_len = duration / n_segments
    tol = tol_frac * seg_len

    used = set()
    break_idx, break_time, matched, confidence = [], [], [], []
    cur_t = time[0]
    for _ in range(n_segments - 1):
        t_exp = cur_t + seg_len
        best = None
        best_dist = tol
        for ci, c in enumerate(candidates):
            if ci in used:
                continue
            dist = abs(c['center_time'] - t_exp)
            if dist <= best_dist:
                best_dist = dist
                best = ci
        if best is not None:
            used.add(best)
            c = candidates[best]
            break_idx.append(c['center_idx'])
            break_time.append(c['center_time'])
            matched.append(True)
            confidence.append(c['mean_ratio'])
            cur_t = c['center_time']
        else:
            idx_fallback = int(np.searchsorted(time, t_exp))
            idx_fallback = min(max(idx_fallback, 1), len(time) - 1)
            break_idx.append(idx_fallback)
            break_time.append(time[idx_fallback])
            matched.append(False)
            confidence.append(np.nan)
            cur_t = time[idx_fallback]

    order = np.argsort(break_idx)
    break_idx = np.array(break_idx)[order]
    break_time = np.array(break_time)[order]
    matched = np.array(matched)[order]
    confidence = np.array(confidence)[order]

    # boolean mask of every point that falls inside a *matched* gap region
    gap_mask = np.zeros(len(time), dtype=bool)
    for ci in used:
        c = candidates[ci]
        gap_mask[c['start_idx']:c['end_idx'] + 1] = True

    segments_idx = np.split(np.arange(len(time)), break_idx)
    segments = []
    for s in segments_idx:
        if remove_gap_points:
            s = s[~gap_mask[s]]
        segments.append((time[s], flux[s]))

    # per-segment scatter, computed with any gap points excluded regardless
    # of remove_gap_points, then flag anomalously high or low ones
    segment_std = np.array([
        np.nanstd(flux[s[~gap_mask[s]]]) if len(s) else np.nan
        for s in segments_idx
    ])
    med = np.nanmedian(segment_std)
    mad = np.nanmedian(np.abs(segment_std - med))
    sigma_equiv = 1.4826 * mad
    if sigma_equiv > 0:
        segment_flagged = np.abs(segment_std - med) > (flag_n_mad * sigma_equiv)
    else:
        segment_flagged = np.zeros_like(segment_std, dtype=bool)

    # A whole-sector interpolation gap (or a long stretch of one) can look
    # like normal noise in segment_std above: linearly-interpolated flux
    # can drift over a wide range across a long segment, giving it an
    # overall scatter similar to real sectors even though point-to-point
    # scatter is ~0 throughout. The rolling std (unlike segment_std) is
    # sensitive to exactly that local, point-to-point scatter regardless
    # of any slow drift, so a segment that is mostly/entirely interpolated
    # shows up as an anomalously low *median rolling std*, and is flagged
    # here even when it was missed by the gap-candidate search above (e.g.
    # because it never dipped below the local baseline used there).
    #
    # This comparison is done in log-space rather than linear space.
    # Rolling std is a strictly-positive quantity that varies
    # multiplicatively -- a gap region is orders of magnitude lower, not
    # just "a few units" lower -- so a linear-scale MAD test can fail
    # outright: with several bad segments in one dataset, the linear
    # median/MAD of segment_median_roll_std can itself get pulled enough
    # that the low-outlier threshold (median - flag_n_mad*sigma) goes
    # negative, at which point nothing can ever be flagged, no matter how
    # small. Working in log10-space makes the test scale-invariant and
    # avoids that failure mode.
    segment_median_roll_std = np.array([
        np.nanmedian(roll[s]) if len(s) else np.nan
        for s in segments_idx
    ])
    with np.errstate(divide='ignore'):
        log_roll = np.log10(segment_median_roll_std)
    med_log_roll = np.nanmedian(log_roll)
    mad_log_roll = np.nanmedian(np.abs(log_roll - med_log_roll))
    sigma_equiv_log_roll = 1.4826 * mad_log_roll
    if sigma_equiv_log_roll > 0:
        segment_flagged_low_roll = (log_roll <
                                     (med_log_roll - flag_n_mad * sigma_equiv_log_roll))
    else:
        segment_flagged_low_roll = np.zeros_like(segment_median_roll_std, dtype=bool)

    segment_flagged = segment_flagged | segment_flagged_low_roll

    diagnostics = dict(roll=roll, baseline=baseline, candidates=candidates,
                        break_time=break_time, matched=matched,
                        confidence=confidence, seg_len=seg_len,
                        gap_mask=gap_mask, segment_std=segment_std,
                        segment_median_roll_std=segment_median_roll_std,
                        segment_flagged=segment_flagged)
    return segments, diagnostics


def plot_split_diagnostics(time, flux, diagnostics, xlim=None, figsize=(14, 6)):
    """
    Plot the flux and the rolling-std gap-detection diagnostics used by
    `split_lightcurve`, so detected breaks can be visually checked.

    Two panels are drawn: the flux timeseries, and the rolling standard
    deviation (log scale) used to detect gaps. Vertical lines mark the
    breaks actually used to split the data -- green for breaks matched to
    a real detected gap, red (dashed) for breaks that had to fall back to
    a nominal, evenly-spaced guess because no convincing gap was found
    nearby (e.g. because a real astrophysical event masked the
    interpolation signature at that point). Points identified as falling
    inside a gap are highlighted in orange on the flux panel. Segments
    flagged as anomalously high- or low-scatter
    (`diagnostics['segment_flagged']`) are shaded light red across both
    panels.

    Parameters
    ----------
    time : array_like
        1D array of time values passed to `split_lightcurve`.
    flux : array_like
        1D array of flux values passed to `split_lightcurve`.
    diagnostics : dict
        The diagnostics dict returned by `split_lightcurve`.
    xlim : tuple of float, optional
        (min, max) time range to zoom into. If None, the full range is
        shown.
    figsize : tuple, optional
        Matplotlib figure size. Default (14, 6).

    Returns
    -------
    fig, axs : matplotlib Figure and array of Axes
    """
    import matplotlib.pyplot as plt

    time = np.asarray(time)
    flux = np.asarray(flux)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axs[0].plot(time, flux, lw=0.4, color='0.2')
    gap_mask = diagnostics.get('gap_mask')
    if gap_mask is not None and gap_mask.any():
        axs[0].plot(time[gap_mask], flux[gap_mask], '.', color='tab:orange',
                    ms=2, label='identified gap points')
        axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_ylabel('flux')
    axs[1].plot(time, diagnostics['roll'], lw=0.6, color='0.2')
    axs[1].plot(time, diagnostics['baseline'], lw=1.0, color='tab:blue',
                label='local baseline scatter')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('rolling std')
    axs[1].set_xlabel('time [days]')
    axs[1].legend(loc='upper right', fontsize=8)

    for bt, m in zip(diagnostics['break_time'], diagnostics['matched']):
        color = 'tab:green' if m else 'tab:red'
        style = '-' if m else '--'
        for ax in axs:
            ax.axvline(bt, color=color, linestyle=style, lw=1.0, alpha=0.8)

    segment_flagged = diagnostics.get('segment_flagged')
    if segment_flagged is not None and np.any(segment_flagged):
        seg_edges = np.concatenate(([time[0]], diagnostics['break_time'], [time[-1]]))
        for i, flag in enumerate(segment_flagged):
            if flag:
                for ax in axs:
                    ax.axvspan(seg_edges[i], seg_edges[i + 1], color='red',
                               alpha=0.08, zorder=0)

    if xlim is not None:
        axs[0].set_xlim(*xlim)
    fig.tight_layout()
    return fig, axs


def segments_to_lightkurve(segments, diagnostics=None, exclude_flagged=False,
                            sectors=None, flux_unit=None, time_format=None):
    """
    Combine the per-orbit segments returned by `split_lightcurve` into
    per-sector `lightkurve.LightCurve` objects, returned as a
    `lightkurve.LightCurveCollection`.

    Each input segment is a single orbit; every two consecutive segments
    (orbits) are combined into one `LightCurve` representing a full
    TESS-like sector. If the number of segments is odd, the final,
    unpaired orbit becomes its own single-orbit "sector".

    Passing `sectors` restricts the output to specific, hand-picked TESS
    sector numbers -- handy for taking a long SMARTS baseline and
    manually degrading it down to a realistic, gappy TESS-like observing
    pattern (e.g. only the sectors a real target would actually have
    been observed in).

    Requires the `lightkurve` package (`pip install lightkurve`), which is
    imported lazily so the rest of this module works without it installed.

    Parameters
    ----------
    segments : list of (time_segment, flux_segment)
        The per-orbit segments returned by `split_lightcurve`.
    diagnostics : dict, optional
        If the diagnostics dict from `split_lightcurve` is passed, each
        LightCurve's metadata gets a 'FLAGGED' entry, True if *any* of
        its orbits were flagged in `diagnostics['segment_flagged']`.
        Required if `exclude_flagged=True`, and used (if available) to
        drop flagged sectors when `sectors` is given (see below).
    exclude_flagged : bool, optional
        Only applies when `sectors` is None (see below). If True,
        flagged orbits are left out of their sector's data rather than
        just being tagged. This is handled per-orbit, not per-sector: if
        only one of the two orbits making up a sector is flagged, that
        sector's LightCurve is still created, but contains only the
        good orbit's data (and the flagged orbit's number is recorded
        in that LightCurve's 'ORBITS_EXCLUDED' metadata). A sector is
        only left out of the collection entirely if *all* of its orbits
        are flagged. Requires `diagnostics` to be passed. Default False.
    sectors : array_like of int, optional
        1-indexed TESS sector numbers to include (e.g. `[1, 2, 6, 7,
        12]`), out of all sectors the data would nominally cover. If
        given, every other sector is left out of the collection
        entirely, and `exclude_flagged` is ignored in favor of the
        following, stricter rule: if *either* orbit belonging to a
        requested sector is flagged bad (per
        `diagnostics['segment_flagged']`, when `diagnostics` is
        passed), that whole sector is dropped -- unlike the
        partial-orbit salvage `exclude_flagged` does, a requested
        sector is either used whole or not at all. A requested sector
        number with no corresponding orbits in the data (i.e. beyond
        the end of the timeseries) is silently ignored. Default None
        (use every sector available).
    flux_unit : astropy.units.Unit, optional
        Unit to attach to the flux column. If None, flux is left
        dimensionless.
    time_format : str, optional
        Time format string passed to `lightkurve.LightCurve` (e.g.
        'btjd' for TESS-like time arrays). If None, lightkurve's default
        is used.

    Returns
    -------
    lightkurve.LightCurveCollection
        One LightCurve per sector (pair of orbits, or a single leftover
        orbit), in time order. Each LightCurve's metadata includes:
        - 'SECTOR' : 1-indexed sector number
        - 'ORBITS' : 1-indexed orbit numbers whose data are included
        - 'ORBITS_EXCLUDED' : 1-indexed orbit numbers left out due to
          flagging (only present if non-empty; only possible when
          `sectors` is None)
        - 'FLAGGED' : True if any orbit belonging to this sector was
          flagged (present only if `diagnostics` was passed)
    included_sectors : list of int
        1-indexed sector numbers actually present in the returned
        collection, in order. Comparing this against `sectors` (when
        given) is an easy way to see which requested sectors got
        dropped for being flagged bad.
    """
    import lightkurve as lk

    flagged = None
    if diagnostics is not None and 'segment_flagged' in diagnostics:
        flagged = diagnostics['segment_flagged']

    if exclude_flagged and sectors is None and flagged is None:
        raise ValueError(
            "exclude_flagged=True requires `diagnostics` (with "
            "'segment_flagged') to be passed.")

    n_orbits = len(segments)
    if sectors is None:
        sector_nums = range(1, -(-n_orbits // 2) + 1)  # 1..ceil(n_orbits/2)
        strict_sector_exclude = False
    else:
        sector_nums = sorted(set(int(s) for s in sectors))
        strict_sector_exclude = True

    lcs = []
    for s in sector_nums:
        orbit_idxs = _orbit_idxs_for_sector(s, n_orbits)
        if not orbit_idxs:
            continue  # sector beyond the end of the timeseries

        if strict_sector_exclude and flagged is not None and \
                any(flagged[oi] for oi in orbit_idxs):
            continue  # requested sector is bad -> drop it entirely

        included, excluded = [], []
        t_chunks, fl_chunks = [], []
        for oi in orbit_idxs:
            if (not strict_sector_exclude) and exclude_flagged and flagged[oi]:
                excluded.append(oi + 1)
                continue
            included.append(oi + 1)
            t_chunks.append(segments[oi][0])
            fl_chunks.append(segments[oi][1])

        if not included:
            # every orbit in this sector was flagged -- drop the sector
            continue

        kwargs = {}
        if flux_unit is not None:
            kwargs['flux_unit'] = flux_unit
        if time_format is not None:
            kwargs['time_format'] = time_format
        lc = lk.LightCurve(time=np.concatenate(t_chunks),
                            flux=np.concatenate(fl_chunks), **kwargs)
        lc.meta['SECTOR'] = s
        lc.meta['ORBITS'] = included
        if excluded:
            lc.meta['ORBITS_EXCLUDED'] = excluded
        if flagged is not None:
            lc.meta['FLAGGED'] = bool(any(flagged[oi] for oi in orbit_idxs))
        lcs.append(lc)

    included_sectors = [lc.meta['SECTOR'] for lc in lcs]
    return lk.LightCurveCollection(lcs), included_sectors


def recombine_segments(segments, diagnostics, drop_gap_points=False,
                        drop_flagged_segments=False, sectors=None):
    """
    Recombine the segments returned by `split_lightcurve` back into a
    single continuous (time, flux) pair, optionally dropping interpolated
    gap points, entire flagged (bad) segments, and/or whole sectors along
    the way.

    This assumes `segments` is the *unfiltered* output of
    `split_lightcurve` -- i.e. it was called with the default
    `remove_gap_points=False`, so every segment still contains any
    interpolated gap points it originally had. (If you already removed
    gap points at split time, just `np.concatenate` the segments
    yourself -- there's nothing left for `drop_gap_points` to do here,
    and `drop_flagged_segments` alone doesn't need this function.)

    Passing `sectors` restricts the recombined timeseries to specific,
    hand-picked TESS sector numbers -- handy for taking a long SMARTS
    baseline and manually degrading it down to a realistic, gappy
    TESS-like observing pattern (e.g. only the sectors a real target
    would actually have been observed in).

    Parameters
    ----------
    segments : list of (time_segment, flux_segment)
        The per-orbit segments returned by `split_lightcurve` (with
        `remove_gap_points=False`).
    diagnostics : dict
        The diagnostics dict returned alongside `segments` by
        `split_lightcurve`. Used to locate gap points (`'gap_mask'`) and,
        if `drop_flagged_segments=True` or `sectors` is given, bad
        segments (`'segment_flagged'`).
    drop_gap_points : bool, optional
        If True, points identified as falling inside an interpolated gap
        are excluded from the recombined arrays. Default False.
    drop_flagged_segments : bool, optional
        Only applies when `sectors` is None (see below). If True,
        segments (orbits) flagged in `diagnostics['segment_flagged']`
        (anomalously high/low scatter, or mostly-interpolated) are left
        out entirely. Default False.
    sectors : array_like of int, optional
        1-indexed TESS sector numbers to include (e.g. `[1, 2, 6, 7,
        12]`), out of all sectors the data would nominally cover. If
        given, orbits belonging to every other sector are left out
        entirely, `drop_flagged_segments` is ignored, and instead: if
        *either* orbit belonging to a requested sector is flagged bad
        (per `diagnostics['segment_flagged']`), that whole sector's
        orbits are left out -- unlike `drop_flagged_segments`, which
        would only drop the individually-flagged orbit, a requested
        sector here is either used whole or not at all. A requested
        sector number with no corresponding orbits in the data (i.e.
        beyond the end of the timeseries) is silently ignored. Default
        None (use every orbit available).

    Returns
    -------
    time : np.ndarray
        The recombined time array.
    flux : np.ndarray
        The recombined flux array, same length as `time`.
    included_sectors : list of int
        1-indexed sector numbers that contributed at least one orbit to
        the output, in order. Comparing this against `sectors` (when
        given) is an easy way to see which requested sectors got
        dropped for being flagged bad.
    """
    gap_mask = diagnostics['gap_mask']
    segment_flagged = diagnostics.get('segment_flagged')
    if drop_flagged_segments and sectors is None and segment_flagged is None:
        raise ValueError(
            "drop_flagged_segments=True requires `diagnostics` to "
            "contain 'segment_flagged'.")

    n_orbits = len(segments)
    if sectors is not None:
        allowed = set()
        for s in sorted(set(int(x) for x in sectors)):
            orbit_idxs = _orbit_idxs_for_sector(s, n_orbits)
            if not orbit_idxs:
                continue  # sector beyond the end of the timeseries
            if segment_flagged is not None and \
                    any(segment_flagged[oi] for oi in orbit_idxs):
                continue  # requested sector is bad -> drop it entirely
            allowed.update(orbit_idxs)
    else:
        allowed = set(range(n_orbits))

    time_chunks, flux_chunks = [], []
    contributing_idxs = []
    offset = 0
    for i, (t, fl) in enumerate(segments):
        n = len(t)
        seg_gap_mask = gap_mask[offset:offset + n]
        offset += n
        if i not in allowed:
            continue
        if sectors is None and drop_flagged_segments and segment_flagged[i]:
            continue
        if drop_gap_points:
            keep = ~seg_gap_mask
            t, fl = t[keep], fl[keep]
        time_chunks.append(t)
        flux_chunks.append(fl)
        contributing_idxs.append(i)

    if offset != len(gap_mask):
        raise ValueError(
            "`segments` don't line up with `diagnostics['gap_mask']` -- "
            "did you pass segments created with remove_gap_points=True? "
            "recombine_segments expects the unfiltered segments "
            "(remove_gap_points=False, the default) so it can apply its "
            "own drop_gap_points/drop_flagged_segments toggles.")

    included_sectors = sorted(set(i // 2 + 1 for i in contributing_idxs))

    if time_chunks:
        time_out = np.concatenate(time_chunks)
        flux_out = np.concatenate(flux_chunks)
    else:
        time_out = np.array([])
        flux_out = np.array([])
    return time_out, flux_out, included_sectors


# ==========================================================================
# Top-level convenience: raw FITS -> pipeline-ready (time, flux)
# ==========================================================================

@dataclass
class PreprocessedLightCurve:
    """Everything produced by load_smarts_fits, in one place."""
    time: np.ndarray
    flux: np.ndarray
    true_period: Optional[float]
    sectors_used: list
    segments: list = field(repr=False)
    diagnostics: dict = field(repr=False)
    source_path: Optional[str] = None
    n_orbits_total: int = 0
    n_orbits_flagged: int = 0


def load_smarts_fits(
    fits_path: Union[str, Path],
    n_segments: int = 27,
    sectors: Optional[list] = None,
    drop_gap_points: bool = True,
    drop_flagged_segments: bool = True,
    timestep: float = 1800,
    lightcurve_hdu: str = "LIGHTCURVE",
    period_header_key: str = "PERIOD",
    split_kwargs: Optional[dict] = None,
) -> PreprocessedLightCurve:
    """
    One-call pipeline: raw SMARTS FITS file -> pipeline-ready (time, flux).

    Steps (see the module docstring for the full rationale):
        1. Read time/flux from the FITS file's LIGHTCURVE HDU (flux is
           shifted from SMARTS' 1.0-centered convention to 0-centered, to
           match what the rest of this codebase -- acf_utils, comb_fit,
           etc. -- expects) and the injected rotation period from the
           primary header.
        2. split_lightcurve: re-detect the original per-orbit gaps and
           break the light curve back into individual orbits.
        3. recombine_segments: stitch the orbits back together, dropping
           identified gap points and/or flagged (bad) orbits, and
           optionally restricting to a hand-picked subset of sectors (to
           deliberately degrade a long SMARTS baseline down to a
           realistic, gappy TESS-like observing pattern).
        4. to_even_cadence: regrid onto a fixed cadence, NaN at any empty
           slot -- ready for acf_utils.compute_acf.

    Parameters
    ----------
    fits_path : path to a SMARTS-format FITS file (PRIMARY header with a
        `period_header_key` entry; a `lightcurve_hdu` binary table with
        'time' and 'flux' columns).
    n_segments : expected number of orbits spanning the full baseline,
        forwarded to split_lightcurve (see its docstring -- this is
        orbits, not sectors; two orbits make one TESS-like sector).
    sectors : optional list of 1-indexed sector numbers to keep, forwarded
        to recombine_segments. Use this to deliberately downgrade a long,
        mostly-complete SMARTS light curve to a sparser, more realistic
        TESS-like observing pattern (e.g. `sectors=[1, 2, 6, 7, 12]`).
        Leave as None to keep every available (non-flagged) sector.
    drop_gap_points, drop_flagged_segments : forwarded to
        recombine_segments -- see its docstring. Both default to True:
        the whole point of this preprocessing is to NOT hand the rest of
        the pipeline artificially-interpolated gap points or contaminated
        orbits.
    timestep : output cadence in seconds, forwarded to to_even_cadence.
        Default 1800 (30 min), matching SMARTS' native cadence.
    lightcurve_hdu : name of the FITS extension containing the time/flux
        table. Default "LIGHTCURVE".
    period_header_key : primary-header keyword holding the injected
        period. Default "PERIOD". If missing, `true_period` is None
        rather than raising -- useful for real (non-simulated) FITS files
        that don't have a known injected period.
    split_kwargs : optional dict of extra keyword arguments forwarded to
        split_lightcurve (e.g. `{"thresh": 0.3}` to loosen gap detection).

    Returns
    -------
    PreprocessedLightCurve
    """
    split_kwargs = split_kwargs or {}

    with fits.open(fits_path) as hdul:
        lc = hdul[lightcurve_hdu].data
        time = np.asarray(lc["time"], dtype=float)
        flux = np.asarray(lc["flux"], dtype=float) - 1.0
        true_period = hdul[0].header.get(period_header_key, None)
        if true_period is not None:
            true_period = float(true_period)

    segments, diagnostics = split_lightcurve(time, flux, n_segments=n_segments, **split_kwargs)
    time_stitch, flux_stitch, sectors_used = recombine_segments(
        segments, diagnostics,
        drop_gap_points=drop_gap_points,
        drop_flagged_segments=drop_flagged_segments,
        sectors=sectors,
    )
    time_even, flux_even = to_even_cadence(time_stitch, flux_stitch, timestep=timestep)

    return PreprocessedLightCurve(
        time=time_even,
        flux=flux_even,
        true_period=true_period,
        sectors_used=sectors_used,
        segments=segments,
        diagnostics=diagnostics,
        source_path=str(fits_path),
        n_orbits_total=len(segments),
        n_orbits_flagged=int(np.sum(diagnostics["segment_flagged"])),
    )
