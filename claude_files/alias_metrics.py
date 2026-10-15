"""
alias_metrics.py

Shared, dependency-light ACF / phase-fold alias metrics (T9 features,
moved here from ml_features.py in T18, 2026-10-14) so that BOTH
ml_features.py (ML feature table) and comb_fit.py (arbitration gates) can
use one implementation without comb_fit importing ml_features (which
imports comb_fit: a circular import).

The code is moved verbatim; ml_features keeps its old private names as
aliases, and tests/test_alias_metrics.py pins the feature values
bit-for-bit to the pre-move output. See FEATURE_DOCUMENTATION.txt
Section G for the definitions and rationale.
"""

from __future__ import annotations

import numpy as np


def acf_window_max(acf_lags: np.ndarray, acf: np.ndarray, center: float, half_width: float) -> float:
    """Maximum of the raw ACF within [center - half_width, center + half_width],
    or NaN if that window falls (even partly) outside the ACF's lag range or
    contains no samples. A window max rather than the value at `center`
    tolerates the small offset between a candidate's comb and the actual
    ACF peak (and the secondary peak of a double-dip star, which is rarely
    at exactly half the lag of the primary).
    """
    lo, hi = center - half_width, center + half_width
    if lo < acf_lags[0] or hi > acf_lags[-1]:
        return float("nan")
    i0, i1 = np.searchsorted(acf_lags, [lo, hi], side="left")
    if i1 <= i0:
        return float("nan")
    return float(np.max(acf[i0:i1 + 1]))


def acf_harmonic_contrasts(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P: float,
    n_max: int = 6,
    win_frac: float = 0.1,
) -> dict:
    """ACF-height contrasts between a candidate's own multiples and its
    half-multiples (McQuillan et al. 2013/2014's alternating-peak test,
    made continuous and evaluated at every candidate).

    With h(x) = max ACF within +/- win_frac*P of lag x:

    - acf_half_lag_contrast = mean_k [h(k P) - h((k - 1/2) P)] over the
      k = 1..n_max whose windows fit in the ACF. If P is the true period
      and the star is single-dipped, the half-lags sit in troughs and the
      contrast is large; for a double-dipped star they hold the lower
      secondary peaks (moderate contrast); if P is a 2P_true alias the
      half-lags ARE true-period peaks and the contrast is ~0 or negative.
    - acf_odd_even_contrast = mean_j [h((2j - 1) P) - h(2j P)] over the
      j = 1..n_max/2 whose windows fit. For the true period this is
      slightly positive (peak decay with lag); if P is a P_true/2 alias,
      odd multiples are the weaker secondary peaks and even multiples the
      true peaks, so it goes clearly negative.

    Also returns acf_height_at_P = h(P) itself: the candidate-level analogue
    of G_ACF (Breton et al. 2021; the ACF height at the period, McQuillan
    et al. 2013), a near-free by-product.

    All in raw ACF units (the ACF is normalized to 1 at lag 0). NaN when
    no complete term fits inside the ACF lag range.
    """
    if not np.isfinite(P) or P <= 0:
        return dict(acf_height_at_P=np.nan, acf_half_lag_contrast=np.nan, acf_odd_even_contrast=np.nan)
    w = win_frac * P
    h = {m: acf_window_max(acf_lags, acf, m * P, w) for m in np.arange(0.5, n_max + 0.5, 0.5)}
    half_terms = [h[k] - h[k - 0.5] for k in range(1, n_max + 1)
                  if np.isfinite(h[k]) and np.isfinite(h[k - 0.5])]
    oe_terms = [h[2 * j - 1] - h[2 * j] for j in range(1, n_max // 2 + 1)
                if np.isfinite(h[2 * j - 1]) and np.isfinite(h[2 * j])]
    return dict(
        acf_height_at_P=h[1.0],
        acf_half_lag_contrast=float(np.mean(half_terms)) if half_terms else np.nan,
        acf_odd_even_contrast=float(np.mean(oe_terms)) if oe_terms else np.nan,
    )


def fold_half_cycle_corr(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    n_bins: int = 20,
    cycles_per_segment: float = 4.0,
    min_points_per_bin: float = 2.0,
) -> float:
    """Half-cycle self-correlation of the phase-folded, binned light curve
    at period P -- the "odd/even" test used to vet eclipsing binaries,
    applied to rotation.

    The light curve is cut into consecutive segments of
    cycles_per_segment * P (so spot evolution over the full baseline
    doesn't wash out the fold); within each segment the flux is folded at
    P and averaged into n_bins phase bins, and the mean-subtracted profile
    x is correlated with itself shifted by half a cycle:
    sum(x * roll(x, n_bins/2)) / sum(x**2). This circular form does not
    depend on the (arbitrary) phase origin, and lies in [-1, 1]. Segments
    are combined by a point-count-weighted mean.

    Interpretation: if P is twice the true period, the two halves of the
    fold are the same rotation twice and the correlation is ~1. For the
    true period of a single-dipped star the halves are dip vs. no dip
    (correlation <= 0); a double-dipped star gives an intermediate value.
    Evaluating this at 2P (see _alias_discrimination_features) gives the
    complementary half-alias test.

    Returns NaN if no segment has at least min_points_per_bin valid points
    per bin and finite variance in both halves.
    """
    if not np.isfinite(P) or P <= 0 or n_bins < 4 or len(time) < 2:
        return float("nan")
    # never more bins than cadences per cycle (else bins are structurally
    # empty when P is near-commensurate with the cadence, e.g. 0.25 d at 30 min)
    pts_per_cycle = P / float(np.median(np.diff(time)))
    n_bins = int(min(n_bins, np.floor(pts_per_cycle)))
    n_bins -= n_bins % 2
    if n_bins < 4:
        return float("nan")
    half = n_bins // 2
    valid = np.isfinite(flux)
    t, f = time[valid], flux[valid]
    if len(t) < n_bins * min_points_per_bin:
        return float("nan")
    # Fully vectorized over segments (one bincount over segment x phase-bin
    # cells): a per-segment Python loop cost ~19 ms/candidate at P = 0.25 d
    # on a 13-sector light curve (~360 segments), vs ~1 ms this way.
    seg_len = cycles_per_segment * P
    rel = t - t[0]
    seg_id = np.floor(rel / seg_len).astype(np.int64)
    n_seg = int(seg_id[-1]) + 1
    # phase measured from each segment's own nominal start (seg_id * seg_len
    # is a whole number of cycles, so this equals the global phase)
    b = np.minimum(((rel / P) % 1.0 * n_bins).astype(np.int64), n_bins - 1)
    cell = seg_id * n_bins + b
    cnt = np.bincount(cell, minlength=n_seg * n_bins).reshape(n_seg, n_bins)
    sums = np.bincount(cell, weights=f, minlength=n_seg * n_bins).reshape(n_seg, n_bins)
    n_pts = cnt.sum(axis=1)
    # t is sorted, so each segment is a contiguous run of seg_id
    seg_range = np.arange(n_seg)
    i_first = np.searchsorted(seg_id, seg_range, side="left")
    i_last = np.searchsorted(seg_id, seg_range, side="right") - 1
    nonempty = i_last >= i_first
    t_first = np.where(nonempty, t[np.minimum(i_first, len(t) - 1)], np.inf)
    t_last = np.where(nonempty, t[np.maximum(i_last, 0)], -np.inf)
    ok = (
        (n_pts >= n_bins * min_points_per_bin)
        & (t_last - t_first >= 2.0 * P)  # need >= 2 cycles to compare halves meaningfully
        & np.all(cnt > 0, axis=1)
    )
    if not ok.any():
        return float("nan")
    prof = sums[ok] / cnt[ok]
    a = prof - prof.mean(axis=1, keepdims=True)
    c = np.roll(a, half, axis=1)  # same profile, half a cycle later
    den = (a ** 2).sum(axis=1)
    good = den > 0
    if not good.any():
        return float("nan")
    corrs = (a * c).sum(axis=1)[good] / den[good]
    return float(np.average(corrs, weights=n_pts[ok][good]))


def half_lag_contrast_terms(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P: float,
    n_max: int = 6,
    win_frac: float = 0.1,
) -> np.ndarray:
    """The individual terms h(k P) - h((k - 1/2) P), k = 1..n_max, whose
    windows fit inside the ACF (h = acf_window_max with half-width
    win_frac * P). Their mean is exactly acf_harmonic_contrasts'
    acf_half_lag_contrast (same windows, same order).

    Used by comb_fit's half-lag consistency gate (T18): for a genuine
    period every half-lag sits in an ACF trough (or on a lower secondary
    peak), so EVERY term is positive; for a noise comb (e.g. a ~0.2 d
    comb riding red-noise ACF wiggles) the terms scatter around zero and
    some are negative; for a 2P alias the half-lags are true peaks.

    Returns
    -------
    np.ndarray of float (possibly empty if P is invalid or no window fits).
    """
    if not np.isfinite(P) or P <= 0:
        return np.array([], dtype=float)
    w = win_frac * P
    h = {m: acf_window_max(acf_lags, acf, m * P, w) for m in np.arange(0.5, n_max + 0.5, 0.5)}
    return np.array([h[k] - h[k - 0.5] for k in range(1, n_max + 1)
                     if np.isfinite(h[k]) and np.isfinite(h[k - 0.5])], dtype=float)
