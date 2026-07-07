"""
comb_fit.py

Identify stellar rotation periods from the autocorrelation function (ACF) of
a light curve by jointly fitting a "comb" of evenly-spaced, downward-opening
parabolae to the repeated peaks of the ACF.

Workflow
--------
1. Get an initial guess (P0, t0) for the period and phase of the first ACF
   peak. Three interchangeable strategies are provided:
       - guess_pairwise_histogram   (peak-finding + pairwise spacing histogram)
       - guess_lombscargle          (Lomb-Scargle periodogram of the light curve)
       - guess_acf_fft              (FFT of the ACF itself)
   All three share the same signature style: they return (P0, t0, info_dict)
   so they can be swapped freely.
2. fit_rotation_period() takes that guess, builds fitting windows around each
   expected peak (t0 + n*P0), and jointly fits a comb of parabolae with tied,
   evenly-spaced centers, using a robust loss and iterative rejection of bad
   peaks.

Conventions
-----------
- `acf_lags` is assumed sorted ascending and (as is typical for an ACF of an
  evenly-sampled light curve) evenly spaced.
- Lag 0 (and a small buffer around it) is always excluded from peak-finding,
  since the trivial peak at lag 0 is not a rotation signal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.signal import find_peaks

try:
    import lmfit
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "comb_fit requires lmfit (`pip install lmfit`)."
    ) from e

try:
    from astropy.timeseries import LombScargle
except ImportError:  # pragma: no cover
    LombScargle = None


# --------------------------------------------------------------------------
# Small shared utilities
# --------------------------------------------------------------------------

def _acf_peak_candidates(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_lag: float,
    max_lag: Optional[float] = None,
    prominence: float = 0.0,
    height: Optional[float] = None,
    distance_in_points: Optional[int] = None,
):
    """Run scipy.find_peaks on the ACF, excluding lag < min_lag (and,
    optionally, lag > max_lag). Returns (peak_lags, peak_heights, peak_idx).
    """
    if max_lag is None:
        max_lag = acf_lags[-1]

    mask = (acf_lags >= min_lag) & (acf_lags <= max_lag)
    sub_lags = acf_lags[mask]
    sub_acf = acf[mask]

    idx, props = find_peaks(
        sub_acf, prominence=prominence, height=height, distance=distance_in_points
    )
    peak_lags = sub_lags[idx]
    peak_heights = sub_acf[idx]

    # sort by lag (find_peaks already returns ascending, but be explicit)
    order = np.argsort(peak_lags)
    return peak_lags[order], peak_heights[order], idx[order]


def _teeth_count(P: float, t0: float, lag_max: float) -> int:
    """Number of comb teeth (n=0,1,2,...) that land within [t0, lag_max]
    for period P. Used to guard candidate-ranking heuristics (comb_score,
    peak coverage) against a failure mode where a large candidate period
    only has 1-2 teeth within the available lag range: with so few teeth,
    "mean height" and "coverage fraction" are both trivially easy to
    maximize (a single tooth sitting on any reasonably tall ACF value
    looks like a perfect score), even though such a candidate carries far
    less evidence than one with many teeth that all land well.
    """
    if P <= 0:
        return 0
    return int(np.floor((lag_max - t0) / P)) + 1


def comb_score(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P: float,
    t0: float,
    n_max: Optional[int] = None,
    lag_max: Optional[float] = None,
) -> float:
    """Score how well a comb of teeth at t0 + n*P (n=0,1,2,...) lines up with
    tall points of the ACF. Larger is better. Used both to pick t0 given a
    candidate P, and to compare/rank candidate periods against each other.

    ACF values are linearly interpolated at the (non-gridded, in general)
    comb positions.
    """
    if lag_max is None:
        lag_max = acf_lags[-1]
    if n_max is None:
        n_max = int(np.floor((lag_max - t0) / P)) if P > 0 else 0
    n_max = max(n_max, 0)

    n = np.arange(0, n_max + 1)
    comb_lags = t0 + n * P
    in_range = comb_lags <= lag_max
    comb_lags = comb_lags[in_range]
    if len(comb_lags) == 0:
        return -np.inf

    vals = np.interp(comb_lags, acf_lags, acf)
    # Normalize by number of teeth so scores are comparable across different
    # trial periods (which naturally have different numbers of teeth within
    # the lag range).
    return float(np.sum(vals) / len(comb_lags))


def _peak_coverage_fraction(
    peak_lags: np.ndarray,
    P: float,
    t0: float,
    tol: Optional[float] = None,
) -> float:
    """Fraction of the *found* ACF peaks (peak_lags) that land within `tol`
    of some comb tooth t0 + n*P. This is the key diagnostic for resolving
    harmonic ambiguity in the pairwise-spacing histogram: a candidate period
    that is actually an integer multiple of the true period (2P, 3P, ...)
    will only ever land on every 2nd, 3rd, ... found peak, so its coverage
    fraction is well below 1, whereas the true fundamental period explains
    essentially all of them (since the found peaks are what generated the
    pairwise differences in the first place).
    """
    if tol is None:
        tol = 0.1 * P
    if len(peak_lags) == 0:
        return 0.0
    n_est = np.round((peak_lags - t0) / P)
    nearest_tooth = t0 + n_est * P
    matched = np.abs(peak_lags - nearest_tooth) <= tol
    return float(np.mean(matched))


def _grid_search_t0(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    min_lag: float,
    n_phase: int = 200,
    n_teeth_for_score: int = 5,
) -> float:
    """Given a candidate period P0, grid-search the phase t0 in
    [min_lag, min_lag + P0) that maximizes comb_score. This anchors t0 to the
    *first* expected peak position, i.e. the returned t0 is itself the lag of
    the (n=0) tooth, not an arbitrary phase offset.
    """
    trial_t0 = np.linspace(min_lag, min_lag + P0, n_phase, endpoint=False)
    scores = [
        comb_score(acf_lags, acf, P0, t0, n_max=n_teeth_for_score)
        for t0 in trial_t0
    ]
    return float(trial_t0[int(np.argmax(scores))])


@dataclass
class InitialGuess:
    """Container returned by every guess_* function, so they're
    interchangeable regardless of internal method."""
    P0: float
    t0: float
    method: str
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Initial guess: find_peaks + pairwise spacing histogram
# --------------------------------------------------------------------------

def guess_pairwise_histogram(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_lag: Optional[float] = None,
    max_lag: Optional[float] = None,
    prominence: float = 0.0,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_hist_bins: Optional[int] = None,
    n_phase_grid: int = 200,
    min_teeth: int = 4,
) -> InitialGuess:
    """Find candidate ACF peaks, histogram all pairwise positive spacings
    between them, and take the smallest strongly-populated spacing as P0.

    This works even if some harmonics are missing (e.g. peaks 1, 2, 4 found
    but not 3), because the (1,2), (2,4) and other pairwise differences all
    still land near the fundamental spacing and reinforce that histogram bin,
    while peaks that don't belong to the comb contribute only weak,
    scattered differences.

    Parameters
    ----------
    time, flux : included for interface consistency with the other guess_*
        functions; not used by this method.
    min_lag : lower cutoff on lag to search for peaks (default: 3x the
        median lag spacing, to exclude the trivial lag-0 peak and its
        immediate shoulder).
    max_lag : upper cutoff on lag to search for peaks (default: full range).
    prominence : passed to scipy.find_peaks; raise this to reject noise
        peaks in noisy ACFs.
    min_period, max_period : if given, restrict candidate spacings to this
        range before histogramming (useful if you have rough prior bounds,
        e.g. from the cadence and baseline of the light curve).
    n_hist_bins : number of bins for the spacing histogram. Default is
        chosen from the lag resolution.
    n_phase_grid : number of phase points to try when refining t0 for the
        chosen P0.
    min_teeth : a candidate period must have at least this many comb teeth
        within the ACF's lag range to be eligible to win the internal
        ranking (see _teeth_count). Prevents a large candidate period with
        only 1-2 teeth from winning purely because a couple of ACF values
        happened to be tall -- such a candidate carries much weaker
        evidence than one supported by many evenly-spaced teeth. If no
        candidate meets this bar, the requirement is relaxed automatically
        (a real signal with a genuinely long period may only have a
        handful of cycles in the available baseline).

    Returns
    -------
    InitialGuess(P0, t0, method="pairwise_histogram", info={...})
        info contains the candidate peak positions, the pairwise spacing
        histogram, and the full ranked list of candidate periods, in case
        you want to inspect/try alternates.
    """
    dt = np.median(np.diff(acf_lags))
    if min_lag is None:
        min_lag = 3 * dt
    if max_lag is None:
        max_lag = acf_lags[-1]

    peak_lags, peak_heights, _ = _acf_peak_candidates(
        acf_lags, acf, min_lag=min_lag, max_lag=max_lag, prominence=prominence
    )

    if len(peak_lags) < 2:
        raise RuntimeError(
            "guess_pairwise_histogram: fewer than 2 ACF peaks found; "
            "try lowering `prominence` or widening the lag range."
        )

    # all pairwise positive differences
    diffs = []
    for i in range(len(peak_lags)):
        for j in range(i + 1, len(peak_lags)):
            diffs.append(peak_lags[j] - peak_lags[i])
    diffs = np.array(diffs)

    if min_period is not None or max_period is not None:
        lo = min_period if min_period is not None else diffs.min()
        hi = max_period if max_period is not None else diffs.max()
        diffs = diffs[(diffs >= lo) & (diffs <= hi)]

    if len(diffs) == 0:
        raise RuntimeError(
            "guess_pairwise_histogram: no pairwise spacings survive the "
            "min_period/max_period cut."
        )

    if n_hist_bins is None:
        # bin width a few times the lag resolution, resolution chosen fine
        # enough to separate P from 2P etc.
        bin_width = max(4 * dt, (diffs.max() - diffs.min()) / 200)
        n_hist_bins = max(int((diffs.max() - diffs.min()) / bin_width), 10)
    else:
        bin_width = (diffs.max() - diffs.min()) / n_hist_bins

    # IMPORTANT: histogram range starts at 0 (or a small floor), not at
    # diffs.min(). If the range started at diffs.min(), the fundamental
    # spacing (which is very often close to the smallest pairwise
    # difference) would sit in the very first bin, and scipy.find_peaks can
    # never flag an edge bin as a local maximum (it requires neighbors on
    # both sides) -- silently discarding the correct answer.
    hist_lo = max(0.0, diffs.min() - bin_width)
    hist_hi = diffs.max() + bin_width
    n_hist_bins = max(int((hist_hi - hist_lo) / bin_width), n_hist_bins)
    hist, bin_edges = np.histogram(diffs, bins=n_hist_bins, range=(hist_lo, hist_hi))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Local maxima of the histogram = candidate fundamental spacings (and
    # their harmonics). Rank candidates by (count, then preference for
    # *smaller* spacing, since harmonics 2P, 3P... will also show up as
    # local maxima but the fundamental is what we want).
    hist_peak_idx, hist_peak_props = find_peaks(hist, height=1)
    if len(hist_peak_idx) == 0:
        # fall back to the single global max bin
        hist_peak_idx = np.array([int(np.argmax(hist))])

    candidate_periods = bin_centers[hist_peak_idx]
    candidate_counts = hist[hist_peak_idx]

    # Score each candidate period by how well a comb at that spacing lines
    # up with the actual ACF (not just the discrete peak list), using the
    # best phase for each. This disambiguates the fundamental from its
    # harmonics better than raw histogram counts alone, since a comb at the
    # true P will find real ACF peaks at every tooth, while a comb at 2P or
    # 3P skips real peaks in between (or, conversely, one at P/2 will often
    # land teeth on troughs half the time).
    ranked = []
    for P_cand in candidate_periods:
        t0_cand = _grid_search_t0(
            acf_lags, acf, P_cand, min_lag=min_lag, n_phase=n_phase_grid
        )
        coverage = _peak_coverage_fraction(peak_lags, P_cand, t0_cand)
        score = comb_score(acf_lags, acf, P_cand, t0_cand)
        n_teeth = _teeth_count(P_cand, t0_cand, acf_lags[-1])
        # Primary sort key: whether the candidate meets the min_teeth bar
        # (see min_teeth docstring -- prevents few-tooth candidates from
        # winning trivially), then coverage (fixes harmonic ambiguity: P,
        # 2P, 3P... all look like local maxima in the spacing histogram,
        # but only the true fundamental explains nearly all found peaks),
        # then comb_score as a final tiebreaker.
        ranked.append((n_teeth >= min_teeth, coverage, score, P_cand, t0_cand, n_teeth))
    ranked.sort(key=lambda x: (x[0], round(x[1], 2), x[2]), reverse=True)
    # strip the min_teeth flag back out before exposing ranked_candidates,
    # keeping the (coverage, score, P, t0) shape documented above
    ranked = [(cov, score, P, t0) for _, cov, score, P, t0, _ in ranked]

    best_coverage, best_score, best_P, best_t0 = ranked[0]

    info = dict(
        peak_lags=peak_lags,
        peak_heights=peak_heights,
        pairwise_diffs=diffs,
        histogram=(bin_centers, hist),
        candidate_periods=candidate_periods,
        candidate_counts=candidate_counts,
        ranked_candidates=ranked,  # [(coverage, score, P, t0), ...] sorted best-first
    )
    return InitialGuess(P0=best_P, t0=best_t0, method="pairwise_histogram", info=info)


# --------------------------------------------------------------------------
# 2. Initial guess: Lomb-Scargle periodogram of the light curve
# --------------------------------------------------------------------------

def guess_lombscargle(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_top_peaks: int = 5,
    samples_per_peak: int = 10,
    min_lag: Optional[float] = None,
    min_teeth: int = 4,
) -> InitialGuess:
    """Compute the Lomb-Scargle periodogram of (time, flux), take the top
    candidate peak periods, and for each ask the ACF to pick the best phase
    (via the same comb_score used elsewhere), then keep whichever candidate
    period scores best against the ACF.

    Routing the LS candidates through the ACF comb score (rather than just
    trusting the single highest LS peak) matters because periodograms are
    prone to picking harmonics/subharmonics or aliases; letting the ACF
    arbitrate among the top few LS peaks is more robust than either
    technique alone.

    Parameters
    ----------
    min_period, max_period : period search range for the periodogram.
        Defaults to [4 * median(dt), (time[-1]-time[0])/2].
    n_top_peaks : number of top periodogram peaks to test against the ACF.
    samples_per_peak : oversampling factor passed to astropy's autopower.
    min_lag : lower cutoff for phase search on the ACF (default: same
        default as guess_pairwise_histogram).
    min_teeth : see guess_pairwise_histogram -- same guard against a
        candidate winning on the strength of only 1-2 comb teeth.

    Returns
    -------
    InitialGuess(P0, t0, method="lombscargle", info={...})
    """
    if LombScargle is None:
        raise ImportError(
            "guess_lombscargle requires astropy (`pip install astropy`)."
        )

    # Real light curves (TESS/Kepler especially) are often shipped as NaN
    # at missing cadences on an otherwise even grid. LombScargle doesn't
    # handle that gracefully (it silently returns an all-NaN periodogram
    # rather than raising), so filter here explicitly.
    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 10:
        raise RuntimeError(
            "guess_lombscargle: fewer than 10 finite (time, flux) points "
            "after dropping NaNs."
        )
    time = time[finite]
    flux = flux[finite]

    dt_lc = np.median(np.diff(time))
    baseline = time[-1] - time[0]
    if min_period is None:
        min_period = 4 * dt_lc
    if max_period is None:
        max_period = baseline / 2

    freq_max = 1.0 / min_period
    freq_min = 1.0 / max_period

    ls = LombScargle(time, flux)
    freq, power = ls.autopower(
        minimum_frequency=freq_min,
        maximum_frequency=freq_max,
        samples_per_peak=samples_per_peak,
    )
    periods = 1.0 / freq

    # find_peaks wants ascending x; periods (from ascending freq) are
    # descending, so sort.
    order = np.argsort(periods)
    periods_sorted = periods[order]
    power_sorted = power[order]

    idx, _ = find_peaks(power_sorted)
    if len(idx) == 0:
        # no interior peak found (e.g. monotonic power) -> fall back to
        # global max
        best_idx = np.array([int(np.argmax(power_sorted))])
    else:
        # rank by power, keep top n_top_peaks
        idx = idx[np.argsort(power_sorted[idx])[::-1][:n_top_peaks]]

    candidate_periods = periods_sorted[idx]
    candidate_powers = power_sorted[idx]

    dt_acf = np.median(np.diff(acf_lags))
    if min_lag is None:
        min_lag = 3 * dt_acf

    ranked = []
    for P_cand in candidate_periods:
        if P_cand >= acf_lags[-1] - min_lag:
            continue  # not even one full period fits in the ACF lag range
        t0_cand = _grid_search_t0(acf_lags, acf, P_cand, min_lag=min_lag)
        score = comb_score(acf_lags, acf, P_cand, t0_cand)
        n_teeth = _teeth_count(P_cand, t0_cand, acf_lags[-1])
        ranked.append((n_teeth >= min_teeth, score, P_cand, t0_cand, n_teeth))

    if len(ranked) == 0:
        raise RuntimeError(
            "guess_lombscargle: no candidate LS period fits within the "
            "available ACF lag range."
        )

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked = [(score, P, t0) for _, score, P, t0, _ in ranked]
    best_score, best_P, best_t0 = ranked[0]

    info = dict(
        periodogram=(periods_sorted, power_sorted),
        candidate_periods=candidate_periods,
        candidate_powers=candidate_powers,
        ranked_candidates=ranked,
    )
    return InitialGuess(P0=best_P, t0=best_t0, method="lombscargle", info=info)


# --------------------------------------------------------------------------
# 3. Initial guess: FFT of the ACF
# --------------------------------------------------------------------------

def guess_acf_fft(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_top_peaks: int = 5,
    window: Optional[Callable[[int], np.ndarray]] = np.hanning,
    min_lag: Optional[float] = None,
    min_teeth: int = 4,
) -> InitialGuess:
    """Take the FFT of the ACF itself (as a function of lag) and look for a
    dominant frequency, i.e. treat the ACF's own quasi-periodicity as a
    signal to be spectrally analyzed. This is a nice complement to the
    Lomb-Scargle-on-the-light-curve method because it operates one
    transform downstream, on a signal (the ACF) that has already had a lot
    of the light curve's stochastic/non-periodic power averaged down by the
    autocorrelation step.

    Requires acf_lags to be evenly spaced (true by construction for the ACF
    of an evenly-sampled light curve).

    Parameters
    ----------
    min_period, max_period : period range to search for a dominant FFT
        frequency. Defaults to [4 * dlag, (lag range)/2].
    n_top_peaks : number of candidate FFT peak periods to test against the
        ACF via comb_score.
    window : windowing function applied to the ACF before the FFT (reduces
        spectral leakage from the finite lag range); set to None to disable.
    min_lag : lower cutoff for phase search on the ACF.
    min_teeth : see guess_pairwise_histogram -- same guard against a
        candidate winning on the strength of only 1-2 comb teeth.

    Returns
    -------
    InitialGuess(P0, t0, method="acf_fft", info={...})
    """
    dt = np.median(np.diff(acf_lags))
    if not np.allclose(np.diff(acf_lags), dt, rtol=1e-3):
        warnings.warn(
            "guess_acf_fft: acf_lags does not appear evenly spaced; "
            "FFT-based period estimate may be unreliable."
        )

    n = len(acf_lags)
    y = acf - np.mean(acf)
    if window is not None:
        y = y * window(n)

    fft_vals = np.fft.rfft(y)
    fft_freq = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft_vals) ** 2

    if min_period is None:
        min_period = 4 * dt
    if max_period is None:
        max_period = (acf_lags[-1] - acf_lags[0]) / 2

    freq_mask = (fft_freq > 1.0 / max_period) & (fft_freq < 1.0 / min_period)
    freq_sub = fft_freq[freq_mask]
    power_sub = power[freq_mask]

    if len(freq_sub) < 3:
        raise RuntimeError(
            "guess_acf_fft: fewer than 3 frequency bins in the requested "
            "period range; widen [min_period, max_period] or check acf_lags."
        )

    idx, _ = find_peaks(power_sub)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(power_sub))])
    else:
        idx = idx[np.argsort(power_sub[idx])[::-1][:n_top_peaks]]

    candidate_freqs = freq_sub[idx]
    candidate_periods = 1.0 / candidate_freqs
    candidate_powers = power_sub[idx]

    if min_lag is None:
        min_lag = 3 * dt

    ranked = []
    for P_cand in candidate_periods:
        if P_cand >= acf_lags[-1] - min_lag:
            continue
        t0_cand = _grid_search_t0(acf_lags, acf, P_cand, min_lag=min_lag)
        score = comb_score(acf_lags, acf, P_cand, t0_cand)
        n_teeth = _teeth_count(P_cand, t0_cand, acf_lags[-1])
        ranked.append((n_teeth >= min_teeth, score, P_cand, t0_cand, n_teeth))

    if len(ranked) == 0:
        raise RuntimeError(
            "guess_acf_fft: no candidate FFT period fits within the "
            "available ACF lag range."
        )

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ranked = [(score, P, t0) for _, score, P, t0, _ in ranked]
    best_score, best_P, best_t0 = ranked[0]

    info = dict(
        fft_freq=freq_sub,
        fft_power=power_sub,
        candidate_periods=candidate_periods,
        candidate_powers=candidate_powers,
        ranked_candidates=ranked,
    )
    return InitialGuess(P0=best_P, t0=best_t0, method="acf_fft", info=info)


# --------------------------------------------------------------------------
# Joint comb fit
# --------------------------------------------------------------------------

@dataclass
class PeakWindow:
    n: int              # harmonic index (0, 1, 2, ...)
    lag_lo: float
    lag_hi: float
    mask: np.ndarray     # boolean mask into acf_lags/acf


def _build_windows(
    acf_lags: np.ndarray,
    P0: float,
    t0: float,
    n_peaks: int,
    window_frac: float = 0.25,
) -> list[PeakWindow]:
    """Build one fitting window per harmonic n=0..n_peaks-1, each spanning
    +/- window_frac * P0 around the expected center t0 + n*P0, clipped to
    the available lag range. Windows are frozen once built (see module
    docstring: point selection happens once, before optimization).
    """
    lag_min, lag_max = acf_lags[0], acf_lags[-1]
    half_width = window_frac * P0
    windows = []
    for n in range(n_peaks):
        center = t0 + n * P0
        lo = max(center - half_width, lag_min)
        hi = min(center + half_width, lag_max)
        if lo >= hi or center > lag_max:
            break
        mask = (acf_lags >= lo) & (acf_lags <= hi)
        if mask.sum() < 4:
            # not enough points to constrain a 3-parameter parabola well
            continue
        windows.append(PeakWindow(n=n, lag_lo=lo, lag_hi=hi, mask=mask))
    return windows


def _build_comb_params(
    windows: list[PeakWindow],
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    t0: float,
    P_bounds_frac: float = 0.3,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
) -> lmfit.Parameters:
    """Construct the tied lmfit.Parameters object for the comb model.

    Model per window n:
        model(lag) = h_n - A_n * (lag - center_n)**2
        center_n   = t0 + n*P0 + delta_n      (delta_n small, optional)
    A_n >= 0 enforces a downward-opening parabola.
    """
    params = lmfit.Parameters()
    params.add("P", value=P0, min=P0 * (1 - P_bounds_frac), max=P0 * (1 + P_bounds_frac))
    params.add("t0", value=t0, min=t0 - 0.5 * P0, max=t0 + 0.5 * P0)

    for w in windows:
        n = w.n
        lag_sub = acf_lags[w.mask]
        acf_sub = acf[w.mask]

        if allow_jitter and n > 0:
            params.add(f"delta_{n}", value=0.0, min=-jitter_frac * P0, max=jitter_frac * P0)
            params.add(f"center_{n}", expr=f"t0 + {n}*P + delta_{n}")
        else:
            # n=0 center is t0 itself, exactly; no jitter needed/allowed
            params.add(f"center_{n}", expr=f"t0 + {n}*P" if n > 0 else "t0")

        h0_guess = float(np.max(acf_sub))
        # crude curvature guess from the window's height drop over its half-width
        half_w = 0.5 * (w.lag_hi - w.lag_lo)
        edge_drop = h0_guess - float(np.min(acf_sub))
        A0_guess = max(edge_drop, 1e-6) / max(half_w**2, 1e-6)

        params.add(f"A_{n}", value=A0_guess, min=0.0)  # forces downward opening
        params.add(f"h_{n}", value=h0_guess)

    return params


def _comb_residual(
    params: lmfit.Parameters,
    windows: list[PeakWindow],
    acf_lags: np.ndarray,
    acf: np.ndarray,
) -> np.ndarray:
    resid = []
    for w in windows:
        n = w.n
        lag_sub = acf_lags[w.mask]
        acf_sub = acf[w.mask]
        c = params[f"center_{n}"].value
        A = params[f"A_{n}"].value
        h = params[f"h_{n}"].value
        model = h - A * (lag_sub - c) ** 2
        resid.append(model - acf_sub)
    return np.concatenate(resid)


@dataclass
class CombFitResult:
    P: float
    P_err: Optional[float]
    t0: float
    t0_err: Optional[float]
    windows: list  # PeakWindow objects actually used in the final fit
    lmfit_result: object
    per_peak: dict  # n -> dict(height, curvature, center, residual_rms, ...)
    n_peaks_used: int
    n_peaks_dropped: int
    redchi: float
    success: bool


def fit_rotation_period(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    initial_guess: InitialGuess,
    n_peaks: int = 5,
    window_frac: float = 0.25,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
    max_reject_iters: int = 3,
    reject_threshold_sigma: float = 4.0,
    min_peaks_required: int = 2,
) -> CombFitResult:
    """Jointly fit a comb of evenly-spaced downward parabolae to the ACF,
    starting from `initial_guess`, with iterative rejection of peaks that
    fit poorly (RANSAC-style).

    Parameters
    ----------
    initial_guess : an InitialGuess from any guess_* function above.
    n_peaks : how many harmonics (n=0..n_peaks-1) to attempt to fit,
        subject to availability within the ACF's lag range.
    window_frac : half-width of each fitting window, as a fraction of P0.
        Points outside these windows never enter the fit (frozen masking).
    allow_jitter : if True, each peak's center may deviate from the exact
        comb position t0+n*P by up to jitter_frac*P, to tolerate gentle
        period drift (e.g. differential rotation) without breaking the
        overall tied structure.
    loss : robust loss function passed to the least_squares backend
        ('linear', 'soft_l1', 'huber', 'cauchy', ...). soft_l1 down-weights
        the influence of any single badly-fit peak on the global (P, t0).
    max_reject_iters : maximum number of prune-and-refit cycles.
    reject_threshold_sigma : a peak is dropped if its window's residual RMS
        exceeds this many sigma above the median residual RMS across all
        currently-fit peaks.
    min_peaks_required : stop rejecting once this few peaks remain, even if
        some still look bad (need at least 2 to define a period at all;
        more is better for a robust result).

    Returns
    -------
    CombFitResult
    """
    P0, t0 = initial_guess.P0, initial_guess.t0
    windows = _build_windows(acf_lags, P0, t0, n_peaks, window_frac=window_frac)
    if len(windows) < min_peaks_required:
        raise RuntimeError(
            f"Only {len(windows)} usable peak window(s) built from the initial "
            f"guess (P0={P0:.4g}, t0={t0:.4g}); need at least "
            f"{min_peaks_required}. Try a larger window_frac, more n_peaks, "
            "or check the initial guess."
        )

    n_dropped_total = 0
    result = None

    for iteration in range(max_reject_iters + 1):
        params = _build_comb_params(
            windows, acf_lags, acf, P0, t0,
            allow_jitter=allow_jitter, jitter_frac=jitter_frac,
        )
        fit_kws = {"loss": loss} if loss != "linear" else {}
        with warnings.catch_warnings():
            # lmfit's stderr/covariance estimate is only approximate under
            # robust losses (soft_l1, huber, ...); it can emit a spurious
            # "invalid value in sqrt" warning while computing it. The P and
            # t0 point estimates themselves are unaffected. If you need
            # trustworthy uncertainties, consider bootstrapping (refit on
            # resampled/perturbed ACF windows) rather than relying on
            # P_err/t0_err from a single robust-loss fit.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = lmfit.minimize(
                _comb_residual,
                params,
                args=(windows, acf_lags, acf),
                method="least_squares",
                **fit_kws,
            )

        # per-peak residual RMS, to decide what (if anything) to drop
        per_peak_rms = {}
        for w in windows:
            n = w.n
            lag_sub = acf_lags[w.mask]
            acf_sub = acf[w.mask]
            c = result.params[f"center_{n}"].value
            A = result.params[f"A_{n}"].value
            h = result.params[f"h_{n}"].value
            model = h - A * (lag_sub - c) ** 2
            rms = float(np.sqrt(np.mean((model - acf_sub) ** 2)))
            per_peak_rms[n] = rms

        if iteration == max_reject_iters or len(windows) <= min_peaks_required:
            break

        rms_vals = np.array(list(per_peak_rms.values()))
        med, mad = np.median(rms_vals), np.median(np.abs(rms_vals - np.median(rms_vals)))
        sigma = 1.4826 * mad if mad > 0 else np.std(rms_vals)
        if sigma == 0:
            break  # nothing to reject on

        worst_n, worst_rms = max(per_peak_rms.items(), key=lambda kv: kv[1])
        if worst_rms > med + reject_threshold_sigma * sigma and len(windows) > min_peaks_required:
            windows = [w for w in windows if w.n != worst_n]
            n_dropped_total += 1
        else:
            break  # nothing egregious left; stop iterating

    # assemble per-peak summary from the final fit
    per_peak = {}
    for w in windows:
        n = w.n
        lag_sub = acf_lags[w.mask]
        acf_sub = acf[w.mask]
        c = result.params[f"center_{n}"].value
        A = result.params[f"A_{n}"].value
        h = result.params[f"h_{n}"].value
        model = h - A * (lag_sub - c) ** 2
        per_peak[n] = dict(
            center=c,
            height=h,
            curvature=A,
            residual_rms=float(np.sqrt(np.mean((model - acf_sub) ** 2))),
            n_points=int(w.mask.sum()),
        )

    P_val = result.params["P"].value
    P_err = result.params["P"].stderr
    t0_val = result.params["t0"].value
    t0_err = result.params["t0"].stderr

    return CombFitResult(
        P=P_val,
        P_err=P_err,
        t0=t0_val,
        t0_err=t0_err,
        windows=windows,
        lmfit_result=result,
        per_peak=per_peak,
        n_peaks_used=len(windows),
        n_peaks_dropped=n_dropped_total,
        redchi=float(result.redchi) if hasattr(result, "redchi") else float("nan"),
        success=bool(result.success) and len(windows) >= min_peaks_required,
    )


# --------------------------------------------------------------------------
# Goodness-of-fit / acceptance helper
# --------------------------------------------------------------------------

def assess_rotation_candidate(
    fit: CombFitResult,
    acf: np.ndarray,
    min_peaks: int = 3,
    max_redchi: float = 5.0,
    min_height_over_local_std: float = 3.0,
) -> dict:
    """Bundle a handful of acceptance diagnostics for a CombFitResult into a
    single dict, analogous to the acceptance criteria used in the original
    SpinSpotter parabola-fit approach, but aggregated across the whole comb
    rather than a single peak.

    This does not make a hard accept/reject call for you (thresholds are
    very target- and noise-regime-dependent) -- it returns the ingredients
    so you can set your own cuts.
    """
    heights = np.array([p["height"] for p in fit.per_peak.values()])
    curvatures = np.array([p["curvature"] for p in fit.per_peak.values()])
    ns = np.array(list(fit.per_peak.keys()))
    order = np.argsort(ns)
    heights, curvatures, ns = heights[order], curvatures[order], ns[order]

    acf_std = float(np.std(acf))
    height_snr = heights / acf_std if acf_std > 0 else heights * np.nan

    # monotonic-ish decay check: fraction of consecutive peaks where height
    # doesn't increase (real rotation signals damp or stay flat with lag;
    # a peak that's much taller than its predecessor is suspicious)
    if len(heights) > 1:
        non_increasing = np.diff(heights) <= 1e-3 * np.abs(heights[:-1])
        frac_non_increasing = float(np.mean(non_increasing))
    else:
        frac_non_increasing = np.nan

    return dict(
        n_peaks_used=fit.n_peaks_used,
        n_peaks_dropped=fit.n_peaks_dropped,
        redchi=fit.redchi,
        heights=heights,
        height_snr=height_snr,
        curvatures=curvatures,
        frac_non_increasing_height=frac_non_increasing,
        passes_min_peaks=fit.n_peaks_used >= min_peaks,
        passes_redchi=fit.redchi <= max_redchi if np.isfinite(fit.redchi) else False,
        passes_height_snr=bool(np.all(height_snr >= min_height_over_local_std)),
    )


# --------------------------------------------------------------------------
# Ensemble arbitration across initial-guess methods
# --------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """One tested candidate period's outcome, for inspection/plotting."""
    period: float
    t0: float
    fit: Optional[CombFitResult]
    diagnostics: Optional[dict]
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    best_fit: CombFitResult
    best_guess: InitialGuess
    candidates: list  # list[CandidateResult], sorted best-first (successful ones only)
    n_candidates_tried: int
    failed_methods: dict  # method name -> exception message, for methods that errored entirely


def find_best_rotation_period(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    methods: tuple = ("pairwise_histogram", "lombscargle", "acf_fft"),
    n_top_peaks: int = 15,
    min_teeth: int = 4,
    extra_period_factors: tuple = (0.5, 2.0, 1.0 / 3.0, 3.0),
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    dedup_rel_tol: float = 0.03,
    n_peaks: int = 8,
    window_frac: float = 0.25,
    min_peaks_required: int = 4,
    fit_kwargs: Optional[dict] = None,
    guess_kwargs: Optional[dict] = None,
) -> EnsembleResult:
    """Ensemble initial-guess + joint-fit arbitration.

    Why this exists: each guess_* function ultimately picks its single best
    candidate using a *cheap* heuristic (comb_score / peak coverage), which
    can be fooled -- especially on noisy or weak-signal data -- by
    candidates that only have to explain a handful of comb teeth. The joint
    parabola fit is a much stronger test (every window's shape has to
    actually look like a downward parabola, not just have a tall-ish
    point), but it's too expensive to use as the *search* criterion, only
    as the *arbiter*. So: cast a wide net across all three guess methods'
    full candidate pools (not just their individual best picks), expand
    with harmonics/subharmonics to catch cases where the true fundamental
    didn't even make a method's own shortlist, then run the real joint fit
    on every surviving candidate and let reduced chi-squared (with a
    minimum peak-count requirement) pick the winner.

    Parameters
    ----------
    methods : which guess_* methods to draw candidates from.
    n_top_peaks : forwarded to guess_lombscargle/guess_acf_fft -- how many
        of *their own* top candidates to pull into the shared pool. Set
        this generously (10-20) for noisy data: the point of this function
        is that a candidate doesn't need to be any single method's #1 pick,
        it just needs to be *somewhere* in the pool.
    min_teeth : forwarded to all three guess_* functions.
    extra_period_factors : each candidate period P pulled from any method
        is also tested at P*factor for factor in this tuple, to catch
        cases where a method's candidate list only contains a harmonic or
        subharmonic of the true period, not the true period itself.
    min_period, max_period : optional hard prior bounds; any candidate
        (including harmonic-expanded ones) outside this range is dropped
        before fitting. Strongly recommended if you have *any* external
        expectation (spectral type, literature, visual inspection) -- it
        both speeds this up and keeps obviously-impossible candidates
        (e.g. sub-cadence or longer-than-baseline) out of contention.
    dedup_rel_tol : candidates within this relative tolerance of each other
        are treated as duplicates (only the first is kept), to avoid
        wastefully re-fitting near-identical periods.
    n_peaks, window_frac, min_peaks_required, fit_kwargs : forwarded to
        fit_rotation_period for every candidate.
    guess_kwargs : optional dict of dicts, e.g. {"lombscargle": {...}},
        forwarded to the respective guess_* call.

    Returns
    -------
    EnsembleResult(best_fit, best_guess, candidates, n_candidates_tried,
                   failed_methods)
        `candidates` is every successfully-fit candidate, sorted best-first
        (by: meets min_peaks_required, then lowest reduced chi-squared), so
        you can inspect runner-up periods -- useful for spotting genuine
        ambiguity (e.g. two candidates with comparably good fits) rather
        than trusting a single top pick blindly.
    """
    guess_kwargs = guess_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    guess_fns = {
        "pairwise_histogram": guess_pairwise_histogram,
        "lombscargle": guess_lombscargle,
        "acf_fft": guess_acf_fft,
    }

    dt_acf = np.median(np.diff(acf_lags))
    lag_span = acf_lags[-1] - acf_lags[0]
    lo_bound = min_period if min_period is not None else 2 * dt_acf
    hi_bound = max_period if max_period is not None else lag_span

    # --- 1. gather candidate periods from every requested method ---
    pool = set()
    failed_methods = {}
    for method in methods:
        if method not in guess_fns:
            raise ValueError(f"Unknown method '{method}'.")
        kwargs = dict(min_teeth=min_teeth)
        if method in ("lombscargle", "acf_fft"):
            kwargs["n_top_peaks"] = n_top_peaks
        if min_period is not None:
            kwargs["min_period"] = min_period
        if max_period is not None:
            kwargs["max_period"] = max_period
        kwargs.update(guess_kwargs.get(method, {}))
        try:
            g = guess_fns[method](time, flux, acf_lags, acf, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- one method failing shouldn't block the rest
            failed_methods[method] = f"{type(exc).__name__}: {exc}"
            continue
        pool.add(float(g.P0))
        cps = g.info.get("candidate_periods")
        if cps is not None:
            pool.update(float(p) for p in np.atleast_1d(cps))

    if len(pool) == 0:
        raise RuntimeError(
            "find_best_rotation_period: every guess method failed. "
            f"Errors: {failed_methods}"
        )

    # --- 2. expand with harmonics/subharmonics ---
    expanded = set(pool)
    for p in pool:
        for factor in extra_period_factors:
            expanded.add(p * factor)

    # --- 3. apply prior bounds, then deduplicate near-identical candidates ---
    survivors = sorted(p for p in expanded if lo_bound <= p <= hi_bound)
    deduped = []
    for p in survivors:
        if not deduped or (p - deduped[-1]) / deduped[-1] > dedup_rel_tol:
            deduped.append(p)

    # --- 4. run the real joint fit on every surviving candidate ---
    min_lag = 3 * dt_acf
    results = []
    for P_cand in deduped:
        t0_cand = _grid_search_t0(acf_lags, acf, P_cand, min_lag=min_lag)
        guess = InitialGuess(P0=P_cand, t0=t0_cand, method="ensemble_candidate")
        try:
            fit = fit_rotation_period(
                acf_lags, acf, guess,
                n_peaks=n_peaks, window_frac=window_frac,
                min_peaks_required=min(min_peaks_required, 2),
                **fit_kwargs,
            )
            diag = assess_rotation_candidate(fit, acf, min_peaks=min_peaks_required)
            results.append(CandidateResult(period=P_cand, t0=t0_cand, fit=fit, diagnostics=diag))
        except Exception as exc:  # noqa: BLE001 -- skip candidates that can't be fit, keep going
            results.append(CandidateResult(
                period=P_cand, t0=t0_cand, fit=None, diagnostics=None,
                error=f"{type(exc).__name__}: {exc}",
            ))

    successful = [r for r in results if r.fit is not None]
    if len(successful) == 0:
        raise RuntimeError(
            "find_best_rotation_period: no candidate period could be "
            "successfully fit. Try lowering min_peaks_required or "
            "widening window_frac."
        )

    def _frac_positive_heights(r) -> float:
        heights = np.array([p["height"] for p in r.fit.per_peak.values()])
        if len(heights) == 0:
            return 0.0
        return float(np.mean(heights > 0))

    def _sort_key(r):
        redchi = r.fit.redchi if np.isfinite(r.fit.redchi) else np.inf
        frac_pos = _frac_positive_heights(r)
        # Gate 1: enough peaks retained. Gate 2: fitted heights are
        # genuinely positive bumps, not just tracking the ACF's smooth
        # decay down through zero near lag=0 (a monotonically-decaying,
        # sign-changing "comb" can trivially achieve a tiny reduced
        # chi-squared by hugging that smooth curve -- it isn't periodicity,
        # it's fitting parabolae to a slope). Requiring >= 80% of teeth to
        # have positive height rules this out without needing a hand-tuned
        # minimum period.
        return (
            r.fit.n_peaks_used >= min_peaks_required,
            frac_pos >= 0.8,
            -redchi,
        )

    successful.sort(key=_sort_key, reverse=True)

    best = successful[0]
    best_guess = InitialGuess(P0=best.period, t0=best.t0, method="ensemble_best")

    return EnsembleResult(
        best_fit=best.fit,
        best_guess=best_guess,
        candidates=successful,
        n_candidates_tried=len(deduped),
        failed_methods=failed_methods,
    )

