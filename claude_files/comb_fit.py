"""
comb_fit.py

Identify stellar rotation periods from the autocorrelation function (ACF) of
a light curve by jointly fitting a "comb" of evenly-spaced, downward-opening
parabolae to the repeated peaks of the ACF.

Design (v2)
-----------
There are two clearly separated stages, and it's worth understanding why
they're separated:

1. CANDIDATE GENERATION (guess_pairwise_histogram, guess_lombscargle,
   guess_acf_fft). Each of these looks at the data through a different lens
   (a discrete list of ACF peaks; the light curve's own periodogram; the
   ACF's own spectrum) and proposes a short list of candidate periods, ranked
   by whatever evidence is native to that lens (peak-spacing support count;
   periodogram power; FFT power). Critically, NONE of these functions checks
   its candidates against the ACF's actual shape, and none of them does any
   curve fitting. They are cheap and fast, and their only job is to narrow
   an enormous search space (periods from hours to tens of days) down to a
   short list worth taking seriously.

2. FITTING AND ARBITRATION (fit_rotation_period). This is where the real
   evaluation happens: every candidate from every method is fed through the
   same joint least-squares comb-of-parabolae fit against the actual ACF,
   and the results are compared on equal footing. This is the only stage
   that looks at how well a candidate's predicted peaks actually match the
   ACF's shape, height, and spacing simultaneously -- which is a much
   stronger test than any single candidate-generation heuristic, and it's
   why candidate generation doesn't need to be clever or "correct" on its
   own: it just needs to not leave the right answer off the list.

Why this split matters in practice: earlier versions of this module had
each guess_* function do its own cross-validation against the ACF (a cheap
"comb score") to try to pick a single best candidate internally. That made
each function harder to reason about, and the cheap score turned out to be
gameable (candidates with very few "teeth" in range could win on weak
evidence). Separating "propose candidates" from "evaluate candidates
properly" removes that failure mode and makes each piece easier to
understand and to test independently.

fit_rotation_period is also deliberately willing to say "I couldn't find a
reliable period" (EnsembleResult.success = False) rather than always
returning its best guess. A best guess that didn't clear basic plausibility
checks (enough peaks, peaks that are genuinely positive bumps rather than
noise, peaks tall enough relative to the ACF's noise floor) is often worse
than no answer at all.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
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

    order = np.argsort(peak_lags)
    return peak_lags[order], peak_heights[order], idx[order]


def _teeth_count(P: float, t0: float, lag_max: float) -> int:
    """Number of comb teeth (n=0,1,2,...) that land within [t0, lag_max]
    for period P. Used only as a cheap sanity filter (e.g. discard a
    candidate period so long that fewer than ~2 teeth would even fit in the
    ACF's lag range) -- not for ranking candidates against each other.
    """
    if P <= 0:
        return 0
    return int(np.floor((lag_max - t0) / P)) + 1


def default_comb_weight(n: int) -> float:
    """Default per-tooth weight for comb_score: 1/(n+1). Teeth at low n
    (short lags) are weighted more heavily than teeth at high n (long
    lags). This reflects a real physical expectation: starspots evolve
    (grow, decay, migrate in longitude) on timescales that are often not
    much longer than the rotation period itself, so the periodic signal
    typically becomes less coherent -- and the corresponding ACF peaks
    genuinely weaker and less trustworthy -- at longer lags. Weighting the
    early teeth more heavily makes phase/period estimates rely more on the
    part of the ACF where the periodic signal is most likely to still look
    like the star's actual current rotation, rather than being pulled
    around by whatever noise happens to be doing at lag 10P.
    """
    return 1.0 / (n + 1)


def comb_score(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P: float,
    t0: float,
    n_max: Optional[int] = None,
    lag_max: Optional[float] = None,
    weight_func: Optional[Callable[[int], float]] = None,
) -> float:
    """Score how well a comb of teeth at t0 + n*P (n=0,1,2,...) lines up
    with tall points of the ACF. Larger is better.

    This is a *weighted* average of the (linearly-interpolated) ACF value
    at each comb tooth, using `weight_func(n)` as the weight for tooth n
    (default: default_comb_weight, i.e. 1/(n+1) -- see its docstring for
    why). It is intentionally cheap (no fitting) and is used only for two
    things in this module: (a) picking the best phase t0 for a given
    candidate period during _grid_search_t0, and (b) optional diagnostics.
    It is NOT used to rank candidate periods against each other -- that's
    what the actual joint comb fit in fit_rotation_period is for.
    """
    if weight_func is None:
        weight_func = default_comb_weight
    if lag_max is None:
        lag_max = acf_lags[-1]
    if n_max is None:
        n_max = int(np.floor((lag_max - t0) / P)) if P > 0 else 0
    n_max = max(n_max, 0)

    n = np.arange(0, n_max + 1)
    comb_lags = t0 + n * P
    in_range = comb_lags <= lag_max
    comb_lags = comb_lags[in_range]
    n = n[in_range]
    if len(comb_lags) == 0:
        return -np.inf

    vals = np.interp(comb_lags, acf_lags, acf)
    weights = np.array([weight_func(int(nn)) for nn in n])
    return float(np.sum(weights * vals) / np.sum(weights))


def _grid_search_t0(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    min_lag: float,
    n_phase: int = 200,
    n_teeth_for_score: int = 5,
    weight_func: Optional[Callable[[int], float]] = None,
) -> float:
    """Given a candidate period P0, grid-search the phase t0 in
    [min_lag, min_lag + P0) that maximizes comb_score. The returned t0 is
    itself the lag of the (n=0) tooth. This is a coarse, cheap phase
    estimate meant to seed the real joint fit -- not a fit in itself.
    """
    trial_t0 = np.linspace(min_lag, min_lag + P0, n_phase, endpoint=False)
    scores = [
        comb_score(acf_lags, acf, P0, t0, n_max=n_teeth_for_score, weight_func=weight_func)
        for t0 in trial_t0
    ]
    return float(trial_t0[int(np.argmax(scores))])


def _peak_coverage_fraction(
    peak_lags: np.ndarray,
    P: float,
    t0: float,
    tolerance: Optional[float] = None,
) -> float:
    """Fraction of the *found* ACF peaks (peak_lags) that land within
    `tolerance` of some comb tooth t0 + n*P. Reported as a diagnostic on
    each pairwise_histogram candidate (see guess_pairwise_histogram) but is
    not used to rank candidates -- ranking is by histogram support count,
    which is simpler to reason about and already resistant to harmonic
    ambiguity (see that function's docstring).
    """
    if tolerance is None:
        tolerance = 0.1 * P
    if len(peak_lags) == 0:
        return 0.0
    n_est = np.round((peak_lags - t0) / P)
    nearest_tooth = t0 + n_est * P
    matched = np.abs(peak_lags - nearest_tooth) <= tolerance
    return float(np.mean(matched))


@dataclass
class InitialGuess:
    """One candidate period proposed by a guess_* function.

    t0 is deliberately optional: candidate generation (stage 1, see module
    docstring) does not compute a phase at all -- that happens later in
    fit_rotation_period, once a candidate is actually being fit. If you
    construct an InitialGuess by hand with a known t0, that's fine too;
    fit_rotation_period will use it as-is instead of re-deriving one.
    """
    P0: float
    method: str
    rank: int = 0          # 1 = strongest candidate from this method's call, 2 = next, ...
    strength: float = float("nan")  # method-specific normalized score, higher = more confident
    t0: Optional[float] = None
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 1. Candidate generation: find_peaks + pairwise spacing histogram
# --------------------------------------------------------------------------

def guess_pairwise_histogram(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_lag: Optional[float] = None,
    max_lag: Optional[float] = None,
    prominence: Optional[float] = None,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_hist_bins: Optional[int] = None,
    n_guesses: int = 5,
) -> list:
    """Propose candidate periods from the spacing between ACF peaks.

    Full mechanism, step by step
    -----------------------------
    1. Find local maxima ("peaks") of the ACF itself, using scipy's
       find_peaks, excluding a small buffer around lag 0 (which is always a
       trivial peak -- every signal is perfectly correlated with itself at
       zero lag -- and isn't rotation information). Call the resulting peak
       positions x_1, x_2, ..., x_m (sorted by lag). If the star's rotation
       signal is present, these should include the ACF's repeated
       "harmonics" at roughly P, 2P, 3P, ... (not necessarily all of them --
       some may be too weak to register as a distinct peak, especially at
       longer lags).

    2. Compute every pairwise POSITIVE difference x_j - x_i for j > i. If
       the true peaks really do sit at P, 2P, 3P, ..., these differences are
       not random: adjacent peaks differ by ~P, peaks two apart differ by
       ~2P, three apart by ~3P, and so on. So the *set* of all pairwise
       differences is a mix of P, 2P, 3P, ... with P itself appearing the
       most often (see step 4).

    3. Histogram all of those differences. Because of measurement noise and
       finite lag resolution, the differences near each of P, 2P, 3P, ...
       won't be exactly equal, but they cluster tightly enough that each of
       these multiples shows up as its own local maximum ("bump") in the
       histogram, rather than being smeared into a flat distribution.

    4. Each local maximum in the histogram is a candidate period. Candidates
       are ranked by the height of their histogram bin, i.e. by how many
       pairs of found peaks support that spacing. This is the key idea that
       makes this method resistant to harmonic confusion without needing
       any extra cross-checking: if there are m peaks found and they are
       (roughly) evenly spaced, the true fundamental spacing P is supported
       by up to (m-1) pairs (every adjacent pair), spacing 2P is supported
       by only (m-2) pairs (every other peak), spacing 3P by (m-3) pairs,
       and so on. The support count strictly decreases as you move to
       higher multiples of the true period. So simply ranking candidates by
       "how many pairs agree this is the spacing" naturally favors the
       fundamental over its harmonics, PROVIDED the underlying peak-finding
       in step 1 is reasonably clean. It is not foolproof (a few spurious
       or missed peaks can shuffle the ranking, which is exactly why this
       function returns its top n_guesses candidates rather than committing
       to just one -- the real arbitration happens later, in
       fit_rotation_period, by actually fitting each candidate against the
       full ACF).

    5. Return the top n_guesses candidates (by support count), each carrying
       `strength` = count / (m-1) -- the fraction of the maximum possible
       support (a perfectly clean, fully-covered fundamental would score
       1.0) -- plus a `coverage` diagnostic (see _peak_coverage_fraction)
       in `info`, computed using the smallest found peak as an approximate
       phase anchor (this is only for your inspection; it is not used for
       ranking).

    This function never looks at the ACF's continuous shape beyond the
    initial peak-finding step, never computes a phase/t0, and never fits
    anything -- see the module docstring for why that's intentional.

    Parameters
    ----------
    time, flux : included for interface consistency with the other guess_*
        functions; not used by this method.
    min_lag : lower cutoff on lag to search for peaks (default: 3x the
        median lag spacing, to exclude the trivial lag-0 peak).
    max_lag : upper cutoff on lag to search for peaks (default: full range).
    prominence : passed to scipy.find_peaks; raise this to reject noise
        peaks in noisy ACFs. Default (None): adaptive, set to
        5x the standard deviation of the ACF's second difference -- a
        robust proxy for the ACF's local point-to-point noise level, which
        scales naturally with how noisy a given target's ACF actually is
        (a fixed absolute default does not: e.g. 0.01 is far too loose for
        a clean, high-amplitude ACF with thousands of tiny noise wiggles
        above it, and could be too strict for a very weak, noisy signal).
    min_period, max_period : if given, restrict candidate spacings to this
        range before histogramming.
    n_hist_bins : number of bins for the spacing histogram. Default is
        chosen from the lag resolution.
    n_guesses : how many top candidates to return.

    Returns
    -------
    list[InitialGuess], sorted strongest-first (by support count), method="pairwise_histogram"
    """
    dt = np.median(np.diff(acf_lags))
    if min_lag is None:
        min_lag = 3 * dt
    if max_lag is None:
        max_lag = acf_lags[-1]
    if prominence is None:
        prominence = 5.0 * np.std(np.diff(acf, 2))

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
        bin_width = max(4 * dt, (diffs.max() - diffs.min()) / 200)
    else:
        bin_width = (diffs.max() - diffs.min()) / n_hist_bins

    # Histogram range starts at 0 (or a small floor), not diffs.min(): if the
    # range started at diffs.min(), the fundamental spacing (often close to
    # the smallest pairwise difference) would sit in the very first bin, and
    # scipy.find_peaks can never flag an edge bin as a local maximum.
    hist_lo = max(0.0, diffs.min() - bin_width)
    hist_hi = diffs.max() + bin_width
    n_bins = max(int((hist_hi - hist_lo) / bin_width), 10)
    hist, bin_edges = np.histogram(diffs, bins=n_bins, range=(hist_lo, hist_hi))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_peak_idx, _ = find_peaks(hist, height=1)
    if len(hist_peak_idx) == 0:
        hist_peak_idx = np.array([int(np.argmax(hist))])

    candidate_periods = bin_centers[hist_peak_idx]
    candidate_counts = hist[hist_peak_idx]

    # rank by support count, descending; keep top n_guesses
    order = np.argsort(candidate_counts)[::-1][:n_guesses]
    max_possible_support = max(len(peak_lags) - 1, 1)

    t0_anchor = float(peak_lags[0])  # cheap phase anchor for the coverage diagnostic only
    guesses = []
    for rank, idx in enumerate(order, start=1):
        P_cand = float(candidate_periods[idx])
        count = int(candidate_counts[idx])
        coverage = _peak_coverage_fraction(peak_lags, P_cand, t0_anchor)
        guesses.append(InitialGuess(
            P0=P_cand,
            method="pairwise_histogram",
            rank=rank,
            strength=count / max_possible_support,
            info=dict(
                peak_lags=peak_lags,
                peak_heights=peak_heights,
                histogram=(bin_centers, hist),
                support_count=count,
                coverage=coverage,
            ),
        ))
    return guesses


# --------------------------------------------------------------------------
# 2. Candidate generation: Lomb-Scargle periodogram of the light curve
# --------------------------------------------------------------------------

def guess_lombscargle(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_guesses: int = 5,
    samples_per_peak: int = 1,
) -> list:
    """Propose candidate periods from peaks in the Lomb-Scargle periodogram
    of the light curve itself.

    Mechanism: compute the LS periodogram (astropy, standard normalization,
    so power is bounded and roughly comparable across targets), find its
    local maxima with scipy.find_peaks, and return the top n_guesses by
    power. This function does not look at the ACF at all -- it is a purely
    light-curve-domain candidate source, complementary to the two ACF-based
    methods below. No phase/t0 is computed here (see module docstring).

    Parameters
    ----------
    min_period, max_period : period search range for the periodogram.
        Defaults to [4 * median(dt), (time[-1]-time[0])/2].
    n_guesses : how many top candidates to return.
    samples_per_peak : oversampling factor passed to astropy's autopower.
        Default lowered from astropy's own default of ~5-10 down to 1:
        this function only needs to propose a *coarse* candidate period for
        fit_rotation_period to refine via the real joint comb fit
        afterward, not a precisely-resolved one, so there's little value
        in a finely-sampled frequency grid here. Measured on a ~140k-point
        real TESS light curve, samples_per_peak=10 (the old default)
        evaluates roughly 4.5x more frequency samples than there are data
        points and takes several seconds; samples_per_peak=1 is ~4x
        faster with an essentially unchanged top candidate (the small
        remaining period error is well within what the joint fit corrects
        for). Raise this if you need this function's own candidates to be
        precise without relying on the joint fit at all.

    Returns
    -------
    list[InitialGuess], sorted strongest-first (by LS power), method="lombscargle"
    """
    if LombScargle is None:
        raise ImportError(
            "guess_lombscargle requires astropy (`pip install astropy`)."
        )

    # Real light curves are often shipped as NaN at missing cadences on an
    # otherwise even grid; LombScargle silently returns all-NaN power if fed
    # NaNs, so filter explicitly.
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

    ls = LombScargle(time, flux, normalization="standard")
    freq, power = ls.autopower(
        minimum_frequency=freq_min,
        maximum_frequency=freq_max,
        samples_per_peak=samples_per_peak,
    )
    periods = 1.0 / freq

    # find_peaks wants ascending x; periods (from ascending freq) descend.
    order = np.argsort(periods)
    periods_sorted = periods[order]
    power_sorted = power[order]

    idx, _ = find_peaks(power_sorted)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(power_sorted))])

    # rank by power, keep top n_guesses
    idx = idx[np.argsort(power_sorted[idx])[::-1][:n_guesses]]

    guesses = []
    for rank, i in enumerate(idx, start=1):
        guesses.append(InitialGuess(
            P0=float(periods_sorted[i]),
            method="lombscargle",
            rank=rank,
            strength=float(power_sorted[i]),  # 'standard' normalization is already ~[0, 1]
            info=dict(
                periodogram=(periods_sorted, power_sorted),
                # kept so later feature-extraction code can compute a
                # false-alarm probability for any period (not just these
                # top n_guesses) without recomputing the periodogram from
                # scratch -- see ml_features.py.
                ls_object=ls,
            ),
        ))
    return guesses


# --------------------------------------------------------------------------
# 3. Candidate generation: FFT of the ACF
# --------------------------------------------------------------------------

def guess_acf_fft(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_guesses: int = 5,
    window: Optional[Callable[[int], np.ndarray]] = np.hanning,
    oversample: int = 4,
) -> list:
    """Propose candidate periods from peaks in the FFT power spectrum of
    the ACF itself (treating the ACF's own quasi-periodicity as a signal).

    Mechanism: window and zero-pad the ACF, take its real FFT, find local
    maxima in the power spectrum within [min_period, max_period], and
    return the top n_guesses by power. Like guess_lombscargle, this does
    not check candidates against the ACF's shape/phase -- see module
    docstring.

    The `oversample` parameter controls zero-padding via the `n` argument
    of np.fft.rfft: the ACF (length N) is padded to `oversample * N` points
    before transforming. This does not add new information, but it
    interpolates the underlying (smooth) power spectrum onto a finer
    frequency grid, which noticeably improves how precisely a real peak's
    location can be read off -- the native (unpadded) FFT of a
    several-hundred-point ACF has quite coarse period resolution,
    especially at long periods where a single native frequency bin can
    span a substantial fraction of a day.

    Parameters
    ----------
    min_period, max_period : period range to search. Defaults to
        [4 * dlag, (lag range)/2].
    n_guesses : how many top candidates to return.
    window : windowing function applied to the ACF before the FFT (reduces
        spectral leakage from the finite lag range); set to None to disable.
    oversample : zero-padding factor for the FFT (see above). 1 disables
        padding (native resolution); 4 (default) quadruples the number of
        frequency samples.

    Returns
    -------
    list[InitialGuess], sorted strongest-first (by FFT power), method="acf_fft"
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

    n_fft = max(int(oversample * n), n)
    fft_vals = np.fft.rfft(y, n=n_fft)
    fft_freq = np.fft.rfftfreq(n_fft, d=dt)
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
            "period range; widen [min_period, max_period], check acf_lags, "
            "or increase `oversample`."
        )

    idx, _ = find_peaks(power_sub)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(power_sub))])
    idx = idx[np.argsort(power_sub[idx])[::-1][:n_guesses]]

    power_max = float(np.max(power_sub))
    guesses = []
    for rank, i in enumerate(idx, start=1):
        guesses.append(InitialGuess(
            P0=float(1.0 / freq_sub[i]),
            method="acf_fft",
            rank=rank,
            strength=float(power_sub[i] / power_max) if power_max > 0 else 0.0,
            info=dict(fft_freq=freq_sub, fft_power=power_sub),
        ))
    return guesses


# --------------------------------------------------------------------------
# Short-period-focused variants (optional)
# --------------------------------------------------------------------------
#
# Motivation: for a genuinely short-period signal (say < 10 days), a
# LombScargle periodogram or ACF FFT spectrum computed over the pipeline's
# usual full range (up to ~baseline/2, i.e. potentially hundreds of days)
# makes that short-period peak compete against every longer-period feature
# in the spectrum -- including spurious long-period power that has nothing
# to do with rotation. Restricting the search band *before* ranking means a
# short-period peak only has to beat other short-period candidates. Neither
# of these methods is in gather_initial_guesses' default `methods` tuple
# (like "wavelet", they're opt-in): pass them explicitly, e.g.
# methods=("pairwise_histogram", "lombscargle", "acf_fft",
# "lombscargle_short", "acf_fft_short").

def guess_lombscargle_short(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    max_period: float = 15.0,
    min_period: Optional[float] = None,
    n_guesses: int = 5,
    samples_per_peak: int = 5,
) -> list:
    """Short-period-focused wrapper around guess_lombscargle.

    Restricts the search grid to [min_period, max_period] *before*
    ranking, so a genuine short-period peak only has to beat other
    short-period candidates -- not every longer-period feature in the
    full periodogram. samples_per_peak is raised from guess_lombscargle's
    default of 1, since narrowing the range means far fewer frequencies
    are evaluated, so a finer grid costs little extra compute here.

    Note on cost: this calls guess_lombscargle again with a different
    (narrower, finer) frequency grid than a normal "lombscargle" call
    would use, so it does re-run astropy's LombScargle.autopower --
    unavoidably, since the two calls genuinely use different resolutions
    and ranges, not the same computation twice. It is still cheap: a
    narrower band at finer sampling evaluates roughly the same order of
    frequency points as the default wide-but-coarse call (see this
    project's timing notes), not several times more.

    Parameters
    ----------
    max_period : upper edge of the short-period search band (days).
    min_period : lower edge; defaults to guess_lombscargle's own default
        (4 * median cadence).
    n_guesses, samples_per_peak : forwarded to guess_lombscargle.

    Returns
    -------
    list[InitialGuess], method relabeled "lombscargle_short" so it's
    tracked separately from the ordinary "lombscargle" candidates.
    """
    guesses = guess_lombscargle(
        time, flux, acf_lags, acf,
        min_period=min_period, max_period=max_period,
        n_guesses=n_guesses, samples_per_peak=samples_per_peak,
    )
    return [replace(g, method="lombscargle_short") for g in guesses]


def guess_acf_fft_short(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    max_period: float = 15.0,
    min_period: Optional[float] = None,
    n_guesses: int = 5,
    oversample: int = 8,
) -> list:
    """Short-period-focused wrapper around guess_acf_fft. oversample is
    raised from the default 4, for the same reason as
    guess_lombscargle_short raises samples_per_peak -- a narrower band
    affords finer resolution cheaply.

    Note on cost: unlike the LombScargle case, this recomputes the FFT of
    the (already-computed) `acf` array at a different zero-padding
    factor -- an FFT of an array with a few thousand points is on the
    order of milliseconds, so this recomputation is not worth avoiding
    even though it is, strictly, a second FFT of the same underlying ACF.

    Returns
    -------
    list[InitialGuess], method relabeled "acf_fft_short".
    """
    guesses = guess_acf_fft(
        time, flux, acf_lags, acf,
        min_period=min_period, max_period=max_period,
        n_guesses=n_guesses, oversample=oversample,
    )
    return [replace(g, method="acf_fft_short") for g in guesses]


def _highpass_flux(time: np.ndarray, flux: np.ndarray, window_days: float) -> np.ndarray:
    """Subtract a centered rolling-mean trend (NaN-aware) from flux, to
    isolate variability faster than `window_days` and remove slower
    variability that would otherwise dominate a periodogram or ACF. Used
    by guess_acf_fft_highpass -- see its docstring for why this helps
    short-period recovery specifically.
    """
    dt = np.median(np.diff(time))
    window_pts = max(1, int(round(window_days / dt)))
    if window_pts <= 1:
        return flux.copy()
    trend = pd.Series(flux).rolling(
        window_pts, center=True, min_periods=max(1, window_pts // 3)
    ).mean().to_numpy()
    return flux - trend


def guess_acf_fft_highpass(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    smooth_windows: tuple = (2.0, 5.0, 10.0, 20.0, 40.0),
    max_period: float = 15.0,
    min_period: Optional[float] = None,
    n_guesses: int = 5,
    oversample: int = 8,
    max_lag_frac: float = 1.0 / 3,
    min_valid_frac: float = 0.3,
) -> list:
    """Short-period-focused candidate generation via a high-pass-filtered
    ACF: remove slow variability from the light curve BEFORE computing the
    ACF, so a weak short-period signal isn't swamped by a stronger
    longer-timescale trend (real long-period rotation, spot evolution, or
    residual systematics) in either the ACF itself or its FFT.

    Mechanism, step by step
    ------------------------
    1. For each window in `smooth_windows`, high-pass filter the flux
       (_highpass_flux): compute a centered rolling-mean trend over that
       window and subtract it, leaving only variability faster than the
       window. Each window is a genuinely different filtering choice, not
       a resolution knob -- a too-short window risks attenuating the very
       short-period signal you're trying to recover along with the trend,
       a too-long window leaves more slow variability in the residual to
       compete with it, so trying a couple of different windows (rather
       than committing to one) hedges against either failure mode. The
       defaults (20, 40 days) are several times the default
       `max_period=15.0` specifically so the short-period band of interest
       is never close to the filtering timescale itself.
    2. Recompute the ACF (via acf_utils.compute_acf, imported locally to
       avoid making this module depend on it for callers who don't use
       this function) on the high-pass-filtered flux -- this is a
       genuinely new computation each time, since the underlying signal
       changed; there is no way to reuse the original (unfiltered) `acf`
       argument here (it is accepted only for interface consistency with
       the other guess_* functions and is not used).
    3. Run guess_acf_fft on that new ACF, restricted to
       [min_period, max_period] (see guess_acf_fft_short for why
       restricting the band before ranking helps).
    4. Tag each returned candidate's method as "acf_fft_hp{window}d" so
       candidates from different smoothing windows are tracked separately
       (e.g. "acf_fft_hp20d", "acf_fft_hp40d").

    Parameters
    ----------
    smooth_windows : rolling-mean trend windows to try, in days.
    max_period, min_period : forwarded to guess_acf_fft for the (fresh)
        high-pass-filtered ACF.
    n_guesses, oversample : forwarded to guess_acf_fft.
    max_lag_frac, min_valid_frac : forwarded to acf_utils.compute_acf for
        each recomputed ACF.

    Returns
    -------
    list[InitialGuess], candidates from every window concatenated
    together, each method-tagged by its originating window.
    """
    from acf_utils import compute_acf as _compute_acf

    all_guesses = []
    for window in smooth_windows:
        flux_hp = _highpass_flux(time, flux, window)
        lags_hp, acf_hp = _compute_acf(
            time, flux_hp, max_lag_frac=max_lag_frac, min_valid_frac=min_valid_frac
        )
        try:
            guesses = guess_acf_fft(
                time, flux_hp, lags_hp, acf_hp,
                min_period=min_period, max_period=max_period,
                n_guesses=n_guesses, oversample=oversample,
            )
        except Exception:  # noqa: BLE001 -- one bad window shouldn't sink the rest
            continue
        tag = f"acf_fft_hp{window:g}d"
        all_guesses.extend(replace(g, method=tag) for g in guesses)
    return all_guesses


# --------------------------------------------------------------------------
# 4. Candidate generation: Global Wavelet Power Spectrum (optional)
# --------------------------------------------------------------------------
#
# Unlike the three methods above, this one is NOT included in
# gather_initial_guesses' default `methods` tuple -- it's registered in its
# guess_fns lookup so it's available on request (methods=(..., "wavelet")),
# but never runs unless a caller explicitly asks for it. See guess_wavelet's
# docstring for why: it needs gap-free, evenly-sampled flux (no NaNs), which
# not every light curve satisfies, and it's meaningfully more expensive than
# the other three methods (see the relative-speed comparison in this
# project's notes).

def _period_to_morlet_scale(period: np.ndarray, dt: float, w0: float) -> np.ndarray:
    """Convert a period (in the same time units as dt) to the `s` (scale)
    argument used by _morlet2_kernel/_cwt_morlet below.

    The Morlet wavelet's power is concentrated at angular frequency w0/s
    (in units of radians per *sample*), i.e. an ordinary frequency of
    w0 / (2*pi*s) cycles per sample. A period of P (in real time units)
    corresponds to dt/P cycles per sample, so setting w0/(2*pi*s) = dt/P
    and solving for s gives the relation used here.
    """
    return w0 * period / (2 * np.pi * dt)


def _morlet2_kernel(M: int, s: float, w0: float) -> np.ndarray:
    """A single complex Morlet wavelet kernel of length M at scale s,
    normalized to unit energy. Reimplements the formula previously
    provided by scipy.signal.morlet2 (removed from recent scipy), since
    this module intentionally avoids adding a new third-party dependency
    (e.g. PyWavelets) just for this.
    """
    x = (np.arange(0, M) - (M - 1.0) / 2.0) / s
    wavelet = np.exp(1j * w0 * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    return np.sqrt(1.0 / s) * wavelet


def _cwt_morlet(data: np.ndarray, scales: np.ndarray, w0: float) -> np.ndarray:
    """Continuous wavelet transform of `data` at each scale in `scales`,
    using a complex Morlet wavelet (reimplements the old
    scipy.signal.cwt(data, morlet2, scales, w=w0) behavior via FFT-based
    convolution for speed). Returns a complex array of shape
    (len(scales), len(data)).
    """
    from scipy.signal import fftconvolve

    output = np.empty((len(scales), len(data)), dtype=complex)
    for i, s in enumerate(scales):
        n_kernel = int(min(10 * s, len(data)))
        n_kernel = max(n_kernel, 3)
        kernel = _morlet2_kernel(n_kernel, s, w0)
        output[i] = fftconvolve(data, kernel, mode="same")
    return output


def _fit_and_subtract_gaussian(
    log_periods: np.ndarray,
    residual: np.ndarray,
    fit_half_width_bins: int = 15,
):
    """One iteration of the ROOSTER-style iterative Gaussian peak-picking
    (see guess_wavelet docstring): locate the tallest remaining point in
    `residual`, fit a single Gaussian to a small window around it (in
    log-period space, since periods of interest span more than a decade
    and a Gaussian in linear period space would fit long-period peaks very
    poorly), and return (peak_info_dict, updated_residual). Returns None
    if the fit fails or collapses to a degenerate width.
    """
    from lmfit.models import GaussianModel

    i_peak = int(np.argmax(residual))
    lo = max(i_peak - fit_half_width_bins, 0)
    hi = min(i_peak + fit_half_width_bins + 1, len(residual))
    x = log_periods[lo:hi]
    y = residual[lo:hi]

    model = GaussianModel()
    params = model.guess(y, x=x)
    params["center"].set(value=log_periods[i_peak], min=x[0], max=x[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = model.fit(y, params, x=x)

    amp = out.params["amplitude"].value
    center = out.params["center"].value
    sigma = out.params["sigma"].value
    if not np.isfinite([amp, center, sigma]).all() or sigma <= 0:
        return None

    full_fit = model.eval(out.params, x=log_periods)
    new_residual = residual - full_fit
    height = float(model.eval(out.params, x=np.array([center]))[0])
    peak_info = dict(height=height, center_log_period=center, sigma_log_period=sigma)
    return peak_info, new_residual


def guess_wavelet(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
    n_guesses: int = 5,
    n_periods: int = 200,
    w0: float = 6.0,
    min_peak_snr: float = 3.0,
) -> list:
    """Propose candidate periods from peaks in the Global Wavelet Power
    Spectrum (GWPS) of the light curve -- see Garcia et al. (2014) and the
    wavelet stage of the Santos/ROOSTER rotation pipeline (Breton et al.
    2021), which this closely follows.

    Mechanism, step by step
    ------------------------
    1. Cross-correlate the (mean-subtracted) flux with a complex Morlet
       wavelet at a log-spaced grid of trial periods (via this module's
       own small FFT-based CWT implementation, _cwt_morlet -- recent scipy
       versions removed scipy.signal.cwt/morlet2, so this avoids adding a
       new third-party dependency just for one candidate-generation
       method). Unlike a single Lomb-Scargle periodogram, this keeps the
       *time* axis: the result is a 2D (period, time) power surface, so a
       signal that is only periodic during part of the baseline (e.g.
       before a flare, or before spot evolution scrambles the phase) in
       principle shows up differently than one that's periodic throughout.
       This function only uses the time-averaged projection of that
       surface (the GWPS) for candidate generation -- see
       `info["wavelet_power"]` if you want the full 2D surface for your
       own inspection.

    2. Time-average the 2D power surface over all time samples to get the
       1D Global Wavelet Power Spectrum, GWPS(period). Real, persistent
       periodicities integrate coherently across time and stand out as
       peaks in this projection; transient, non-periodic power does not
       survive the averaging.

    3. Iteratively fit single Gaussians to the GWPS in log-period space
       (see _fit_and_subtract_gaussian): fit the tallest remaining peak,
       subtract that fit, and repeat on the residual. This mirrors what
       ROOSTER's wavelet stage does, and (like guess_pairwise_histogram's
       support-count ranking) is a way of separating out multiple distinct
       candidate periods from one spectrum without just taking every local
       maximum verbatim -- overlapping/nearby peaks get absorbed into a
       single wider Gaussian rather than being double-counted. Iteration
       stops after n_guesses peaks are found, or once the tallest
       remaining peak drops below `min_peak_snr` times a robust noise
       estimate of the (already peak-subtracted) GWPS.

    4. Return each fitted peak as an InitialGuess, `strength` = the
       Gaussian's peak height normalized by the tallest peak found, and
       `info["fitted_gaussians"]` carrying every fitted peak's height,
       center period, and log-period width (a rough analog of the
       G_ACF/H_ACF-style peak-height diagnostics from the composite-
       spectrum literature -- see `composite_spectrum_diagnostics`
       elsewhere in this module for more in that spirit).

    Caveats
    -------
    This function requires `flux` on an evenly-sampled time grid with no
    NaNs (the CWT implementation used here has no gap-awareness). For real
    gappy light curves, either restrict to your longest gap-free stretch,
    or linearly interpolate short gaps before calling this -- interpolating
    is a reasonable approximation for wavelet analysis specifically
    because, unlike the ACF, a short interpolated stretch only locally
    dilutes the time-frequency power there rather than biasing the whole
    lag axis (see acf_utils.py's docstring for why the ACF needs a
    genuinely gap-aware estimator instead).

    Parameters
    ----------
    min_period, max_period : period search range. Defaults to
        [4 * median(dt), (time[-1]-time[0])/2], matching guess_lombscargle.
    n_guesses : maximum number of candidate peaks to extract.
    n_periods : number of log-spaced trial periods spanning
        [min_period, max_period].
    w0 : Morlet wavelet's characteristic (nondimensional) frequency,
        controlling the time/frequency resolution trade-off. Larger w0
        gives sharper period resolution but blurrier time resolution.
        6.0 is a standard default balancing the two.
    min_peak_snr : minimum height (in units of the median absolute
        deviation of the once-peak-subtracted GWPS) for a Gaussian fit to
        be kept as a genuine candidate rather than noise.

    Returns
    -------
    list[InitialGuess], sorted strongest-first (by fitted peak height),
    method="wavelet"
    """
    if not np.isfinite(flux).all():
        raise ValueError(
            "guess_wavelet: flux contains NaNs; the CWT implementation "
            "used here has no gap-awareness. Interpolate short gaps "
            "first, or restrict to a gap-free stretch -- see this "
            "function's docstring."
        )

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    dt = np.median(np.diff(time))
    if not np.allclose(np.diff(time), dt, rtol=1e-3):
        warnings.warn(
            "guess_wavelet: time does not appear evenly spaced; wavelet "
            "period estimates may be unreliable."
        )

    baseline = time[-1] - time[0]
    if min_period is None:
        min_period = 4 * dt
    if max_period is None:
        max_period = baseline / 2
    if max_period <= min_period:
        raise RuntimeError(
            "guess_wavelet: max_period <= min_period; widen the requested "
            "range or check the light curve's baseline."
        )

    periods = np.geomspace(min_period, max_period, n_periods)
    scales = _period_to_morlet_scale(periods, dt, w0)

    y = flux - np.mean(flux)
    wavelet_coeffs = _cwt_morlet(y, scales, w0)
    wavelet_power = np.abs(wavelet_coeffs) ** 2  # shape (n_periods, n_time)

    gwps = np.mean(wavelet_power, axis=1)

    # --- iterative Gaussian peak extraction in log-period space ---
    log_periods = np.log(periods)
    residual = gwps.copy()
    fitted = []
    for _ in range(n_guesses):
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        noise_sigma = 1.4826 * mad if mad > 0 else np.std(residual)
        if noise_sigma == 0 or np.max(residual) < med + min_peak_snr * noise_sigma:
            break
        out = _fit_and_subtract_gaussian(log_periods, residual)
        if out is None:
            break
        peak_info, residual = out
        fitted.append(peak_info)

    if len(fitted) == 0:
        raise RuntimeError(
            "guess_wavelet: no wavelet GWPS peak exceeded min_peak_snr; "
            "try lowering min_peak_snr or widening [min_period, max_period]."
        )

    fitted.sort(key=lambda d: d["height"], reverse=True)
    height_max = fitted[0]["height"]

    guesses = []
    for rank, peak in enumerate(fitted, start=1):
        P_cand = float(np.exp(peak["center_log_period"]))
        guesses.append(InitialGuess(
            P0=P_cand,
            method="wavelet",
            rank=rank,
            strength=float(peak["height"] / height_max) if height_max > 0 else 0.0,
            info=dict(
                periods=periods,
                gwps=gwps,
                wavelet_power=wavelet_power,
                fitted_gaussians=fitted,
                peak_height=peak["height"],
                peak_log_width=peak["sigma_log_period"],
            ),
        ))
    return guesses


def gather_initial_guesses(
    time: np.ndarray,
    flux: np.ndarray,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    methods: tuple = ("pairwise_histogram", "lombscargle", "acf_fft"),
    n_guesses: int = 5,
    method_kwargs: Optional[dict] = None,
) -> tuple:
    """Convenience wrapper: call each requested guess_* function and
    concatenate their candidate lists into one pool, ready to hand to
    fit_rotation_period. Returns (guesses, failed_methods), where
    failed_methods maps method name -> error message for any method that
    raised (e.g. astropy missing, too few ACF peaks found) so one method's
    failure doesn't stop the others from contributing candidates.

    `methods` defaults to the three cheap, always-safe methods. "wavelet"
    (guess_wavelet) is registered and available but deliberately NOT
    included by default -- it requires gap-free flux and is meaningfully
    more expensive than the other three (see this project's relative-speed
    notes), so it only runs if you explicitly ask for it, e.g.
    `methods=("pairwise_histogram", "lombscargle", "acf_fft", "wavelet")`.

    "lombscargle_short", "acf_fft_short", and "acf_fft_highpass" are
    likewise opt-in only: short-period-focused variants of the LS/FFT
    methods (see their docstrings), useful when you have a prior
    expectation that the target rotates fast (say < 15 days) and the
    default full-range candidates are being crowded out by longer-period
    features. "acf_fft_highpass" in particular can return candidates from
    multiple smoothing windows in one call -- see guess_acf_fft_highpass.
    """
    method_kwargs = method_kwargs or {}
    guess_fns = {
        "pairwise_histogram": guess_pairwise_histogram,
        "lombscargle": guess_lombscargle,
        "acf_fft": guess_acf_fft,
        "wavelet": guess_wavelet,
        "lombscargle_short": guess_lombscargle_short,
        "acf_fft_short": guess_acf_fft_short,
        "acf_fft_highpass": guess_acf_fft_highpass,
    }
    guesses = []
    failed = {}
    for method in methods:
        if method not in guess_fns:
            raise ValueError(f"Unknown method '{method}'.")
        kwargs = dict(n_guesses=n_guesses)
        kwargs.update(method_kwargs.get(method, {}))
        try:
            guesses.extend(guess_fns[method](time, flux, acf_lags, acf, **kwargs))
        except Exception as exc:  # noqa: BLE001 -- one method failing shouldn't block the rest
            failed[method] = f"{type(exc).__name__}: {exc}"
    return guesses, failed


# --------------------------------------------------------------------------
# Joint comb fit (single candidate)
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
) -> list:
    """Build one fitting window per harmonic n=0..n_peaks-1, each spanning
    +/- window_frac * P0 around the expected center t0 + n*P0, clipped to
    the available lag range. Windows are frozen once built: point selection
    happens once, before optimization, rather than being re-derived every
    iteration as P and t0 are refined.
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
    windows: list,
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    t0: float,
    P_bounds_frac: float = 0.3,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
):
    """Construct the tied lmfit.Parameters object for the comb model.

    Model per window n:
        model(lag) = h_n - A_n * (lag - center_n)**2
        center_n   = t0 + n*P0 + delta_n      (delta_n small, optional)

    P and t0 are single, shared parameters -- every window's center is
    algebraically tied to them via an lmfit parameter *expression*
    (`expr="t0 + {n}*P"`), which is what enforces "evenly spaced" as a hard
    structural constraint rather than something checked after the fact.
    A_n >= 0 is likewise a hard bound, enforcing "opens downward" (a
    genuine ACF peak, not a trough) for every window independently.
    """
    params = lmfit.Parameters()
    params.add("P", value=P0, min=P0 * (1 - P_bounds_frac), max=P0 * (1 + P_bounds_frac))
    params.add("t0", value=t0, min=t0 - 0.5 * P0, max=t0 + 0.5 * P0)

    for w in windows:
        n = w.n
        lag_sub = acf_lags[w.mask]
        acf_sub = acf[w.mask]

        if allow_jitter and n > 0:
            # delta_n lets window n's center drift a little from the exact
            # comb position, bounded to +/- jitter_frac*P. This tolerates
            # gentle real period drift (e.g. differential rotation, spot
            # evolution) without abandoning the tied structure entirely --
            # every center is still anchored to (P, t0), just with a small
            # per-window correction.
            params.add(f"delta_{n}", value=0.0, min=-jitter_frac * P0, max=jitter_frac * P0)
            params.add(f"center_{n}", expr=f"t0 + {n}*P + delta_{n}")
        else:
            # n=0's center IS t0 by definition; no jitter term needed.
            params.add(f"center_{n}", expr=f"t0 + {n}*P" if n > 0 else "t0")

        # Rough starting values for this window's own shape parameters,
        # read directly off the windowed data (not fit yet, just a
        # reasonable starting point for the optimizer).
        h0_guess = float(np.max(acf_sub))
        half_w = 0.5 * (w.lag_hi - w.lag_lo)
        edge_drop = h0_guess - float(np.min(acf_sub))
        A0_guess = max(edge_drop, 1e-6) / max(half_w**2, 1e-6)

        params.add(f"A_{n}", value=A0_guess, min=0.0)  # min=0 forces downward opening
        params.add(f"h_{n}", value=h0_guess)

    return params


def _comb_residual(
    params,
    windows: list,
    acf_lags: np.ndarray,
    acf: np.ndarray,
) -> np.ndarray:
    """Residual vector for lmfit: for every window, evaluate that window's
    parabola at its own data points and subtract the real ACF values.
    Concatenated across all windows into one flat vector, since lmfit's
    least_squares backend just wants a single 1D array to minimize the
    sum-of-squares of.
    """
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
    windows: list
    lmfit_result: object
    per_peak: dict
    n_peaks_used: int
    n_peaks_dropped: int
    redchi: float
    success: bool


def _fit_single_candidate(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    P0: float,
    t0: float,
    n_peaks: int = 8,
    window_frac: float = 0.25,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
    max_reject_iters: int = 3,
    reject_threshold_sigma: float = 4.0,
    min_peaks_required: int = 2,
) -> CombFitResult:
    """Run the actual joint least-squares comb-of-parabolae fit for ONE
    candidate (P0, t0). This is the expensive, authoritative step that
    fit_rotation_period calls once per candidate; see that function for the
    multi-candidate orchestration and the module docstring for why
    candidate generation and fitting are kept separate.

    What happens here, in order:

    1. Build one fitting window per expected peak (harmonic n = 0, 1, 2,
       ..., n_peaks-1), each a fixed lag range around t0 + n*P0. These
       windows are frozen for the whole fit -- see _build_windows.

    2. Set up the joint parameter set (_build_comb_params): a single shared
       P and t0, with every window's parabola center algebraically tied to
       them (center_n = t0 + n*P [+ small jitter]), each window's own
       curvature/height left free. This is what "evenly spaced" and
       "downward-opening" mean as HARD constraints on the model, rather
       than properties we'd have to check after an unconstrained fit.

    3. Fit with a robust loss (soft_l1 by default, via scipy's
       least_squares under lmfit) rather than plain least-squares, so that
       one badly-behaved window doesn't dominate the fit of the *shared*
       P and t0 that every other window also depends on.

    4. Iteratively drop the worst-fitting window and refit, up to
       `max_reject_iters` times, IF that window's residual RMS is a
       clear outlier (more than `reject_threshold_sigma` robust-sigma above
       the median residual RMS across all currently-fit windows) AND doing
       so wouldn't drop below `min_peaks_required` windows. This is a
       simple RANSAC-style cleanup: real data sometimes has one cycle
       disrupted by a flare, a data gap, or a genuinely anomalous spot
       configuration, and letting that one window silently degrade the fit
       of P and t0 for every other (good) window would be worse than
       excluding it.

    5. Once no more windows are dropped (or the iteration budget runs out),
       summarize each surviving window's fitted center/height/curvature and
       residual RMS into `per_peak`, and package everything into a
       CombFitResult.

    Returns
    -------
    CombFitResult for this single candidate.
    """
    windows = _build_windows(acf_lags, P0, t0, n_peaks, window_frac=window_frac)
    if len(windows) < min_peaks_required:
        raise RuntimeError(
            f"Only {len(windows)} usable peak window(s) built from "
            f"(P0={P0:.4g}, t0={t0:.4g}); need at least {min_peaks_required}."
        )

    n_dropped_total = 0
    result = None

    for iteration in range(max_reject_iters + 1):
        # --- fit with the current set of windows ---
        params = _build_comb_params(
            windows, acf_lags, acf, P0, t0,
            allow_jitter=allow_jitter, jitter_frac=jitter_frac,
        )
        fit_kws = {"loss": loss} if loss != "linear" else {}
        with warnings.catch_warnings():
            # lmfit's stderr/covariance estimate is only approximate under
            # robust losses and can emit a spurious "invalid value in sqrt"
            # warning while computing it; the P/t0 point estimates
            # themselves are unaffected.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = lmfit.minimize(
                _comb_residual,
                params,
                args=(windows, acf_lags, acf),
                method="least_squares",
                **fit_kws,
            )

        # --- compute each window's own residual RMS, to decide what (if
        # anything) is bad enough to drop before the next iteration ---
        per_peak_rms = {}
        for w in windows:
            n = w.n
            lag_sub = acf_lags[w.mask]
            acf_sub = acf[w.mask]
            c = result.params[f"center_{n}"].value
            A = result.params[f"A_{n}"].value
            h = result.params[f"h_{n}"].value
            model = h - A * (lag_sub - c) ** 2
            per_peak_rms[n] = float(np.sqrt(np.mean((model - acf_sub) ** 2)))

        if iteration == max_reject_iters or len(windows) <= min_peaks_required:
            break  # out of iterations, or can't afford to drop any more

        # robust (MAD-based) outlier threshold across the current windows'
        # residual RMS values
        rms_vals = np.array(list(per_peak_rms.values()))
        med = np.median(rms_vals)
        mad = np.median(np.abs(rms_vals - med))
        sigma = 1.4826 * mad if mad > 0 else np.std(rms_vals)
        if sigma == 0:
            break  # everything fits identically well (or there's only 1-2 windows); nothing to reject

        worst_n, worst_rms = max(per_peak_rms.items(), key=lambda kv: kv[1])
        if worst_rms > med + reject_threshold_sigma * sigma and len(windows) > min_peaks_required:
            windows = [w for w in windows if w.n != worst_n]
            n_dropped_total += 1
        else:
            break  # nothing egregious left; stop iterating

    # --- final per-peak summary from the surviving windows' last fit ---
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

    return CombFitResult(
        P=result.params["P"].value,
        P_err=result.params["P"].stderr,
        t0=result.params["t0"].value,
        t0_err=result.params["t0"].stderr,
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

# --------------------------------------------------------------------------
# Phase-dispersion and composite-spectrum diagnostics (optional)
# --------------------------------------------------------------------------
#
# These are model-different cross-checks on a candidate that's already been
# through the joint comb fit -- they don't generate new candidates, and
# they're never computed unless assess_rotation_candidate is explicitly
# given the extra data they need (see that function below). Motivation: the
# joint comb fit and its redchi evaluate a candidate against the ACF's
# *shape* in narrow local windows; both of the diagnostics below instead
# check the candidate directly against the light curve's own phase
# coherence (phase_dispersion_stat) or against the raw ACF's local
# prominence rather than a fitted parabola (acf_peak_prominence_diagnostics)
# -- genuinely different failure modes than height/redchi-based checks can
# catch. See this project's half-period-alias case study (two unequal
# starspot groups) for a worked example where every existing method AND the
# joint fit's own redchi confidently prefer a wrong P/2 answer, and
# phase_dispersion_theta is the one diagnostic that isn't fooled.

def phase_dispersion_stat(
    time: np.ndarray,
    flux: np.ndarray,
    P: float,
    n_bins: int = 10,
) -> float:
    """Phase Dispersion Minimization statistic (Stellingwerf 1978) for a
    candidate period P: fold the light curve on P, bin it in phase, and
    compare the scatter *within* phase bins to the scatter of the whole
    (unfolded) light curve.

        theta = [sum_j (n_j - 1) * s_j^2] / [(N - M) * sigma_total^2]

    where s_j^2 is the variance of the points in phase bin j, n_j is the
    number of points in bin j, N is the total number of points, M is the
    number of non-empty bins, and sigma_total^2 is the variance of the
    full (unfolded) light curve.

    Interpretation: if P is (close to) the true period, folding the light
    curve on it lines up points from different cycles that really do
    belong at the same phase, so each phase bin should look tight compared
    to the light curve's overall scatter -- theta << 1. If P is wrong, the
    fold scrambles unrelated points together in every bin, so each bin's
    scatter approaches the light curve's overall scatter -- theta -> 1 (or
    even slightly above 1, since folding on a bad period doesn't reduce
    variance at all). theta is bounded below by 0 for a perfectly periodic,
    noise-free signal correctly folded.

    This is a genuinely different diagnostic from anything else in this
    module: comb_score and the joint fit's redchi both evaluate a
    candidate against the ACF's *shape*, whereas theta evaluates it
    directly against the light curve's phase coherence, with no ACF or
    parabola model involved at all. A candidate that fits the ACF
    reasonably well but is actually a harmonic of the true period (e.g. 2x
    or 0.5x too long/short) will often show a visibly worse (higher) theta
    than the true period, since folding on the wrong multiple misaligns
    the underlying light curve's repeating shape even when the ACF's comb
    of peaks looks superficially plausible -- making theta a useful,
    cheap, independent cross-check on a joint comb fit's winning candidate
    (or on close runner-ups) before trusting it.

    Parameters
    ----------
    time, flux : the light curve (NaNs are dropped automatically).
    P : candidate period to test, same units as `time`.
    n_bins : number of phase bins in [0, 1) to fold into. Default 10,
        following Stellingwerf (1978)'s original recommendation of order
        ~10 bins for typical sampling; too few bins washes out real
        structure, too many starves each bin of points.

    Returns
    -------
    theta : float. Lower is better; theta ~ 0 indicates strong phase
        coherence at this period, theta ~ 1 indicates no more phase
        coherence than a random period would show.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    if len(flux) < n_bins * 2 or P <= 0:
        return float("nan")

    phase = np.mod(time, P) / P
    bin_idx = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)

    sigma_total_sq = np.var(flux, ddof=1)
    if sigma_total_sq == 0:
        return float("nan")

    numerator = 0.0
    n_nonempty_bins = 0
    for j in range(n_bins):
        in_bin = bin_idx == j
        n_j = int(np.sum(in_bin))
        if n_j < 2:
            continue
        numerator += (n_j - 1) * np.var(flux[in_bin], ddof=1)
        n_nonempty_bins += 1

    dof = len(flux) - n_nonempty_bins
    if dof <= 0:
        return float("nan")

    theta = numerator / (dof * sigma_total_sq)
    return float(theta)


def refine_period_by_pdm(
    time: np.ndarray,
    flux: np.ndarray,
    P0: float,
    n_bins: int = 10,
    search_frac: float = 0.02,
    n_trial: int = 201,
) -> dict:
    """Locally refine a candidate period by minimizing phase_dispersion_stat
    over a small grid of trial periods around P0, and return the best one.

    Why this matters (found empirically while adding this diagnostic):
    theta is a much less forgiving function of period precision than the
    joint comb fit's own P is. The comb fit only ever looks at a handful of
    narrow windows near each expected peak, so a fractional-percent error
    in P barely nudges its redchi. Folding the ENTIRE light curve on P,
    however, accumulates that same fractional error over every cycle in
    the baseline -- for a baseline spanning N_cycles = baseline/P rotations,
    a period error of order 1/N_cycles is already enough to smear a phase
    fold into near-total incoherence (theta -> 1) regardless of whether P0
    was close to correct. Concretely, a light curve with ~20 cycles in its
    baseline needs P known to within roughly 1/20 = 5% just to avoid this,
    and considerably better than that to get a theta clean enough to be
    useful as a discriminating statistic.

    Practically: don't hand fit.P from a CombFitResult straight to
    phase_dispersion_stat and expect a reliable answer if the light curve
    spans many cycles -- refine it first, exactly as this function does.
    assess_rotation_candidate below does this automatically.

    Parameters
    ----------
    time, flux : the light curve.
    P0 : starting-point period to refine (e.g. a CombFitResult.P or an
        InitialGuess.P0).
    n_bins : phase bins for phase_dispersion_stat.
    search_frac : half-width of the search grid, as a fraction of P0 (e.g.
        0.02 searches +/- 2% around P0).
    n_trial : number of trial periods in the search grid.

    Returns
    -------
    dict with 'P_refined' (the trial period achieving the lowest theta) and
    'theta_min' (its theta value).
    """
    trial_periods = np.linspace(P0 * (1 - search_frac), P0 * (1 + search_frac), n_trial)
    thetas = np.array([
        phase_dispersion_stat(time, flux, P, n_bins=n_bins) for P in trial_periods
    ])
    if not np.isfinite(thetas).any():
        return dict(P_refined=float(P0), theta_min=float("nan"))
    i_best = int(np.nanargmin(thetas))
    return dict(P_refined=float(trial_periods[i_best]), theta_min=float(thetas[i_best]))


def acf_peak_prominence_diagnostics(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    peak_lag: float,
    search_frac: float = 0.5,
    P_for_search_window: Optional[float] = None,
) -> dict:
    """Composite-spectrum-style peak height/prominence diagnostics for a
    single ACF peak, following the G_ACF/H_ACF statistics used in the
    Santos/ROOSTER rotation pipeline (see Ceillier et al. 2017): the
    height of the peak itself, and its prominence relative to the two
    local minima flanking it on either side.

    This deliberately mirrors, but is independent of, the joint comb fit:
    _fit_single_candidate's fitted `height` for a window comes from a
    parabola fit local to a narrow window, while G here is read directly
    off the raw ACF at its actual local maximum near `peak_lag` (which may
    drift slightly from the comb's algebraic prediction). Computing both
    gives you two largely independent height estimates for the same
    feature -- if the joint-fit height and the raw ACF's G disagree a lot,
    that's a sign the joint fit's tied structure is dragging the fitted
    parabola away from the ACF's actual local peak, worth a look via
    plot_comb_fit.

    Parameters
    ----------
    acf_lags, acf : the full ACF.
    peak_lag : the lag near which to look for the actual local ACF
        maximum -- typically fit.t0 (for the fundamental) or
        fit.t0 + n*fit.P for harmonic n, taken from a CombFitResult.
    search_frac : how far (as a fraction of P_for_search_window) to look
        on either side of `peak_lag` for the true local maximum and its
        flanking local minima. Default 0.5 (i.e. +/- half a period).
    P_for_search_window : the period defining the search window width. If
        None, defaults to 10% of the ACF's full lag range (a reasonable
        width when you don't have a specific P in hand).

    Returns
    -------
    dict with:
        peak_lag_actual : lag of the true local ACF maximum nearest
            `peak_lag` (may differ slightly from the input).
        G : height of that local maximum (the ACF value there).
        left_min, right_min : ACF values at the local minima immediately
            to the left and right of the peak, within the search window.
        H : mean of (G - left_min) and (G - right_min) -- the peak's
            prominence relative to its immediate surroundings. A tall but
            *unprominent* peak (e.g. one just riding down the ACF's broad
            envelope near lag 0) will have a high G but a low H; H is
            usually the more trustworthy "is this a real, distinct bump"
            indicator of the two.
    """
    if P_for_search_window is None:
        P_for_search_window = 0.1 * (acf_lags[-1] - acf_lags[0])
    half_width = search_frac * P_for_search_window

    lo = peak_lag - half_width
    hi = peak_lag + half_width
    mask = (acf_lags >= lo) & (acf_lags <= hi)
    if mask.sum() < 3:
        return dict(peak_lag_actual=np.nan, G=np.nan, left_min=np.nan,
                     right_min=np.nan, H=np.nan)

    sub_lags = acf_lags[mask]
    sub_acf = acf[mask]
    i_peak = int(np.argmax(sub_acf))
    peak_lag_actual = float(sub_lags[i_peak])
    G = float(sub_acf[i_peak])

    left_vals = sub_acf[:i_peak + 1]
    right_vals = sub_acf[i_peak:]
    left_min = float(np.min(left_vals)) if len(left_vals) > 0 else np.nan
    right_min = float(np.min(right_vals)) if len(right_vals) > 0 else np.nan

    prominences = [G - v for v in (left_min, right_min) if np.isfinite(v)]
    H = float(np.mean(prominences)) if prominences else np.nan

    return dict(peak_lag_actual=peak_lag_actual, G=G, left_min=left_min,
                right_min=right_min, H=H)


def composite_spectrum_diagnostics(
    fit: CombFitResult,
    acf_lags: np.ndarray,
    acf: np.ndarray,
) -> dict:
    """Convenience wrapper: apply acf_peak_prominence_diagnostics to the
    fundamental (n=0 tooth, i.e. fit.t0) of a CombFitResult, giving a
    single G_ACF/H_ACF-style height/prominence summary for the winning (or
    any candidate) fit -- ready to drop straight into a feature vector
    alongside fit.redchi, fit.n_peaks_used, and phase_dispersion_stat for
    an eventual ML selection step (see ROOSTER, Breton et al. 2021, for
    the precedent this is modeled on).

    Returns
    -------
    dict, the output of acf_peak_prominence_diagnostics evaluated at
    fit.t0, with P_for_search_window=fit.P.
    """
    return acf_peak_prominence_diagnostics(
        acf_lags, acf, peak_lag=fit.t0, P_for_search_window=fit.P,
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
    time: Optional[np.ndarray] = None,
    flux: Optional[np.ndarray] = None,
    acf_lags: Optional[np.ndarray] = None,
    n_pdm_bins: int = 10,
) -> dict:
    """Bundle a handful of acceptance diagnostics for a CombFitResult into a
    single dict. Does not make a hard accept/reject call (thresholds are
    target- and noise-regime-dependent) -- returns the ingredients so you
    can set your own cuts, or use fit_rotation_period's built-in gating.

    Two additional, OPTIONAL diagnostics are computed if you supply the
    extra data they need, on top of the original height/curvature-based
    ones (which only ever needed `fit` and `acf`, and still only need
    those):

    - `phase_dispersion_theta` (needs `time` and `flux`): the Stellingwerf
      (1978) PDM statistic, LOCALLY REFINED around fit.P -- see
      refine_period_by_pdm's docstring for why a single evaluation at
      fit.P is not good enough (the joint comb fit's own period precision
      is generally not tight enough for a many-cycle phase fold to stay
      coherent, even when fit.P is essentially correct). This checks phase
      coherence directly in the light curve, independent of the
      ACF/parabola model entirely, so it's a genuinely different failure
      mode than anything the height/redchi-based checks below can catch
      (e.g. it will often flag a harmonic alias that nonetheless produces
      a deceptively clean-looking comb fit). `phase_dispersion_P_refined`
      (the locally-refined period PDM actually settled on) is also
      included, so you can see how far it moved from fit.P.

    - `G_ACF`, `H_ACF` (needs `acf_lags`; reuses `acf`): the composite-
      spectrum-style peak height and flanking-minima prominence of the
      fundamental (fit.t0), from acf_peak_prominence_diagnostics /
      composite_spectrum_diagnostics -- see those docstrings. H_ACF in
      particular is a cheap, model-free prominence check that complements
      `height_snr` below (which uses the *fitted* parabola height and the
      ACF's global standard deviation, not the local flanking minima).

    Both are omitted (left out of the returned dict) if their required
    inputs aren't supplied, so this function's default, minimal call
    signature (just `fit` and `acf`) is unchanged from before -- e.g.
    fit_rotation_period's internal gating calls remain exactly as fast and
    exactly as they were before these two diagnostics existed.
    """
    heights = np.array([p["height"] for p in fit.per_peak.values()])
    curvatures = np.array([p["curvature"] for p in fit.per_peak.values()])
    ns = np.array(list(fit.per_peak.keys()))
    order = np.argsort(ns)
    heights, curvatures, ns = heights[order], curvatures[order], ns[order]

    acf_std = float(np.std(acf))
    height_snr = heights / acf_std if acf_std > 0 else heights * np.nan

    if len(heights) > 1:
        non_increasing = np.diff(heights) <= 1e-3 * np.abs(heights[:-1])
        frac_non_increasing = float(np.mean(non_increasing))
    else:
        frac_non_increasing = np.nan

    frac_positive = float(np.mean(heights > 0)) if len(heights) > 0 else 0.0

    diagnostics = dict(
        n_peaks_used=fit.n_peaks_used,
        n_peaks_dropped=fit.n_peaks_dropped,
        redchi=fit.redchi,
        heights=heights,
        height_snr=height_snr,
        curvatures=curvatures,
        frac_non_increasing_height=frac_non_increasing,
        frac_positive_heights=frac_positive,
        passes_min_peaks=fit.n_peaks_used >= min_peaks,
        passes_redchi=fit.redchi <= max_redchi if np.isfinite(fit.redchi) else False,
        passes_height_snr=bool(np.all(height_snr >= min_height_over_local_std)),
    )

    if time is not None and flux is not None:
        pdm = refine_period_by_pdm(time, flux, fit.P, n_bins=n_pdm_bins)
        diagnostics["phase_dispersion_theta"] = pdm["theta_min"]
        diagnostics["phase_dispersion_P_refined"] = pdm["P_refined"]

    if acf_lags is not None:
        composite = composite_spectrum_diagnostics(fit, acf_lags, acf)
        diagnostics["G_ACF"] = composite["G"]
        diagnostics["H_ACF"] = composite["H"]

    return diagnostics


# --------------------------------------------------------------------------
# Multi-candidate fitting and arbitration
# --------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """The outcome of testing one candidate period all the way through the
    joint comb fit."""
    period: float
    t0: float
    source_guess: InitialGuess
    fit: Optional[CombFitResult]
    diagnostics: Optional[dict]
    passed_gates: bool = False
    error: Optional[str] = None
    # How many raw candidate proposals (across all guess_* methods, before
    # the dedup step below merged near-identical periods into this one
    # representative) supported this same period, and which methods they
    # came from. n_duplicate_guesses=0 and contributing_methods=(source
    # method,) means this candidate was the only one proposing this period
    # -- no cross-method corroboration at the candidate-generation stage.
    # This is computed BEFORE fitting, so don't confuse it with post-fit
    # agreement between different (still-distinct) fitted candidates.
    n_duplicate_guesses: int = 0
    contributing_methods: tuple = ()


@dataclass
class EnsembleResult:
    success: bool
    message: str
    best_fit: Optional[CombFitResult]
    best_guess: Optional[InitialGuess]
    candidates: list        # list[CandidateResult], successfully-fit ones, sorted best-first
    n_candidates_tried: int
    failed_candidates: list  # list[CandidateResult] that errored during fitting


def fit_rotation_period(
    acf_lags: np.ndarray,
    acf: np.ndarray,
    initial_guesses: Union[InitialGuess, list],
    n_peaks: int = 8,
    window_frac: float = 0.25,
    allow_jitter: bool = True,
    jitter_frac: float = 0.05,
    loss: str = "soft_l1",
    max_reject_iters: int = 3,
    reject_threshold_sigma: float = 4.0,
    min_peaks_required: int = 4,
    min_frac_positive_heights: float = 0.8,
    min_mean_height_snr: float = 1.0,
    dedup_rel_tol: float = 0.03,
) -> EnsembleResult:
    """Fit the joint comb model to EVERY candidate in `initial_guesses`,
    and pick whichever one produces the most convincing result -- or
    report that none of them do.

    This is the arbitration stage described in the module docstring: each
    guess_* function only proposes candidates using its own cheap,
    method-specific evidence, and never checks them against the ACF's
    actual shape. This function is where that real check happens, on equal
    footing, for every candidate regardless of which method proposed it.

    Step by step
    ------------
    1. Normalize `initial_guesses` to a list (a single InitialGuess is
       also accepted, for convenience/backward compatibility).

    2. Sanity-filter (period not absurdly short, and long enough baseline
       to fit at least 2 cycles) then deduplicate: candidates within
       `dedup_rel_tol` relative difference of each other are treated as
       the same candidate (only the first encountered, after sorting by
       period, is kept) -- there's no point fitting nearly-identical
       periods twice just because two different methods happened to
       propose them independently.

    3. For each surviving candidate:
         a. If it doesn't already have a t0 (candidate generation doesn't
            compute one -- see module docstring), find a reasonable
            starting phase via a coarse grid search against the ACF
            (_grid_search_t0). This is a cheap heuristic to seed the fit,
            not a fit in itself.
         b. Run the real joint least-squares comb fit at that (P, t0) via
            _fit_single_candidate. This is the expensive, authoritative
            step -- see that function's docstring for what it does.
         c. Compute acceptance diagnostics (assess_rotation_candidate) for
            the result.
       A candidate that can't even be fit (e.g. too few usable windows) is
       recorded with an error message rather than silently dropped, so you
       can see what was tried.

    4. Apply three reliability gates to every successfully-fit candidate.
       A candidate must satisfy ALL three to be considered "passed":
         - n_peaks_used >= min_peaks_required: enough of the expected
           peaks survived the fit (and the RANSAC-style rejection inside
           _fit_single_candidate) to be confident this is a real,
           sustained periodicity rather than a couple of coincidental
           matches.
         - frac_positive_heights >= min_frac_positive_heights: the fitted
           peak heights must mostly be genuine positive bumps. A comb that
           just rides the ACF's smooth decay down through zero near lag 0
           can otherwise achieve a deceptively good (tiny) reduced
           chi-squared without corresponding to any real periodicity at
           all -- this gate catches that.
         - mean(height_snr) >= min_mean_height_snr: the peaks must be tall
           enough relative to the ACF's overall noise level (its standard
           deviation) to be distinguishable from noise fluctuations, on
           average across the surviving peaks.

    5. Among candidates that pass all three gates, pick the one with the
       lowest reduced chi-squared -- the tightest joint fit. Candidates
       are returned sorted this way (passed candidates first, each
       sub-sorted by reduced chi-squared) so you can inspect runner-ups.

    6. If NO candidate passes all three gates, this function does NOT fall
       back to just returning its single best (but not-good-enough)
       attempt as if it were reliable. Instead it returns
       success=False, with a plain-language `message` explaining why, and
       `best_fit`/`best_guess` still populated with the closest attempt
       (clearly documented as unreliable) purely so you can inspect what
       almost worked -- e.g. via the diagnostic plots -- rather than being
       left with nothing to look at.

    Parameters
    ----------
    initial_guesses : an InitialGuess, or (typically) a list of them, e.g.
        the concatenated output of gather_initial_guesses() or of calling
        several guess_* functions yourself.
    n_peaks, window_frac, allow_jitter, jitter_frac, loss,
    max_reject_iters, reject_threshold_sigma : forwarded to
        _fit_single_candidate for every candidate.
    min_peaks_required, min_frac_positive_heights, min_mean_height_snr :
        the three reliability gates described above.
    dedup_rel_tol : relative-difference tolerance for treating two
        candidate periods as duplicates.

    Returns
    -------
    EnsembleResult
    """
    if isinstance(initial_guesses, InitialGuess):
        initial_guesses = [initial_guesses]
    if len(initial_guesses) == 0:
        return EnsembleResult(
            success=False,
            message="No candidate periods were provided to fit_rotation_period.",
            best_fit=None, best_guess=None, candidates=[], n_candidates_tried=0,
            failed_candidates=[],
        )

    dt_acf = np.median(np.diff(acf_lags))
    lag_min, lag_max = acf_lags[0], acf_lags[-1]
    min_lag = 3 * dt_acf

    # --- sanity filter + dedup (by period, ascending) ---
    sane = [
        g for g in initial_guesses
        if g.P0 > 2 * dt_acf and _teeth_count(g.P0, g.t0 or lag_min, lag_max) >= 2
    ]
    sane.sort(key=lambda g: g.P0)
    deduped = []
    merged_groups = []  # parallel to `deduped`: every raw guess merged into it
    for g in sane:
        if not deduped or (g.P0 - deduped[-1].P0) / deduped[-1].P0 > dedup_rel_tol:
            deduped.append(g)
            merged_groups.append([g])
        else:
            # within dedup_rel_tol of the last kept candidate: don't fit it
            # separately, but DO remember it supported this same period --
            # otherwise this information (e.g. "two different methods
            # independently proposed ~this period") is silently lost before
            # it ever reaches feature extraction / the ML ranker.
            merged_groups[-1].append(g)

    if len(deduped) == 0:
        return EnsembleResult(
            success=False,
            message=(
                "No candidate period survived basic sanity checks (period "
                "too short, or too long for at least 2 cycles to fit in "
                "the ACF's lag range)."
            ),
            best_fit=None, best_guess=None, candidates=[],
            n_candidates_tried=0, failed_candidates=[],
        )

    # --- fit every surviving candidate ---
    results = []
    for guess, group in zip(deduped, merged_groups):
        n_dup = len(group) - 1
        contributing_methods = tuple(sorted(set(gg.method for gg in group)))
        t0 = guess.t0 if guess.t0 is not None else _grid_search_t0(
            acf_lags, acf, guess.P0, min_lag=min_lag
        )
        try:
            fit = _fit_single_candidate(
                acf_lags, acf, guess.P0, t0,
                n_peaks=n_peaks, window_frac=window_frac,
                allow_jitter=allow_jitter, jitter_frac=jitter_frac, loss=loss,
                max_reject_iters=max_reject_iters,
                reject_threshold_sigma=reject_threshold_sigma,
                min_peaks_required=min(min_peaks_required, 2),
            )
            diag = assess_rotation_candidate(fit, acf, min_peaks=min_peaks_required)
            passed = (
                fit.n_peaks_used >= min_peaks_required
                and diag["frac_positive_heights"] >= min_frac_positive_heights
                and np.nanmean(diag["height_snr"]) >= min_mean_height_snr
            )
            results.append(CandidateResult(
                period=guess.P0, t0=t0, source_guess=guess, fit=fit,
                diagnostics=diag, passed_gates=bool(passed),
                n_duplicate_guesses=n_dup, contributing_methods=contributing_methods,
            ))
        except Exception as exc:  # noqa: BLE001 -- keep trying other candidates
            results.append(CandidateResult(
                period=guess.P0, t0=t0, source_guess=guess, fit=None,
                diagnostics=None, passed_gates=False,
                error=f"{type(exc).__name__}: {exc}",
                n_duplicate_guesses=n_dup, contributing_methods=contributing_methods,
            ))

    fit_ok = [r for r in results if r.fit is not None]
    failed = [r for r in results if r.fit is None]

    if len(fit_ok) == 0:
        return EnsembleResult(
            success=False,
            message=(
                f"None of the {len(deduped)} candidate period(s) could even "
                "be fit (too few usable windows in every case). Try "
                "widening window_frac or lowering min_peaks_required."
            ),
            best_fit=None, best_guess=None, candidates=[],
            n_candidates_tried=len(deduped), failed_candidates=failed,
        )

    def _sort_key(r):
        redchi = r.fit.redchi if np.isfinite(r.fit.redchi) else np.inf
        return (r.passed_gates, -redchi)

    fit_ok.sort(key=_sort_key, reverse=True)
    passing = [r for r in fit_ok if r.passed_gates]

    if len(passing) == 0:
        best = fit_ok[0]
        return EnsembleResult(
            success=False,
            message=(
                f"Tested {len(deduped)} candidate period(s); none met the "
                f"reliability thresholds (min_peaks_required="
                f"{min_peaks_required}, min_frac_positive_heights="
                f"{min_frac_positive_heights}, min_mean_height_snr="
                f"{min_mean_height_snr}). The closest attempt was P="
                f"{best.fit.P:.4g} (n_peaks_used={best.fit.n_peaks_used}, "
                f"frac_positive_heights={best.diagnostics['frac_positive_heights']:.2f}, "
                f"mean_height_snr={np.nanmean(best.diagnostics['height_snr']):.2f}) "
                "-- attached for inspection, but should NOT be treated as a "
                "reliable rotation period measurement."
            ),
            best_fit=best.fit,
            best_guess=InitialGuess(P0=best.period, t0=best.t0, method="ensemble_best_unreliable"),
            candidates=fit_ok, n_candidates_tried=len(deduped),
            failed_candidates=failed,
        )

    best = passing[0]
    return EnsembleResult(
        success=True,
        message=(
            f"Selected P={best.fit.P:.4g} from {len(passing)} candidate(s) "
            f"that passed reliability gates (of {len(deduped)} tested)."
        ),
        best_fit=best.fit,
        best_guess=InitialGuess(P0=best.period, t0=best.t0, method="ensemble_best"),
        candidates=fit_ok, n_candidates_tried=len(deduped),
        failed_candidates=failed,
    )