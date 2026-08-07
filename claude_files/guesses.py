"""
guesses.py

Stage 1 of the rotation-period pipeline: candidate-period generation.

This module proposes short lists of candidate rotation periods using
several independent, cheap, method-native heuristics -- a discrete list of
ACF peaks (guess_pairwise_histogram), the light curve's own Lomb-Scargle
periodogram (guess_lombscargle), the ACF's own FFT spectrum (guess_acf_fft),
short-period-focused variants of those two (guess_lombscargle_short,
guess_acf_fft_short, guess_acf_fft_highpass), and a Global Wavelet Power
Spectrum (guess_wavelet) -- plus gather_initial_guesses, a convenience
wrapper that runs a chosen subset of them and pools the results.

Critically, NONE of the functions in this module checks its candidates
against the ACF's actual shape, and none of them does any curve fitting.
They are cheap and fast, and their only job is to narrow an enormous search
space (periods from hours to tens of days) down to a short list worth
taking seriously. The real evaluation -- fitting a joint comb-of-parabolae
model to every candidate and arbitrating between them -- is Stage 2, and
lives in comb_fit.py (see fit_rotation_period there). This module's
functions never compute a phase/t0 for exactly that reason: phase-seeding
(_grid_search_t0) and everything downstream of it belongs to the fitting
stage, not candidate generation -- see comb_fit.py.

Why the split matters in practice: earlier versions of this pipeline had
each guess_* function do its own cross-validation against the ACF (a cheap
"comb score") to try to pick a single best candidate internally. That made
each function harder to reason about, and the cheap score turned out to be
gameable (candidates with very few "teeth" in range could win on weak
evidence). Separating "propose candidates" (this module) from "evaluate
candidates properly" (comb_fit.py) removes that failure mode and makes
each piece easier to understand and to test independently.

This module was split out of what used to be a single, much longer
comb_fit.py, specifically to keep each file to a manageable size as the
pipeline grew (wavelet and short-period-focused candidate generation were
later additions). See comb_fit.py's module docstring for the Stage 2 half
of this same design discussion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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
    max_period: float = 50.0,
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
       a resolution knob -- shorter windows filter more aggressively,
       trading away real signal for periods comparable to or longer than
       the window itself in exchange for cleaner isolation of whatever is
       faster. The defaults now deliberately span both regimes: 20 and 40
       days sit well above `max_period=15.0` (safe for anything in the
       target band), while 2, 5, and 10 days are shorter than -- or
       comparable to -- it, aggressively pushing to expose very-short-
       period signal (sub-day to a few days) at real risk of attenuating
       or destroying longer-period signal within the same target band.
       That tradeoff is intentional and is not a problem in practice: this
       function's candidates are only ever a few entries in a much larger
       pool gathered from every method (see gather_initial_guesses), and a
       period this function smooths away is exactly the kind of thing a
       method that never touches the light curve's trend (e.g.
       guess_lombscargle_short, or this same function's own longer
       windows) is left to supply instead. More windows costs relatively
       little: each is one more ACF recomputation on an already-short
       light curve array, not another expensive periodogram search -- see
       this project's timing notes.
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
       (e.g. "acf_fft_hp2d", "acf_fft_hp5d", ..., "acf_fft_hp40d").

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

    Computes the FFT of `data` ONCE (at a fixed size large enough for
    every scale's kernel) and reuses it across all scales, rather than
    calling scipy.signal.fftconvolve per scale -- which would silently
    recompute fft(data) from scratch on every one of the ~200 calls this
    function makes for a typical n_periods grid. That redundant FFT of
    the full (often several-thousand-point) light curve was measured to
    be the majority of guess_wavelet's cost (see this project's
    fitting-performance notes). Each scale's own (much shorter) kernel
    still needs its own FFT, since the kernel genuinely differs per
    scale -- an earlier attempt to also batch all scales' kernel FFTs
    into one call was tried and measured SLOWER, not faster (zero-padding
    every kernel, including the many short ones from small scales, up to
    the single largest kernel's FFT size wastes more than it saves), so
    that part is deliberately left as a per-scale loop.
    """
    from scipy.fft import fft, ifft, next_fast_len

    n_data = len(data)
    kernel_lens = [max(int(min(10 * s, n_data)), 3) for s in scales]
    n_fft = next_fast_len(n_data + max(kernel_lens) - 1)

    data_fft = fft(data, n=n_fft)

    output = np.empty((len(scales), n_data), dtype=complex)
    for i, s in enumerate(scales):
        n_kernel = kernel_lens[i]
        kernel = _morlet2_kernel(n_kernel, s, w0)
        full = ifft(data_fft * fft(kernel, n=n_fft))
        # mode="same" extraction, matching scipy.signal.fftconvolve's
        # convention when the first argument (data) is the longer input:
        # centered window of length n_data out of the length
        # (n_data + n_kernel - 1) linear convolution.
        start = (n_kernel - 1) // 2
        output[i] = full[start:start + n_data]
    return output


def _gaussian(x: np.ndarray, height: float, center: float, sigma: float) -> np.ndarray:
    """height-parameterized Gaussian: height * exp(-(x-center)^2 / (2*sigma^2)).
    Used by _fit_and_subtract_gaussian -- deliberately NOT lmfit's
    GaussianModel (which uses an area/amplitude parameterization requiring
    an extra evaluation step to recover the peak height), so the fit
    directly returns what we actually want.
    """
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


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

    Uses scipy.optimize.curve_fit directly rather than lmfit.GaussianModel:
    this is a tiny, well-conditioned 3-parameter fit on ~30 points, called
    up to n_guesses times per light curve, and profiling found lmfit's
    per-call Parameters/Model machinery was costing more than the actual
    fit here (see this project's fitting-performance notes -- the same
    class of overhead already removed from the main comb fit).
    """
    i_peak = int(np.argmax(residual))
    lo = max(i_peak - fit_half_width_bins, 0)
    hi = min(i_peak + fit_half_width_bins + 1, len(residual))
    x = log_periods[lo:hi]
    y = residual[lo:hi]

    height0 = float(np.max(y))
    center0 = float(log_periods[i_peak])
    sigma0 = max((x[-1] - x[0]) / 4.0, 1e-6)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            popt, _ = curve_fit(
                _gaussian, x, y, p0=[height0, center0, sigma0],
                bounds=([-np.inf, x[0], 1e-8], [np.inf, x[-1], max(x[-1] - x[0], 1e-7)]),
                maxfev=200,
            )
    except RuntimeError:
        return None

    height, center, sigma = (float(v) for v in popt)
    if not np.isfinite([height, center, sigma]).all() or sigma <= 0:
        return None

    full_fit = _gaussian(log_periods, height, center, sigma)
    new_residual = residual - full_fit
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
