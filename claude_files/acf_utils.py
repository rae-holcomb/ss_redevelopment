"""
acf_utils.py

ACF computation that tolerates gaps (NaN-filled missing cadences) on an
otherwise evenly-sampled time grid -- the norm for real TESS/Kepler light
curves, which have data downlink gaps, momentum-dump cuts, and multi-sector
dropouts, but still sit on a fixed-cadence grid once those points are
represented as NaN rather than dropped.

A naive FFT autocorrelation (`ifft(fft(x)*conj(fft(x)))`) breaks completely
in the presence of NaNs: a single NaN propagates through the whole
transform. The fix used here is a *masked* FFT autocorrelation: zero out
missing points (after mean-subtracting only over the valid points), and
separately autocorrelate the validity mask itself to get, at each lag, the
actual number of valid (i, i+k) pairs that contributed to the raw sum. This
gives a correctly gap-normalized ACF in O(N log N), without needing to
interpolate over gaps (which can bias the ACF, especially for the very
large multi-week dropouts common in TESS light curves) or restrict to a
single contiguous segment (which, for gappy data, is often far shorter than
even one rotation period).
"""

from __future__ import annotations

import numpy as np

# Lag range (fraction of baseline) of the ACF handed to the Stage-1 guess_*
# methods. The comb fit (Stage 2) uses compute_acf's default 0.7 B, but the
# candidate generators keep the old B/3 view: on the longer ACF,
# guess_acf_fft's top-N spectral peaks fill up with low-frequency structure
# from the long-lag tail and drop short periods off its list (measured
# 2026-10-13 on synthetic n=200 seed 0: 6/34 stars with P < 1 d lost their
# only correct candidate). Stage 1 already proposes periods up to B/3 from
# the B/3 view (via guess_lombscargle, max_period = B/2), so nothing is lost.
STAGE1_MAX_LAG_FRAC = 1 / 3


class _ExactCache:
    """Tiny most-recently-used cache keyed on the exact bytes of numpy arrays.

    T12 (2026-11-12): the per-star pipeline autocorrelates the same flux twice
    (Stage 2 view, then the Stage 1 view, which only truncates differently)
    and the same gap mask up to seven times (both views plus the five
    high-pass windows of guess_acf_fft_highpass, whose filtered flux keeps the
    original mask almost always). The key is the arrays' raw bytes plus
    dtype/shape, so a hit means bit-identical input and therefore returns
    bit-identical output; nothing is approximated. Values are treated as
    read-only by the callers below.
    """

    def __init__(self, maxsize: int = 2):
        """Keep up to `maxsize` entries, evicting the least recently used."""
        self.maxsize = maxsize
        self._d: dict = {}

    @staticmethod
    def key(*arrays) -> tuple:
        """Exact identity key for a tuple of arrays."""
        return tuple((a.dtype.str, a.shape, a.tobytes()) for a in arrays)

    def get(self, key):
        """Return the cached value or None (and mark it most recently used)."""
        val = self._d.pop(key, None)
        if val is not None:
            self._d[key] = val
        return val

    def put(self, key, val) -> None:
        """Insert `val` under `key`, evicting the oldest entry if full."""
        self._d.pop(key, None)
        self._d[key] = val
        while len(self._d) > self.maxsize:
            self._d.pop(next(iter(self._d)))


_PAIR_COUNT_CACHE = _ExactCache(maxsize=2)
_ACF_CACHE = _ExactCache(maxsize=2)


def _pair_counts(mask: np.ndarray) -> np.ndarray:
    """Valid-pair count at every lag 0..n-1 for a finite-sample mask (cached).

    The autocorrelation of the 0/1 mask via a zero-padded FFT, clipped at 0
    (FFT round-off can leave tiny negative counts). Returned array is shared
    with the cache: do not modify it in place.
    """
    key = _ExactCache.key(mask)
    hit = _PAIR_COUNT_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(mask)
    fm = np.fft.fft(mask.astype(float), n=2 * n)
    valid_count = np.fft.ifft(fm * np.conj(fm)).real[:n]
    # numerical noise can leave tiny negative/zero counts; clip
    valid_count = np.clip(valid_count, 0, None)
    _PAIR_COUNT_CACHE.put(key, valid_count)
    return valid_count


def _full_acf(flux: np.ndarray, mask: np.ndarray) -> tuple:
    """Gap-normalized ACF (normalized to 1 at lag 0) and pair counts at every lag (cached).

    The full-length, untruncated core of compute_acf; see its docstring for
    the masked-FFT method. Returned arrays are shared with the cache: do not
    modify them in place.
    """
    key = _ExactCache.key(flux)
    hit = _ACF_CACHE.get(key)
    if hit is not None:
        return hit
    x = np.where(mask, flux - np.mean(flux[mask]), 0.0)
    n = len(x)
    nfft = 2 * n  # zero-padded to avoid circular wraparound

    fx = np.fft.fft(x, n=nfft)
    raw_acf = np.fft.ifft(fx * np.conj(fx)).real[:n]

    valid_count = _pair_counts(mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        acf = raw_acf / valid_count
    acf[valid_count == 0] = np.nan

    # normalize to acf(0) = 1 (acf[0] equals the variance of the valid data)
    acf = acf / acf[0]
    _ACF_CACHE.put(key, (acf, valid_count))
    return acf, valid_count


def compute_acf(
    time: np.ndarray,
    flux: np.ndarray,
    max_lag_frac: float = 0.7,
    min_valid_frac: float = 0.3,
    return_counts: bool = False,
):
    """Compute a gap-aware ACF of an evenly-sampled light curve that may
    contain NaNs.

    Parameters
    ----------
    time, flux : evenly-sampled arrays (flux may contain NaNs at missing
        cadences; time must not).
    max_lag_frac : upper bound on returned lags, as a fraction of the full
        baseline. Default 0.7 (was 1/3 until 2026-10-13): the pair-coverage
        cut below is the real limit; 0.7 just keeps the extreme tail, where
        only a few contiguous stretches overlap, out of reach. With 1/3 the
        comb fit could never see more than ~B/(3P) peaks, which capped
        recoverable periods at ~B/12 (4 teeth required) -- far too short
        for 1-2 sector TESS stars. Pass 1/3 to reproduce the old behavior.
    min_valid_frac : additionally truncate the returned ACF wherever the
        number of valid (i, i+k) pairs at lag k drops below this fraction
        of the number of valid points at lag 0. Lags beyond this point are
        computed from too few surviving pairs to be trustworthy -- this
        matters a lot more for gappy data than for clean data, since a
        single big dropout (e.g. a multi-week TESS downlink gap) can make
        the *nominal* max_lag_frac cut include lag ranges where almost no
        pairs actually survive, giving a noise-dominated tail that can
        fool peak-finding. For contiguous data the coverage at lag L is
        ~(1 - L/B), so the default 0.3 binds at L ~ 0.7 B.
    return_counts : if True, also return the number of valid pairs at each
        returned lag. The ACF at lag L is a mean over that many products,
        so its noise scales ~1/sqrt(count); comb_fit.fit_rotation_period
        uses these (acf_counts=) to weight its residuals.

    Returns
    -------
    lags, acf : arrays truncated to min(max_lag_frac * baseline, the point
        where valid-pair coverage drops below min_valid_frac).
    counts : (only if return_counts) valid-pair count per returned lag.
    """
    flux = np.asarray(flux, dtype=float)
    time = np.asarray(time, dtype=float)
    mask = np.isfinite(flux)
    if mask.sum() < 10:
        raise ValueError("compute_acf: fewer than 10 finite flux points.")

    acf, valid_count = _full_acf(flux, mask)
    n = len(flux)

    dt = np.median(np.diff(time))
    lags = np.arange(n) * dt

    max_lag = lags[-1] * max_lag_frac
    coverage_ok = valid_count >= min_valid_frac * valid_count[0]
    # find the first index beyond which coverage permanently fails to
    # recover above threshold (walk from the start; a few isolated dips are
    # fine, but once it falls and stays low we stop trusting it)
    good = np.ones(n, dtype=bool)
    below = ~coverage_ok
    if np.any(below):
        # first index where coverage drops below threshold and does not
        # come back up for a sustained stretch: use the first breach as the
        # conservative cutoff
        first_bad = np.argmax(below)
        good[first_bad:] = False

    cutoff_mask = (lags <= max_lag) & good & np.isfinite(acf)
    if cutoff_mask.sum() < 10:
        # fall back: ignore the coverage cut if it's overly aggressive,
        # keep only the max_lag_frac and finiteness cuts
        cutoff_mask = (lags <= max_lag) & np.isfinite(acf)

    if return_counts:
        # counts are integers computed through an FFT; round off float noise
        return lags[cutoff_mask], acf[cutoff_mask], np.rint(valid_count[cutoff_mask])
    return lags[cutoff_mask], acf[cutoff_mask]
