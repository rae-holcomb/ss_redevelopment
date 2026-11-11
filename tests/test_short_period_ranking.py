"""T5: short-period peaks must survive top-N selection on long, red-noise baselines.

On a full-year TESS-like light curve the Lomb-Scargle periodogram and the ACF's
FFT have ~3-7x more local maxima than on a 1-2 sector one, and red noise makes
power rise toward long periods. Ranking peaks by raw power then fills the
top-10 with long-period noise peaks and drops a real sub-day signal (dev_medium:
true peak at rank ~15-90 full-year). The fix ranks part of the list by power
over the local continuum (rank_by="hybrid"), keeping the strongest raw peaks.
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
from guesses import guess_acf_fft, guess_lombscargle

P_SHORT = 0.3


def red_noise_lc(seed, amp, period=P_SHORT, baseline=350.0, tau=2.0, red=0.002, white=0.001):
    """30-min cadence light curve: sinusoid + AR(1) red noise (correlation time tau d)
    + white noise, with a 1-d gap every 13.7 d (TESS orbits)."""
    rng = np.random.default_rng(seed)
    dt = 1 / 48
    t = np.arange(0, baseline, dt)
    phi = np.exp(-dt / tau)
    r = lfilter([1.0], [1.0, -phi], rng.normal(0, 1, len(t))) * red * np.sqrt(1 - phi**2)
    f = amp * np.sin(2 * np.pi * t / period) + r + rng.normal(0, white, len(t))
    f[(t % 13.7) > 12.7] = np.nan
    return t, f


def _hit(guesses, period=P_SHORT, tol=0.1):
    """True if any guess is within tol (relative) of period."""
    return any(abs(g.P0 / period - 1) < tol for g in guesses)


@pytest.mark.parametrize("seed", range(4))
def test_lombscargle_keeps_weak_short_period_on_full_year(seed):
    """Default ranking puts a weak 0.3 d signal in the top 10 under red noise (old: rank 16-23)."""
    t, f = red_noise_lc(seed, amp=0.0003)
    lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    assert _hit(guess_lombscargle(t, f, lags, acf, n_guesses=10))


@pytest.mark.parametrize("seed", range(4))
def test_acf_fft_keeps_weak_short_period_on_full_year(seed):
    """Same for the ACF's FFT spectrum, at a lower amplitude."""
    t, f = red_noise_lc(seed, amp=0.0002)
    lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    assert _hit(guess_acf_fft(t, f, lags, acf, n_guesses=10))


@pytest.mark.parametrize("fn", [guess_lombscargle, guess_acf_fft])
def test_rank_by_power_is_the_old_top_n(fn):
    """rank_by='power' returns the top-N peaks by raw power, strongest first (pre-T5 behavior)."""
    t, f = red_noise_lc(0, amp=0.0003)
    lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    allg = fn(t, f, lags, acf, n_guesses=100000, rank_by="power")
    top = fn(t, f, lags, acf, n_guesses=10, rank_by="power")
    assert [g.P0 for g in top] == [g.P0 for g in allg[:10]]
    s = [g.strength for g in allg]
    assert s == sorted(s, reverse=True)


@pytest.mark.parametrize("fn", [guess_lombscargle, guess_acf_fft])
def test_hybrid_keeps_strong_long_period_first(fn):
    """A strong 8 d signal stays the rank-1 guess: the raw-power slots come first."""
    t, f = red_noise_lc(1, amp=0.01, period=8.0)
    lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    g = fn(t, f, lags, acf, n_guesses=10)
    assert abs(g[0].P0 / 8.0 - 1) < 0.1
    assert [x.rank for x in g] == list(range(1, len(g) + 1))


@pytest.mark.parametrize("fn", [guess_lombscargle, guess_acf_fft])
def test_hybrid_returns_distinct_peaks_and_respects_n(fn):
    """No peak is returned twice and the list length is min(n_guesses, n_peaks)."""
    t, f = red_noise_lc(2, amp=0.0003)
    lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    for n in (1, 3, 10):
        g = fn(t, f, lags, acf, n_guesses=n)
        assert len(g) == n
        assert len({x.P0 for x in g}) == n
