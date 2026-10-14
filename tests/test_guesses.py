"""Contract and recovery tests for every guess_* candidate method.

Methods are discovered by introspection (see conftest.discover_guess_methods),
so a new guess_* function is automatically held to the same contract.
"""

import math

import pytest

from conftest import GUESS_METHODS, prep_flux_for
from evaluation import CORRECT, best_outcome
from guesses import InitialGuess

METHODS = list(GUESS_METHODS)

# Known open bugs, keyed by (method, fixture). strict=True means that when
# the bug is fixed the test "unexpectedly passes" and FAILS, forcing whoever
# fixed it to delete the entry here -- so this list can't silently go stale.
KNOWN_FAILURES = {}


def _run(method, lcx, n_guesses=5):
    """Call one guess method on a fixture bundle."""
    fn = GUESS_METHODS[method]
    return fn(lcx.time, prep_flux_for(method, lcx.flux), lcx.acf_lags, lcx.acf, n_guesses=n_guesses)


def test_discovery_finds_core_methods():
    """Introspection still finds the three default methods."""
    assert {"pairwise_histogram", "lombscargle", "acf_fft"} <= set(METHODS)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fixture", ["clean_lc", "noise_lc", "ultrashort_lc"])
def test_returns_well_formed_guesses(method, fixture, request):
    """Every method returns list[InitialGuess] with finite positive P0 and a
    method tag, on any input -- including pure noise and near-Nyquist
    periods. Returning [] is fine; raising is not (a batch run over 1M
    stars shouldn't depend on try/except at every call site)."""
    if (method, fixture) in KNOWN_FAILURES:
        request.applymarker(pytest.mark.xfail(reason=KNOWN_FAILURES[(method, fixture)], strict=True))
    out = _run(method, request.getfixturevalue(fixture))
    assert isinstance(out, list)
    for g in out:
        assert isinstance(g, InitialGuess)
        assert math.isfinite(g.P0) and g.P0 > 0
        assert isinstance(g.method, str) and g.method
        assert g.t0 is None, "Stage 1 must never compute a phase (see guesses.py docstring)"


@pytest.mark.parametrize("method", METHODS)
def test_recovers_clean_period_in_top_n(method, clean_lc):
    """On a strong, clean 3-day rotator every method should put the true
    period in its top-10 list. This is the minimum bar for a candidate
    generator: it doesn't need to rank P first, only not to miss it."""
    out = _run(method, clean_lc, n_guesses=10)
    outcome, rank = best_outcome([g.P0 for g in out], clean_lc.true_period)
    assert outcome == CORRECT, f"{method}: best outcome {outcome}, candidates {[round(g.P0, 3) for g in out]}"


def test_pairwise_histogram_degenerate_inputs_return_empty(clean_lc):
    """guess_pairwise_histogram returns [] (never raises) on every early
    exit: too few lags, an empty period window, and a flat ACF with no
    peaks (PROJECT_CONTEXT open issue 8)."""
    import numpy as np
    from guesses import guess_pairwise_histogram

    t, f, lags, acf = clean_lc.time, clean_lc.flux, clean_lc.acf_lags, clean_lc.acf
    assert guess_pairwise_histogram(t, f, lags[:2], acf[:2]) == []
    assert guess_pairwise_histogram(t, f, lags, acf, min_period=1e5, max_period=2e5) == []
    assert guess_pairwise_histogram(t, f, lags, np.zeros_like(acf)) == []


def test_pairwise_histogram_two_peaks_with_explicit_bins():
    """Exactly two ACF peaks give a single pairwise spacing; with an explicit
    n_hist_bins that used to mean zero-width bins. It should still return
    the spacing as a candidate."""
    import numpy as np
    from guesses import guess_pairwise_histogram

    lags = np.arange(0, 10, 0.01)
    acf = np.exp(-0.5 * ((lags - 3) / 0.2) ** 2) + np.exp(-0.5 * ((lags - 6) / 0.2) ** 2)
    out = guess_pairwise_histogram(lags, lags, lags, acf, n_hist_bins=50)
    assert out and abs(out[0].P0 - 3.0) < 0.1


# --------------------------------------------------------------------------
# T15: every guess_* returns [] (never raises) on degenerate input
# --------------------------------------------------------------------------

def _degenerate_bundles(lcx):
    """Degenerate (time, flux, acf_lags, acf) bundles built from fixture `lcx`.

    Each bundle is degenerate in *both* the light curve and the ACF, so every
    method -- whichever input it reads -- has nothing to propose.
    """
    import numpy as np

    t, lags = lcx.time, lcx.acf_lags
    dt = float(np.median(np.diff(t)))
    t_even = t[0] + dt * np.arange(len(t))  # gap-free grid (wavelet's precondition)
    return {
        "all_nan": (t, np.full_like(t, np.nan), lags, np.full_like(lags, np.nan)),
        "constant": (t_even, np.ones_like(t_even), lags, np.zeros_like(lags)),
        "too_few_points": (t_even[:5], np.ones(5) + 1e-3 * np.arange(5), lags[:2], lcx.acf[:2]),
        "single_point": (t_even[:1], np.ones(1), lags[:1], lcx.acf[:1]),
        "empty": (t[:0], t[:0], lags[:0], lags[:0]),
    }


DEGENERATE_CASES = ["all_nan", "constant", "too_few_points", "single_point", "empty"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("case", DEGENERATE_CASES)
def test_degenerate_input_returns_empty(method, case, clean_lc):
    """All-NaN, constant, too-short or empty input: every method returns []
    rather than raising or proposing meaningless candidates (T15)."""
    t, f, lags, acf = _degenerate_bundles(clean_lc)[case]
    assert GUESS_METHODS[method](t, f, lags, acf) == []


def _run_kw(method, lcx, **kwargs):
    """Call one guess method on a fixture bundle with extra kwargs."""
    fn = GUESS_METHODS[method]
    return fn(lcx.time, prep_flux_for(method, lcx.flux), lcx.acf_lags, lcx.acf, **kwargs)


@pytest.mark.parametrize("method", METHODS)
def test_inverted_period_window_returns_empty(method, clean_lc):
    """min_period > max_period is an empty search window -> [] (T15)."""
    out = _run_kw(method, clean_lc, min_period=10.0, max_period=1.0)
    assert out == []


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("window", [(0.0, 1.0), (1e5, 2e5), (float("nan"), 5.0)])
def test_odd_period_windows_never_raise(method, window, clean_lc):
    """A zero/NaN lower edge or a window far beyond the baseline must not
    raise; any candidates returned must still be well formed (T15)."""
    out = _run_kw(method, clean_lc, min_period=window[0], max_period=window[1])
    assert isinstance(out, list)
    for g in out:
        assert math.isfinite(g.P0) and g.P0 > 0 and math.isfinite(g.strength)


def test_lombscargle_too_few_finite_points_returns_empty(clean_lc):
    """Fewer than 10 finite (time, flux) points -> [] (was RuntimeError)."""
    import numpy as np
    from guesses import guess_lombscargle

    f = np.full_like(clean_lc.flux, np.nan)
    f[:9] = 1.0 + 1e-3 * np.arange(9)
    assert guess_lombscargle(clean_lc.time, f, clean_lc.acf_lags, clean_lc.acf) == []


def test_acf_fft_too_few_frequency_bins_returns_empty(clean_lc):
    """A period window holding < 3 FFT frequency bins -> [] (was RuntimeError)."""
    from guesses import guess_acf_fft

    lc = clean_lc
    assert guess_acf_fft(lc.time, lc.flux, lc.acf_lags, lc.acf, min_period=1e5, max_period=2e5) == []


def test_acf_fft_highpass_too_few_finite_points_returns_empty(clean_lc):
    """compute_acf needs >= 10 finite points; the high-pass method must not
    let its ValueError escape (T15)."""
    import numpy as np
    from guesses import guess_acf_fft_highpass

    f = np.full_like(clean_lc.flux, np.nan)
    f[:9] = 1.0
    assert guess_acf_fft_highpass(clean_lc.time, f, clean_lc.acf_lags, clean_lc.acf) == []


def test_wavelet_nan_flux_returns_empty(clean_lc):
    """Gappy (NaN-containing) flux violates wavelet's precondition -> []
    (was ValueError)."""
    from guesses import guess_wavelet

    lc = clean_lc
    assert guess_wavelet(lc.time, lc.flux, lc.acf_lags, lc.acf) == []


def test_wavelet_no_peak_above_snr_returns_empty(clean_lc):
    """No GWPS peak above min_peak_snr -> [] (was RuntimeError)."""
    from guesses import guess_wavelet

    lc = clean_lc
    f = prep_flux_for("wavelet", lc.flux)
    assert guess_wavelet(lc.time, f, lc.acf_lags, lc.acf, min_peak_snr=1e9) == []


def test_wavelet_short_baseline_returns_empty(clean_lc):
    """A baseline so short that max_period <= min_period -> [] (was RuntimeError)."""
    from guesses import guess_wavelet

    lc = clean_lc
    f = prep_flux_for("wavelet", lc.flux)
    assert guess_wavelet(lc.time[:6], f[:6], lc.acf_lags, lc.acf) == []
