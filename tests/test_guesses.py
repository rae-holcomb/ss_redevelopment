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
