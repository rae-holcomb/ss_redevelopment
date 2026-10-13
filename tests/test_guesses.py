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
KNOWN_FAILURES = {
    ("pairwise_histogram", "noise_lc"): "PROJECT_CONTEXT open issue 8: raises when <2 ACF peaks",
    ("pairwise_histogram", "ultrashort_lc"): "PROJECT_CONTEXT open issue 8: raises when <2 ACF peaks",
}


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
