"""Shared pytest fixtures.

Puts the flat `claude_files/` package on sys.path (the modules import each
other by bare name, e.g. `from guesses import ...`), and provides a few
cached synthetic light curves with known periods so tests never need real
data on disk.
"""

import inspect
import sys
from pathlib import Path

# Pin BLAS to one thread *before* numpy is imported (T14): multithreaded BLAS
# reductions change the last bits of fitted periods, which would make the
# bit-identity tests against committed baselines flaky.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

import guesses  # noqa: E402
from acf_utils import compute_acf  # noqa: E402
from synthetic import make_lightcurve  # noqa: E402


def discover_guess_methods() -> dict:
    """Return {method_name: fn} for every guess_* function in guesses.py.

    Introspection rather than a hardcoded list, so a newly added guess_*
    method is automatically covered by the contract tests.
    """
    required = {"time", "flux", "acf_lags", "acf"}
    out = {}
    for name, fn in inspect.getmembers(guesses, inspect.isfunction):
        if name.startswith("guess_") and required.issubset(inspect.signature(fn).parameters):
            out[name[len("guess_"):]] = fn
    return dict(sorted(out.items()))


GUESS_METHODS = discover_guess_methods()


def prep_flux_for(method: str, flux: np.ndarray) -> np.ndarray:
    """Return flux suitable for `method` (wavelet needs gap-free input).

    Mirrors batch_test_guesses._prep_wavelet_flux: linear interpolation
    across gaps.
    """
    if method != "wavelet":
        return flux
    idx = np.arange(len(flux))
    ok = np.isfinite(flux)
    return np.interp(idx, idx[ok], flux[ok])


class LCWithACF:
    """A synthetic light curve bundled with its gap-aware ACF."""

    def __init__(self, lc):
        """Compute and attach the ACF for synthetic light curve `lc`."""
        self.lc = lc
        self.time, self.flux, self.true_period = lc.time, lc.flux, lc.true_period
        self.acf_lags, self.acf = compute_acf(lc.time, lc.flux)


@pytest.fixture(scope="session")
def clean_lc():
    """Strong, clean, 3-day rotator on a 2-sector baseline."""
    return LCWithACF(make_lightcurve(3.0, amplitude=0.01, white_noise=0.001, n_sectors=2, seed=11))


@pytest.fixture(scope="session")
def noise_lc():
    """Pure white noise -- no periodic signal at all."""
    return LCWithACF(make_lightcurve(1.0, amplitude=0.0, white_noise=0.002, n_sectors=2, seed=12))


@pytest.fixture(scope="session")
def ultrashort_lc():
    """P = 0.1 d (~5 cadences per cycle): near the 30-min sampling limit."""
    return LCWithACF(make_lightcurve(0.1, amplitude=0.01, white_noise=0.002, n_sectors=2, seed=13))
