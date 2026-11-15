"""P39: complex |c|^2 in guess_acf_fft and ml_features uses the reproducible guesses._abs2."""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

import guesses  # noqa: E402
import ml_features  # noqa: E402


def test_abs2_matches_re2_plus_im2_and_is_close_to_np_abs():
    """_abs2 is exactly re*re + im*im, and within rounding of np.abs(c)**2."""
    rng = np.random.default_rng(0)
    c = rng.normal(size=4097) + 1j * rng.normal(size=4097)
    a = guesses._abs2(c)
    assert np.array_equal(a, c.real * c.real + c.imag * c.imag)
    assert np.allclose(a, np.abs(c) ** 2, rtol=1e-14, atol=0)


def test_no_complex_np_abs_power_left_in_scored_paths():
    """guess_acf_fft and ml_features no longer square a complex np.abs (the SIMD-flip pattern)."""
    for src in (inspect.getsource(guesses.guess_acf_fft), inspect.getsource(ml_features)):
        assert "np.abs(fft_vals) ** 2" not in src
        assert "np.abs(np.fft.rfft" not in src
