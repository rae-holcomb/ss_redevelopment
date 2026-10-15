"""T18: the T9 alias-discrimination helpers moved from ml_features.py into the
shared module alias_metrics.py (so comb_fit can use them without a circular
import). These tests pin the ML feature values bit-for-bit to what
ml_features produced BEFORE the move (values recorded at main 589cb8d)."""

import numpy as np
import pytest

from acf_utils import compute_acf

# repr() of _alias_discrimination_features at main 589cb8d, before the move
EXPECTED = {
    3.7: dict(acf_height_at_P=0.952139666763179, acf_half_lag_contrast=1.5789903623735082,
              acf_odd_even_contrast=-0.0029838448157132693, fold_half_corr=-0.7333327202391462,
              fold_half_corr_at_2P=0.9986877322794966),
    1.85: dict(acf_height_at_P=-0.6562971128039466, acf_half_lag_contrast=0.011649752972298533,
               acf_odd_even_contrast=-1.6161083006365289, fold_half_corr=-0.9672926313370329,
               fold_half_corr_at_2P=-0.7333327202391462),
    7.4: dict(acf_height_at_P=0.9564923505622837, acf_half_lag_contrast=-0.004112893484461555,
              acf_odd_even_contrast=0.0028513835364045725, fold_half_corr=0.9986877322794966,
              fold_half_corr_at_2P=0.9994581509475848),
    0.41: dict(acf_height_at_P=0.6834548540877584, acf_half_lag_contrast=-0.10894380565537991,
               acf_odd_even_contrast=0.202443629602153, fold_half_corr=-0.2765437674963943,
               fold_half_corr_at_2P=-0.1484302603410743),
}


@pytest.fixture(scope="module")
def double_dip_lc():
    """Deterministic 60-d double-dip light curve (P = 3.7 d) with a gap; no synthetic.py dependency."""
    rng = np.random.default_rng(12345)
    t = np.arange(0, 60, 1 / 48.)
    f = 0.01 * np.sin(2 * np.pi * t / 3.7) + 0.004 * np.sin(4 * np.pi * t / 3.7 + 0.6) + rng.normal(0, 0.002, t.size)
    f[(t > 27) & (t < 28.5)] = np.nan
    lags, acf = compute_acf(t, f)
    return t, f, lags, acf


@pytest.mark.parametrize("P", sorted(EXPECTED))
def test_ml_alias_features_bit_identical_after_move(double_dip_lc, P):
    """ml_features' alias features are unchanged, bit for bit."""
    from ml_features import _alias_discrimination_features
    t, f, lags, acf = double_dip_lc
    got = _alias_discrimination_features(t, f, lags, acf, P)
    assert got == EXPECTED[P]


def test_ml_features_uses_shared_helpers():
    """ml_features re-exports the shared implementations (one copy of the code)."""
    import alias_metrics
    import ml_features
    assert ml_features._acf_harmonic_contrasts is alias_metrics.acf_harmonic_contrasts
    assert ml_features._fold_half_cycle_corr is alias_metrics.fold_half_cycle_corr
    assert ml_features._acf_window_max is alias_metrics.acf_window_max


def test_comb_fit_does_not_import_ml_features():
    """No circular dependency: comb_fit must not import ml_features."""
    import comb_fit
    import inspect
    src = inspect.getsource(comb_fit)
    assert "import ml_features" not in src and "from ml_features" not in src
