"""Tests for ml_features.extract_candidate_features."""

import numpy as np
import pytest

from comb_fit import fit_rotation_period
from guesses import gather_initial_guesses
from ml_features import extract_candidate_features


@pytest.mark.slow
def test_feature_table_shape_and_label(clean_lc):
    """One row per fitted candidate, unique columns, star_id attached, and
    exactly the candidates near the truth labelled 1."""
    lcx = clean_lc
    g, _ = gather_initial_guesses(lcx.time, lcx.flux, lcx.acf_lags, lcx.acf, n_guesses=10)
    res = fit_rotation_period(lcx.acf_lags, lcx.acf, g)
    df = extract_candidate_features(res, g, lcx.time, lcx.flux, lcx.acf_lags, lcx.acf,
                                    star_id="t", true_period=lcx.true_period)
    assert len(df) == len(res.candidates)
    assert df.columns.is_unique
    assert (df["star_id"] == "t").all()
    assert set(df["label"].unique()) <= {0, 1}
    assert df["label"].sum() >= 1
    new_cols = ["acf_height_at_P", "acf_half_lag_contrast", "acf_odd_even_contrast",
                "fold_half_corr", "fold_half_corr_at_2P"]
    assert set(new_cols) <= set(df.columns)
    # the candidate nearest the truth should not look like a 2P alias
    best = df.loc[(df["fitted_P"] - lcx.true_period).abs().idxmin()]
    assert best["fold_half_corr"] < 0.5
    numeric = df.select_dtypes(include=[np.number])
    assert numeric.shape[1] > 50, "feature set unexpectedly small"


# --------------------------------------------------------------------------
# T9 alias-discrimination features (fast: no fitting needed)
# --------------------------------------------------------------------------

from ml_features import _acf_harmonic_contrasts, _fold_half_cycle_corr  # noqa: E402


def test_acf_harmonic_contrasts_flag_half_and_double_aliases(clean_lc):
    """At the true period the half-lag contrast is large; at 2P it collapses
    (half-lags are real peaks); at P/2 the odd/even contrast goes strongly
    negative (odd multiples fall on troughs/secondary peaks)."""
    lcx, P = clean_lc, clean_lc.true_period
    at_p = _acf_harmonic_contrasts(lcx.acf_lags, lcx.acf, P)
    at_2p = _acf_harmonic_contrasts(lcx.acf_lags, lcx.acf, 2 * P)
    at_half = _acf_harmonic_contrasts(lcx.acf_lags, lcx.acf, P / 2)
    assert at_p["acf_half_lag_contrast"] > 0.5
    assert at_p["acf_height_at_P"] > 0.5
    assert at_2p["acf_half_lag_contrast"] < 0.25 * at_p["acf_half_lag_contrast"]
    assert at_half["acf_odd_even_contrast"] < -0.5
    assert abs(at_p["acf_odd_even_contrast"]) < 0.2


def test_acf_harmonic_contrasts_nan_outside_lag_range(clean_lc):
    """A period whose first multiple is beyond the ACF's lag range gives NaN, not an error."""
    lcx = clean_lc
    out = _acf_harmonic_contrasts(lcx.acf_lags, lcx.acf, 2.0 * lcx.acf_lags[-1])
    assert np.isnan(out["acf_half_lag_contrast"]) and np.isnan(out["acf_odd_even_contrast"])
    out = _acf_harmonic_contrasts(lcx.acf_lags, lcx.acf, np.nan)
    assert np.isnan(out["acf_half_lag_contrast"])


def test_fold_half_cycle_corr_detects_double_period(clean_lc):
    """Folding at 2P repeats the same rotation twice (corr ~ 1); folding at
    the true P compares dip with non-dip (much lower)."""
    lcx, P = clean_lc, clean_lc.true_period
    assert _fold_half_cycle_corr(lcx.time, lcx.flux, 2 * P) > 0.9
    assert _fold_half_cycle_corr(lcx.time, lcx.flux, P) < 0.5
    # too long to fold even 2 cycles into one segment of the 2-sector baseline
    assert np.isnan(_fold_half_cycle_corr(lcx.time, lcx.flux, 100.0))
