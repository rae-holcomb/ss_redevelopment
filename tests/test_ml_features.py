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
                "fold_half_corr", "fold_half_corr_at_2P",
                # T16e coverage context
                "n_sectors_observed", "n_contiguous_segments", "longest_contiguous_days",
                "largest_gap_days", "P_over_longest_segment", "n_cycles_observed_total",
                "tess_timescale_rel_dist"]
    assert set(new_cols) <= set(df.columns)
    # the candidate nearest the truth should not look like a 2P alias
    best = df.loc[(df["fitted_P"] - lcx.true_period).abs().idxmin()]
    assert best["fold_half_corr"] < 0.5
    # 2 consecutive sectors: one contiguous stretch, so the T16e features
    # reduce to the existing baseline features (documented relationship)
    assert (df["n_sectors_observed"] == 2).all() and (df["n_contiguous_segments"] == 1).all()
    np.testing.assert_allclose(df["longest_contiguous_days"], df["lc_baseline_days"])
    np.testing.assert_allclose(df["P_over_longest_segment"], df["P_over_lc_baseline"])
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


# --------------------------------------------------------------------------
# T16e coverage-context features (fast: no fitting needed)
# --------------------------------------------------------------------------

from ml_features import (  # noqa: E402
    _candidate_coverage_features, _star_coverage_features, _tess_timescale_rel_dist,
)
from synthetic import make_lightcurve  # noqa: E402


def _coverage_feats(time, flux, periods=(0.3, 3.0, 13.7, 30.0)):
    """Star-level coverage features plus the per-candidate ones at several periods."""
    star, seg = _star_coverage_features(time, flux)
    out = dict(star)
    for P in periods:
        out.update({f"{k}@{P}": v for k, v in _candidate_coverage_features(P, star, seg).items()})
    return out


@pytest.mark.parametrize("pattern", [[5], [3, 4], [1, 13], [2, 4, 5, 9], [1, 2, 3, 4, 5, 6, 8]])
def test_coverage_features_depend_only_on_time_sampling(pattern):
    """No leakage: the coverage features are a function of (time, finite-flux
    mask) alone. Replacing every flux value (keeping the NaN mask), shifting
    the time origin, or generating a different star with the same coverage
    gives identical values; nothing reads the `sectors` label."""
    lc = make_lightcurve(2.0, amplitude=0.01, white_noise=0.002, n_sectors=13, sectors=pattern, seed=3)
    ref = _coverage_feats(lc.time, lc.flux)

    rng = np.random.default_rng(0)
    scrambled = np.where(np.isfinite(lc.flux), rng.normal(5.0, 3.0, len(lc.flux)), np.nan)
    other = make_lightcurve(17.0, amplitude=0.05, white_noise=0.01, n_sectors=13, sectors=pattern, seed=99)
    for t, f in [(lc.time, scrambled), (lc.time + 1234.5, lc.flux), (other.time, other.flux)]:
        got = _coverage_feats(t, f)
        assert got.keys() == ref.keys()
        for k in ref:
            assert got[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-9), k
    assert ref["n_sectors_observed"] == len(pattern)


def test_coverage_features_follow_the_mask_not_the_label():
    """Blanking a sector of a full-year light curve changes the features
    exactly as if that sector had never been observed (no label consulted)."""
    full = make_lightcurve(2.0, amplitude=0.01, white_noise=0.002, n_sectors=13, seed=4)
    star_full, _ = _star_coverage_features(full.time, full.flux)
    assert star_full["n_sectors_observed"] == 13 and star_full["n_contiguous_segments"] == 1
    f = full.flux.copy()
    f[(full.time >= 6 * 27.4) & (full.time < 7 * 27.4)] = np.nan  # drop sector 7
    star, seg = _star_coverage_features(full.time, f)
    assert star["n_sectors_observed"] == 12 and star["n_contiguous_segments"] == 2
    assert star["largest_gap_days"] == pytest.approx(27.4 + 1.0, abs=0.1)
    assert star["longest_contiguous_days"] == pytest.approx(seg.max())


def test_candidate_coverage_feature_values():
    """Toy segments (10 d and 25 d, with a big gap): exact per-candidate values."""
    t = np.r_[np.arange(0, 10.0001, 0.5), np.arange(40, 65.0001, 0.5)]
    f = np.ones_like(t)
    star, seg = _star_coverage_features(t, f)
    np.testing.assert_allclose(seg, [10.0, 25.0])
    assert star["n_contiguous_segments"] == 2 and star["largest_gap_days"] == pytest.approx(30.0)
    c = _candidate_coverage_features(4.0, star, seg)
    assert c["P_over_longest_segment"] == pytest.approx(4.0 / 25.0)
    assert c["n_cycles_observed_total"] == 2 + 6  # floor(10/4) + floor(25/4)
    c = _candidate_coverage_features(30.0, star, seg)
    assert c["n_cycles_observed_total"] == 0 and c["P_over_longest_segment"] > 1
    c = _candidate_coverage_features(np.nan, star, seg)
    assert np.isnan(c["P_over_longest_segment"]) and np.isnan(c["tess_timescale_rel_dist"])


def test_tess_timescale_rel_dist():
    """Zero on the orbit ladder (13.7 d x 1/4..4), relative distance elsewhere."""
    for P in (13.7, 27.4, 6.85, 13.7 / 3, 41.1, 54.8):
        assert _tess_timescale_rel_dist(P) == pytest.approx(0.0, abs=1e-12)
    assert _tess_timescale_rel_dist(14.385) == pytest.approx(0.05)
    assert _tess_timescale_rel_dist(20.0) == pytest.approx(min(abs(20 / 13.7 - 1), abs(20 / 27.4 - 1)))
    assert np.isnan(_tess_timescale_rel_dist(-1.0))


def test_coverage_features_degenerate_inputs():
    """All-NaN or single-point light curves give zeros/NaN, never an error."""
    star, seg = _star_coverage_features(np.arange(10.0), np.full(10, np.nan))
    assert star["n_contiguous_segments"] == 0 and star["n_sectors_observed"] == 0
    assert np.isnan(star["longest_contiguous_days"])
    c = _candidate_coverage_features(3.0, star, seg)
    assert np.isnan(c["P_over_longest_segment"]) and c["n_cycles_observed_total"] == 0
    # one finite point = one zero-length segment
    star, seg = _star_coverage_features(np.arange(10.0), np.r_[1.0, np.full(9, np.nan)])
    assert star["n_contiguous_segments"] == 1 and star["longest_contiguous_days"] == 0.0
    c = _candidate_coverage_features(3.0, star, seg)
    assert np.isnan(c["P_over_longest_segment"]) and c["n_cycles_observed_total"] == 0
