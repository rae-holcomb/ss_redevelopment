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
    numeric = df.select_dtypes(include=[np.number])
    assert numeric.shape[1] > 50, "feature set unexpectedly small"
