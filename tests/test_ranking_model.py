"""Tests for claude_files/ranking_model.py (T11 ranker v0). Synthetic tables only; no drive."""
import numpy as np
import pandas as pd
import pytest

import evaluation
import ranking_model as rm

xgb = pytest.importorskip("xgboost")


def _synthetic(n_stars=40, n_cand=6, seed=0):
    """Small T10-shaped table: one candidate per star is correct, and `signal` marks it."""
    rng = np.random.default_rng(seed)
    crow, srow = [], []
    for i in range(n_stars):
        P = float(rng.uniform(1, 20))
        good = int(rng.integers(n_cand))
        sid = f"s{i:03d}"
        srow.append(dict(star_id=sid, noise_tic=i // 2, true_period=P, period_lo=np.nan, period_hi=np.nan,
                         period_bin=evaluation.period_bin(P), final_period=np.nan, final_outcome="none",
                         max_relevance=2, band_width_class=["zero", "ge_0p5"][i % 2], searchable=True,
                         eval_pbin=["a", "b"][i % 2], train_order=i, noise_classification="quiet",
                         noise_exclude_from_training=False, noise_coherent_quiet=False))
        for j in range(n_cand):
            Pj = P if j == good else P * float(rng.choice([0.37, 0.71, 1.6, 2.9]))
            crow.append(dict(star_id=sid, noise_tic=i // 2, cand_index=j, P_eval=Pj, period=Pj,
                             relevance=2 if j == good else (1 if j == (good + 1) % n_cand else 0),
                             signal=float(j == good) + 0.1 * rng.normal(), noise=rng.normal(),
                             fit_error=(j == 5), beyond_reach=False, method=rm.PIPELINE_METHODS[j % 3],
                             selected=False, passed_gates=["True", "False"][j % 2],
                             gate_x_threshold=3.0, true_period=P, outcome="x"))
    return pd.DataFrame(crow), pd.DataFrame(srow)


def test_feature_matrix_encodes_and_excludes_truth():
    """Truth/label columns never become features; method is one-hot; booleans become 0/1; constants drop."""
    c, _ = _synthetic()
    X, cols = rm.feature_matrix(c)
    for bad in ("relevance", "true_period", "outcome", "P_eval", "star_id", "noise_tic"):
        assert bad not in cols
    assert "method_is_acf_fft" in cols and "method" not in cols
    assert "gate_x_threshold" not in cols  # constant
    assert set(np.unique(X["passed_gates"])) <= {0.0, 1.0}
    X2, cols2 = rm.feature_matrix(c.drop(columns=["noise"]), cols)
    assert cols2 == cols and X2["noise"].isna().all()


def test_feature_matrix_replaces_inf():
    """Infinite feature values are mapped to a finite sentinel XGBoost accepts."""
    c, _ = _synthetic()
    c.loc[0, "noise"] = np.inf
    X, _ = rm.feature_matrix(c)
    assert np.isfinite(X["noise"]).all()


def test_label_variants():
    """Ablation variants change exactly the rows/queries they describe."""
    c, s = _synthetic()
    y0, w0 = rm.labels_and_weights(c, s, "base")
    assert (y0 == c.relevance).all() and (w0 == 1).all()
    y, _ = rm.labels_and_weights(c, s, "fiterr0")
    assert (y[c.fit_error.values] == 0).all() and (y[~c.fit_error.values] == y0[~c.fit_error.values]).all()
    _, w = rm.labels_and_weights(c, s, "wideband_w05")
    wide = s.band_width_class == "ge_0p5"
    assert (w[s.star_id[wide]] == 0.5).all() and (w[s.star_id[~wide]] == 1).all()
    y, _ = rm.labels_and_weights(c, s, "wideband_rel0")
    on_wide = c.star_id.isin(s.star_id[wide]).values
    assert not ((y == 1) & on_wide).any() and (y[~on_wide] == y0[~on_wide]).all()
    with pytest.raises(ValueError):
        rm.labels_and_weights(c, s, "nope")


def test_grouped_folds_keep_noise_sources_together():
    """No noise_tic (hence no star) appears in two folds."""
    _, s = _synthetic()
    f = rm.grouped_folds(s, 5, 0)
    per_tic = s.assign(fold=s.star_id.map(f)).groupby("noise_tic").fold.nunique()
    assert (per_tic == 1).all() and len(set(f.values())) == 5


def test_stratified_prefix_nested_and_balanced():
    """Subsets keep strata shares and are nested."""
    _, s = _synthetic(n_stars=40)
    a, b = rm.stratified_prefix(s, 10), rm.stratified_prefix(s, 20)
    assert len(a) == 10 and len(b) == 20 and set(a) <= set(b)
    assert s.set_index("star_id").loc[a].eval_pbin.value_counts().tolist() == [5, 5]


def test_training_is_deterministic_and_learns():
    """Same seed -> identical OOF scores; the ranker finds the planted correct candidate."""
    c, s = _synthetic(n_stars=60)
    X, _ = rm.feature_matrix(c)
    y, w = rm.labels_and_weights(c, s)
    p = dict(nthread=1, max_depth=2, eta=0.3)
    r1 = rm.cross_validate(c, s, X, y, w, p, n_rounds=30)
    r2 = rm.cross_validate(c, s, X, y, w, p, n_rounds=30)
    assert np.array_equal(r1.oof, r2.oof)
    top = rm.top1(c, r1.oof, s)
    assert (top.top_outcome == evaluation.CORRECT).mean() > 0.9


def test_thresholds_and_decide():
    """Precision/coverage thresholds pick the right cut on a hand-made top-1 table."""
    top = pd.DataFrame(dict(top_score=[5, 4, 3, 2, 1.0],
                            top_outcome=["correct", "correct", "wrong", "diffrot", "wrong"],
                            period_bin=["<1d"] * 5))
    assert rm.threshold_for_precision(top, 1.0) == 4
    assert rm.threshold_for_precision(top, 0.75) == 2  # 3/4 answered correct at cut 2
    assert rm.threshold_for_precision(top, 1.01) == np.inf
    assert rm.threshold_for_coverage(top, 0.4) == 4
    d = rm.decide(top, 3)
    assert d.tolist() == ["correct", "correct", "wrong", "none", "none"]
    summ = rm.outcome_summary(d, top.period_bin).set_index("bin")
    assert summ.loc["all", "success"] == pytest.approx(0.4) and summ.loc["all", "declined"] == pytest.approx(0.4)


def test_feature_family_and_method():
    """Spectrum prefixes map to their guess_* method, longest prefix first."""
    assert rm.feature_method("fft_highpass_nearest_peak_rank") == "acf_fft_highpass"
    assert rm.feature_method("fft_nearest_peak_rank") == "acf_fft"
    assert rm.feature_method("ls_short_nearest_peak_snr") == "lombscargle_short"
    assert rm.feature_method("hist_nearest_bin_rank") == "pairwise_histogram"
    assert rm.feature_family("gate_half_lag_consistency_value") == "gates"
    assert rm.feature_family("P_over_reach") == "coverage_reach"
    assert rm.feature_family("redchi") == "comb_fit"
