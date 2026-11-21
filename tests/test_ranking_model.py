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


# ----------------------------------------------------------------------------- ranker v1
def _with_negatives(n_neg=10, seed=1):
    """Synthetic rotators plus non-rotator stars (every row relevance 0, is_rotator False)."""
    c, s = _synthetic()
    s["is_rotator"] = True
    c["is_rotator"] = True
    nc, ns = _synthetic(n_stars=n_neg, seed=seed)
    ns["star_id"] = "n" + ns["star_id"]
    nc["star_id"] = "n" + nc["star_id"]
    ns["noise_tic"] += 1000
    nc["relevance"] = 0
    nc["signal"] -= 2.0  # non-rotators: weaker best candidate
    ns["is_rotator"] = False
    nc["is_rotator"] = False
    ns["final_outcome"] = ["wrong"] + ["none"] * (n_neg - 1)
    ns["final_period"] = [2.0] + [np.nan] * (n_neg - 1)
    c = pd.concat([c, nc], ignore_index=True).sort_values(["star_id", "cand_index"]).reset_index(drop=True)
    return c, pd.concat([s, ns], ignore_index=True)


def test_persistent_weak_period_uses_per_bin_cut():
    """A source is persistent only above its P_ref bin's quiet 95th percentile; NaN never is."""
    P = np.array([2.0, 2.0, 15.0, 15.0, np.nan, 30.0])
    snr = np.array([6.0, 6.2, 3.5, 3.7, 50.0, np.nan])
    assert rm.persistent_weak_period(P, snr).tolist() == [False, True, False, True, False, False]


def test_top1_any_marks_negatives_wrong_and_bins():
    """Non-rotator top-1 is 'wrong' (any period); rotators keep classify_period outcomes; top_bin set."""
    c, s = _with_negatives()
    score = np.where(c["relevance"].to_numpy() == 2, 1.0, 0.0)
    t = rm.top1_any(c, score, s)
    assert len(t) == len(s)
    neg = ~t["is_rotator"].astype(bool)
    assert (t.loc[neg, "top_outcome"] == evaluation.WRONG).all()
    assert t.loc[~neg, "top_outcome"].eq("correct").all()
    assert set(t["top_bin"]) <= set(evaluation.PERIOD_BIN_LABELS) | {rm.NONE_BIN}


def test_weighted_curve_counts_negatives_r_per_rotator():
    """With r=1 the negatives weigh as much in total as the rotators do."""
    t = pd.DataFrame(dict(is_rotator=[True, True, False, False, False, False],
                          top_outcome=["correct", "wrong", "wrong", "wrong", "wrong", "wrong"],
                          score_v1=[0.9, 0.5, 0.8, 0.1, 0.1, 0.1]))
    c = rm.weighted_curve(t, 1.0)
    # at threshold 0.8: answered = rotator(0.9, success) + one negative of weight 2/4 = 0.5
    row = c[c.threshold == 0.8].iloc[0]
    assert row.precision == pytest.approx(1 / 1.5) and row.fp_rate == pytest.approx(0.25)
    assert rm.threshold_for_precision_w(t, 0.99, 1.0) == 0.9
    assert rm.threshold_for_precision_w(t, 1.01, 1.0) == np.inf
    assert rm.weighted_curve(t, 0.0).precision.iloc[1] == 1.0  # r=0: negatives ignored


def test_perbin_thresholds_and_apply():
    """Per-bin thresholds are set inside each reported-period bin; 'none' bin is never answered."""
    t = pd.DataFrame(dict(is_rotator=[True] * 6, top_bin=["<1d"] * 3 + [">30d"] * 2 + [rm.NONE_BIN],
                          top_outcome=["correct", "wrong", "correct", "correct", "wrong", "none"],
                          score_v1=[0.9, 0.8, 0.2, 0.3, 0.6, 0.99]))
    thr = rm.perbin_thresholds(t, 0.99, 0.0)
    assert thr["<1d"] == 0.9 and thr[">30d"] == np.inf and thr[rm.NONE_BIN] == np.inf
    ans = rm.apply_thresholds(t, thr)
    assert ans.tolist() == [True, False, False, False, False, False]
    assert rm.apply_thresholds(t, 0.5).tolist() == [True, True, False, False, True, True]
    out = rm.decided_outcomes(t, ans)
    assert out.tolist() == ["correct"] + ["none"] * 5


def test_hand_precision_with_negatives():
    """Hand precision: rotator successes over answered, negatives' answers weighted r per rotator."""
    c, s = _with_negatives(n_neg=10)
    s["final_outcome"] = ["correct"] * 20 + ["wrong"] * 10 + ["none"] * 10 + list(s["final_outcome"].iloc[40:])
    s["final_period"] = 2.0
    assert rm.hand_precision(s, 0.0) == pytest.approx(20 / 30)
    # 1 of 10 negatives answered (FP 0.1) -> weighted 0.1 * 40 rotators = 4 extra answers
    assert rm.hand_precision(s, 1.0) == pytest.approx(20 / 34)


def test_head_features_and_training():
    """Head rows: one per star with its top-1 row's features; the head learns top-1 correctness."""
    c, s = _with_negatives()
    X, cols = rm.feature_matrix(c)
    score = c["signal"].to_numpy()
    H = rm.head_features(c, score, X)
    assert len(H) == s["star_id"].nunique() and H["star_id"].is_unique
    assert (H["h_margin"] >= 0).all() and (H["h_n_rows"] == 6).all()
    top = rm.top1_any(c, score, s)
    y = (rm._success(top.set_index("star_id").loc[H["star_id"], "top_outcome"])
         & top.set_index("star_id").loc[H["star_id"], "is_rotator"].to_numpy(bool)).astype(float)
    m = rm.train_head(H, y, rounds=30, params=dict(nthread=1, subsample=1.0, colsample_bytree=1.0, min_child_weight=0.1))
    p = rm.predict_head([m], H)
    assert p[y == 1].mean() > p[y == 0].mean()


def test_grouped_folds_by_tic_and_side_table():
    """Fold map covers every tic once; side-table join is one-to-one with NaN for missing rows."""
    f = rm.grouped_folds_by_tic([1, 1, 2, 3, 4, 5, 5], n_splits=2)
    assert set(f) == {1, 2, 3, 4, 5} and set(f.values()) <= {0, 1}
    c, _ = _synthetic(n_stars=3)
    side = c[["star_id", "cand_index"]].iloc[:4].assign(half_snr_min=1.0)
    j = rm.join_side_table(c, side)
    assert len(j) == len(c) and j["half_snr_min"].notna().sum() == 4
    assert rm.join_side_table(c, None, ["half_snr_min"])["half_snr_min"].isna().all()
