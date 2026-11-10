"""T10 feature-table builder (tools/build_feature_tables.py): labels, shards, combiner, scorecard parity.

Synthetic light curves only: no drive and no local SMARTS files needed.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))

import build_feature_tables as bft  # noqa: E402
import evaluation as ev  # noqa: E402
import scorecard  # noqa: E402
from synthetic import make_suite  # noqa: E402


def test_relevance_labels():
    """Graded relevance: correct 2, diffrot 1, everything else (aliases included) 0."""
    assert bft.relevance_label(ev.CORRECT) == 2
    assert bft.relevance_label(ev.DIFFROT) == 1
    for o in (ev.HALF, ev.DOUBLE, ev.WRONG, ev.NONE, "noise_period"):
        assert bft.relevance_label(o) == 0


def test_feature_columns_exclude_truth():
    """feature_columns drops identifiers, truth and labels but keeps features."""
    df = pd.DataFrame(columns=["star_id", "noise_tic", "true_period", "relevance", "outcome",
                               "alias_class", "fitted_P", "redchi", "gate_n_peaks_value", "method"])
    assert bft.feature_columns(df) == ["fitted_P", "redchi", "gate_n_peaks_value", "method"]


def test_load_noise_context(tmp_path):
    """Flags join on catalog row; noise_coherent_quiet comes from the TIC list; missing rows -> NaN."""
    flags = pd.DataFrame(dict(row=[1, 2, 3], tic=[10, 20, 30],
                              classification=["quiet", "quiet", "systematic_common_mode"],
                              periodic=[False, False, True], exclude_from_training=[False, False, True]))
    flags.to_csv(tmp_path / "flags.csv", index=False)
    pd.DataFrame(dict(tic=[20])).to_csv(tmp_path / "coh.csv", index=False)
    man = pd.DataFrame(dict(row=[3, 2, 9]))
    out = bft.load_noise_context(man, tmp_path / "flags.csv", tmp_path / "coh.csv")
    assert list(out.noise_tic.iloc[:2]) == [30, 20]
    assert list(out.noise_coherent_quiet) == [False, True, False]
    assert bool(out.noise_exclude_from_training.iloc[0]) is True
    assert np.isnan(out.noise_tic.iloc[2])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build shards for two synthetic stars and combine them."""
    out = tmp_path_factory.mktemp("t10")
    (out / "shards").mkdir()
    suite = make_suite(2, seed=0)
    tasks = [dict(kind="synthetic", lc=lc, star_id=lc.star_id,
                  context=dict(noise_tic=123, noise_coherent_quiet=False)) for lc in suite]
    summaries = [bft.build_one_star(t, str(out / "shards"), scorecard.DEFAULT_PIPELINE_METHODS, ev.REL_TOL)
                 for t in tasks]
    bft.combine(out)
    return out, tasks, summaries


@pytest.mark.slow
def test_shards_and_combiner(built):
    """One shard per star; combined tables have every star, star_id and noise_tic on every row."""
    out, tasks, summaries = built
    assert all(s["error"] is None for s in summaries)
    assert sorted(p.stem for p in (out / "shards").glob("*.pkl")) == sorted(t["star_id"] for t in tasks)
    st = pd.read_csv(out / "stars.csv.gz")
    ca = pd.read_csv(out / "candidates.csv.gz")
    assert len(st) == 2 and set(ca.star_id) <= set(st.star_id)
    assert ca.noise_tic.eq(123).all()
    assert ca.relevance.isin([0, 1, 2]).all()
    assert set(ca.alias_class.fillna("")) <= {"", "half", "double"}
    # every candidate the fit saw is a row: fitted + fit-error + beyond-reach
    for _, s in st.iterrows():
        n = int((ca.star_id == s.star_id).sum())
        assert n == s.n_fit_candidates + s.n_fit_error + s.n_beyond_reach == s.n_rows


@pytest.mark.slow
def test_relevance_matches_outcome(built):
    """relevance is a function of outcome, outcome = classify_period(P_eval)."""
    out, tasks, _ = built
    ca = pd.read_csv(out / "candidates.csv.gz")
    assert (ca.relevance == ca.outcome.map(bft.relevance_label)).all()
    lo, hi = ca.period_lo.to_numpy(), ca.period_hi.to_numpy()
    for i, r in ca.iterrows():
        pr = None if not np.isfinite(lo[i]) else (lo[i], hi[i])
        assert ev.classify_period(r.P_eval, r.true_period, ev.REL_TOL, pr) == r.outcome


@pytest.mark.slow
def test_selected_row_reproduces_scorecard(built):
    """The selected row's fitted period equals scorecard.score_one_star's final answer."""
    out, tasks, _ = built
    ca = pd.read_csv(out / "candidates.csv.gz")
    st = pd.read_csv(out / "stars.csv.gz").set_index("star_id")
    for t in tasks:
        rec = scorecard.score_one_star(t, scorecard.DEFAULT_PIPELINE_METHODS, ev.REL_TOL)
        sel = ca[(ca.star_id == t["star_id"]) & ca.selected]
        assert st.loc[t["star_id"], "final_outcome"] == rec["final_outcome"]
        if rec["final_period"] is None:
            assert len(sel) == 0
        else:
            assert len(sel) == 1
            assert sel.fitted_P.iloc[0] == pytest.approx(rec["final_period"], rel=1e-12)
            assert st.loc[t["star_id"], "final_period"] == pytest.approx(rec["final_period"], rel=1e-12)


@pytest.mark.slow
def test_shard_is_self_contained(built):
    """A shard pickles a star dict and a candidate DataFrame."""
    out, tasks, _ = built
    with open(out / "shards" / f"{tasks[0]['star_id']}.pkl", "rb") as fh:
        d = pickle.load(fh)
    assert d["star"]["star_id"] == tasks[0]["star_id"]
    assert isinstance(d["candidates"], pd.DataFrame)


def test_error_star_still_writes_shard(tmp_path):
    """A star that fails to load writes a shard with an error and no rows (so --resume skips it)."""
    task = dict(kind="fits", path=str(tmp_path / "missing.fits"), star_id="bad", true_period=1.0,
                load_kwargs={}, context={})
    s = bft.build_one_star(task, str(tmp_path), scorecard.DEFAULT_PIPELINE_METHODS, ev.REL_TOL)
    assert s["error"] and s["n_rows"] == 0
    assert (tmp_path / "bad.pkl").exists()


def test_unfit_features_empty_when_all_fit():
    """extract_unfit_candidate_features returns an empty frame when nothing is unfit."""
    import ml_features
    from comb_fit import EnsembleResult
    res = EnsembleResult(success=False, message="", best_fit=None, best_guess=None, candidates=[],
                         n_candidates_tried=0, failed_candidates=[])
    t = np.linspace(0, 27, 2000)
    assert ml_features.extract_unfit_candidate_features(res, [], t, np.sin(t), t[:100], np.ones(100)).empty


def test_combine_refresh_noise(tmp_path):
    """combine(noise=...) overwrites the noise_* columns in both tables and leaves the shards alone."""
    (tmp_path / "shards").mkdir()
    cands = pd.DataFrame(dict(star_id=["a", "a"], relevance=[0, 2], noise_tic=[1, 1],
                              noise_classification=["quiet"] * 2, noise_periodic=[False] * 2,
                              noise_exclude_from_training=[False] * 2, noise_coherent_quiet=[False] * 2))
    star = dict(star_id="a", noise_tic=1, noise_classification="quiet", noise_periodic=False,
                noise_exclude_from_training=False, noise_coherent_quiet=False)
    with open(tmp_path / "shards" / "a.pkl", "wb") as fh:
        pickle.dump({"star": star, "candidates": cands}, fh)
    noise = pd.DataFrame(dict(noise_tic=[1], noise_classification=["periodic_low_amplitude"], noise_periodic=[False],
                              noise_exclude_from_training=[True], noise_coherent_quiet=[True]), index=["a"])
    bft.combine(tmp_path, noise)
    ca, st = pd.read_csv(tmp_path / "candidates.csv.gz"), pd.read_csv(tmp_path / "stars.csv.gz")
    assert ca.noise_exclude_from_training.all() and st.noise_classification.eq("periodic_low_amplitude").all()
    assert list(ca.relevance) == [0, 2]
    with open(tmp_path / "shards" / "a.pkl", "rb") as fh:
        assert pickle.load(fh)["star"]["noise_classification"] == "quiet"
