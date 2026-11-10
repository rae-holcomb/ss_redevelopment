"""T27 non-rotator negatives: manifest selection / pool rules / coverage draws, and the builder's
``--nonrotator`` labelling. Synthetic and toy data only: no drive needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))

import build_feature_tables as bft  # noqa: E402
import build_negatives_manifest as bnm  # noqa: E402
import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import scorecard  # noqa: E402
from synthetic import make_suite  # noqa: E402


def _flags():
    """Toy per-file flags: 6 TICs, two files each, mixed classes and pools."""
    rows = []
    spec = {  # tic: (class, exclude, pools of its two files)
        1: ("quiet", False, ("lt800k", "lt800k")),
        2: ("systematic_common_mode", False, ("lt800k", "lt800k")),
        3: ("quiet", False, ("lt800k", "ge800k")),          # both pools: never a train negative
        4: ("contamination_known_variable", True, ("lt800k", "lt800k")),
        5: ("quiet", False, ("ge800k", "ge800k")),          # dev_medium source
        6: ("quiet", False, ("ge800k", "ge800k")),          # holdout-only source
        7: ("misclassified_star", False, ("ge800k", "ge800k")),
    }
    r = 0
    for tic, (cl, ex, pools) in spec.items():
        for p in pools:
            rows.append(dict(row=r, tic=tic, galaxy=tic, pool=p, classification=cl, exclude_from_training=ex))
            r += 1
    return pd.DataFrame(rows)


def test_select_sources_pool_rules():
    """Train = quiet/SCM, lt800k only, not in dev_medium/holdout; eval = in dev_medium; never holdout-only."""
    fl = _flags()
    src = bnm.select_sources(fl, dev_medium_tics={5, 7}, holdout_tics={5, 6})
    got = dict(zip(src.tic, src.eval_set))
    assert got == {1: "train", 2: "train", 5: "dev_medium"}
    leak = bnm.check_disjoint(src, train_v0_tics={1}, dev_medium_tics={5, 7}, holdout_tics={5, 6})
    assert leak["train_neg_vs_dev_neg"] == 0 and leak["dev_neg_vs_train_v0_rot"] == 0


def test_check_disjoint_raises_on_leak():
    """A dev_medium negative whose TIC is in train_v0 is a leak."""
    src = pd.DataFrame(dict(tic=[1, 5], eval_set=["train", "dev_medium"]))
    with pytest.raises(AssertionError):
        bnm.check_disjoint(src, train_v0_tics={5}, dev_medium_tics={5}, holdout_tics=set())


def test_manifest_tics_by_row():
    """Manifest TICs join on catalog row."""
    assert bnm.manifest_tics(pd.DataFrame(dict(row=[0, 9, 13])), _flags()) == {1, 5, 7}


def _toy_library():
    """A small coverage library covering every class."""
    pats = ["1", "5", "1-2", "3-4", "1-3", "2-6", "1-2-3", "1-3-5-7", "1-2-3-4-5-6", "1-3-5-7-9-11",
            "1-2-3-4-5-6-7-8-9-10-11-12-13"]
    lib = pd.DataFrame(dict(pattern=pats, n_stars=np.arange(1, len(pats) + 1) * 10))
    lib["class"] = [cov.coverage_class(cov.parse_pattern(p)) for p in lib.pattern]
    return lib


def test_draws_feasible_and_balanced():
    """Every drawn pattern lies inside the source's native whole sectors; classes are dealt evenly."""
    lib = _toy_library()
    n = 40
    native = ["1 2 3 4 5 6 7 8 9 10 11 12 13"] * 30 + ["1 2 3 4 5 6"] * 10
    src = pd.DataFrame(dict(tic=np.arange(n), native_whole_sectors=native))
    d = bnm.assign_draw(src, lib, seed=1, set_index=0, draw=0)
    assert len(d) == n and set(d.tic) == set(range(n))
    for t, s in zip(d.tic, d.sectors):
        assert set(cov.parse_pattern(s)) <= set(map(int, native[t].split()))
    counts = d.coverage_class.value_counts()
    assert counts.max() - counts.min() <= 1 + d.coverage_fallback.sum()
    # reproducible
    pd.testing.assert_frame_equal(d, bnm.assign_draw(src, lib, seed=1, set_index=0, draw=0))


def test_build_manifest_ids_and_fullyear():
    """Unique star ids neg-<tic>-d<k> / neg-<tic>-full; full-year rows have empty sectors; paths relative."""
    lib = _toy_library()
    src = pd.DataFrame(dict(tic=[11, 12, 13], galaxy=[1, 2, 3], source_class="quiet", pools="lt800k",
                            n_files=2, eval_set=["train", "train", "dev_medium"],
                            native_sectors="1 2 3 4 5 6 7 8 9 10 11 12 13",
                            native_whole_sectors="1 2 3 4 5 6 7 8 9 10 11 12 13"))
    d = bnm.build_manifest(src, lib, n_draws=2, seed=5)
    assert len(d) == 3 * 3 and d.star_id.is_unique
    assert set(d.star_id[d.noise_tic == 11]) == {"neg-11-d0", "neg-11-d1", "neg-11-full"}
    full = d[d.draw_id == "full"]
    assert (full.sectors == "").all() and (full.coverage_class == "fullyear").all()
    assert (~d.is_rotator).all() and not d.path.str.startswith("/").any()
    assert d.path.iloc[0].endswith("_all.fits")


def test_classify_nonrotator():
    """Non-rotator outcomes: any period is wrong, none is none; rotators unchanged."""
    assert bft.classify(3.0, None, ev.REL_TOL, None) == ev.WRONG
    assert bft.classify(None, None, ev.REL_TOL, None) == ev.NONE
    assert bft.classify(3.0, 3.0, ev.REL_TOL, None) == ev.CORRECT
    assert bft.best_outcome_any([np.nan, 2.0], None, ev.REL_TOL, None) == ev.WRONG
    assert bft.best_outcome_any([], None, ev.REL_TOL, None) == ev.NONE


def test_noise_context_by_tic(tmp_path):
    """Negatives join the flags on noise_tic (first file per TIC)."""
    fl = _flags().assign(periodic=False)
    fl.to_csv(tmp_path / "flags.csv", index=False)
    pd.DataFrame(dict(tic=[2])).to_csv(tmp_path / "coh.csv", index=False)
    man = pd.DataFrame(dict(noise_tic=[2, 1, 2]))
    out = bft.load_noise_context_by_tic(man, tmp_path / "flags.csv", tmp_path / "coh.csv")
    assert list(out.noise_classification) == ["systematic_common_mode", "quiet", "systematic_common_mode"]
    assert list(out.noise_coherent_quiet) == [True, False, True]


@pytest.mark.slow
def test_nonrotator_rows(tmp_path):
    """A star built as a non-rotator: relevance 0 everywhere, no truth, scorecard-identical answer."""
    (tmp_path / "shards").mkdir()
    lc = make_suite(1, seed=3)[0]
    ctx = dict(noise_tic=77, is_rotator=False, label_source="smarts_noise_stack", draw_id="0")
    task = dict(kind="synthetic", lc=lc, star_id="neg-77-d0", context=ctx)
    s = bft.build_one_star(task, str(tmp_path / "shards"), scorecard.DEFAULT_PIPELINE_METHODS, ev.REL_TOL)
    assert s["error"] is None
    bft.combine(tmp_path)
    st = pd.read_csv(tmp_path / "stars.csv.gz").iloc[0]
    ca = pd.read_csv(tmp_path / "candidates.csv.gz")
    assert not st.is_rotator and np.isnan(st.true_period) and st.period_bin == bft.NONROTATOR_BIN
    assert len(ca) > 0 and (ca.relevance == 0).all() and (~ca.is_rotator).all()
    assert set(ca.outcome) <= {ev.WRONG, ev.NONE} and ca.true_period.isna().all()
    assert (ca.label_source == "smarts_noise_stack").all() and ca.noise_tic.eq(77).all()
    rec = scorecard.score_one_star(dict(kind="synthetic", lc=lc, star_id="x"),
                                   scorecard.DEFAULT_PIPELINE_METHODS, ev.REL_TOL)
    if rec["final_period"] is None:
        assert st.final_outcome == ev.NONE and not ca.selected.any()
    else:
        assert st.final_outcome == ev.WRONG
        assert st.final_period == pytest.approx(rec["final_period"], rel=1e-12)
