"""Scorecard v5 (T24 noise_period breakdown of `wrong`, T25 quiet-noise subsets, P22b beyond_reach rows). No drive needed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))

import evaluation as ev  # noqa: E402
import scorecard  # noqa: E402
import build_noise_table  # noqa: E402


def test_noise_period_match_window():
    assert ev.noise_period_match(5.4, 5.0)
    assert not ev.noise_period_match(5.6, 5.0)
    assert not ev.noise_period_match(None, 5.0) and not ev.noise_period_match(5.0, float("nan"))


def test_noise_alias_kind():
    assert ev.noise_alias_kind(2.5, 5.0) == ev.HALF
    assert ev.noise_alias_kind(10.2, 5.0) == ev.DOUBLE
    assert ev.noise_alias_kind(5.0, 5.0) is None and ev.noise_alias_kind(7.0, 5.0) is None


def test_classify_period_unchanged_by_noise_helpers():
    assert ev.classify_period(3.0, 10.0) == ev.WRONG and "noise_period" not in ev.OUTCOMES


def _noise(cls="quiet", periodic=False, exclude=False, coh=False, pref=5.0):
    return dict(tic=1, noise_class=cls, noise_periodic=periodic, noise_exclude=exclude,
                noise_quiet_coherent=coh, noise_P_ref=pref)


def test_noise_columns_only_flag_wrong_answers():
    c = scorecard._noise_columns(_noise(), 5.1, ev.WRONG, 0.1)
    assert c["noise_period_match"] and c["noise_alias"] == ""
    assert not scorecard._noise_columns(_noise(), 5.1, ev.CORRECT, 0.1)["noise_period_match"]
    assert scorecard._noise_columns(_noise(), 2.5, ev.WRONG, 0.1)["noise_alias"] == "half"
    assert scorecard._noise_columns(None, 5.1, ev.WRONG, 0.1) == {}
    assert not scorecard._noise_columns(_noise(pref=float("nan")), 5.1, ev.WRONG, 0.1)["noise_period_match"]


def test_in_noise_subset():
    q, coh, per = _noise(), _noise(coh=True), _noise("contamination_known_variable", periodic=True, exclude=True)
    assert scorecard.in_noise_subset(q, "quiet") and scorecard.in_noise_subset(q, "quiet_strict")
    assert scorecard.in_noise_subset(coh, "quiet") and not scorecard.in_noise_subset(coh, "quiet_strict")
    assert not scorecard.in_noise_subset(per, "quiet") and not scorecard.in_noise_subset(None, "quiet")
    assert scorecard.in_noise_subset(None, "all")


def _per_star():
    rows = []
    def add(cls, outcome, match, coh=False, per=False, exc=False, b="1-3d"):
        rows.append(dict(final_outcome=outcome, period_bin=b, noise_class=cls, noise_periodic=per, noise_exclude=exc,
                         noise_quiet_coherent=coh, noise_period_match=match, noise_alias="", coverage_class="full"))
    add("quiet", ev.CORRECT, False); add("quiet", ev.WRONG, False); add("quiet", ev.WRONG, True, coh=True)
    add("contamination_known_variable", ev.WRONG, True, per=True, exc=True)
    add("contamination_known_variable", ev.NONE, False, per=True, exc=True)
    return pd.DataFrame(rows)


def test_v5_breakdowns_split_wrong_without_changing_it():
    out = scorecard._v5_noise_breakdowns(_per_star())
    o = out["noise"]["overall"]
    assert o["n"] == 5 and o["n_wrong"] == 3 and o["n_wrong_noise_period"] == 2 and o["n_wrong_other"] == 1
    assert o["wrong"] == pytest.approx(o["wrong_noise_period"] + o["wrong_other"])
    assert out["quiet_noise"]["overall"]["n"] == 3 and out["quiet_noise_strict"]["overall"]["n"] == 2
    assert out["periodic_noise"]["overall"]["n"] == 2
    assert out["noise"]["by_noise_class"]["quiet"]["n"] == 3
    assert scorecard._v5_noise_breakdowns(_per_star().drop(columns="noise_class")) == {}


def test_committed_noise_table_matches_manifests():
    t = pd.read_csv(ROOT / "benchmarks" / "noise_sources_dev.csv")
    for name in ("dev_small", "dev_medium"):
        man = pd.read_csv(ROOT / "datasets" / f"{name}.csv")
        assert set(man["sim_id"]) <= set(t["sim_id"])
    assert t["sim_id"].is_unique and t.noise_quiet_coherent.sum() > 0
    assert not set(t.sim_id) & set(pd.read_csv(ROOT / "datasets" / "holdout.csv")["sim_id"]) if "sim_id" in pd.read_csv(ROOT / "datasets" / "holdout.csv").columns else True


def test_build_noise_table_flags_coherent_by_tic():
    flags = pd.DataFrame(dict(sim_id=[1, 2], tic=[10, 20], classification=["quiet", "quiet"], periodic=[False, False],
                              exclude_from_training=[False, False], P_ref=[1.0, 2.0], amp1_ppm_all=[1., 2.],
                              persist=[True, True], near_lmc=[False, False]))
    t = build_noise_table.build(flags, [20], {"in_dev_small": pd.DataFrame(dict(sim_id=[1])),
                                              "in_dev_medium": pd.DataFrame(dict(sim_id=[1, 2]))})
    assert t.set_index("sim_id").noise_quiet_coherent.to_dict() == {1: False, 2: True}
    assert t.set_index("sim_id").in_dev_small.to_dict() == {1: True, 2: False}


def test_candidate_rows_append_beyond_reach_after_v4_rows():
    mk = lambda P: SimpleNamespace(period=P, source_guess=SimpleNamespace(method="acf"), contributing_methods=("acf",),
                                   n_duplicate_guesses=0)
    res = SimpleNamespace(success=False, candidates=[], failed_candidates=[], beyond_reach_candidates=[mk(40.0)])
    lags = np.linspace(0, 10, 11)
    rows = scorecard._candidate_rows(res, lags, np.ones(11), 40.0, 0.1, None)
    assert len(rows) == 1 and rows[0]["beyond_reach"] is True and rows[0]["P0_outcome"] == ev.CORRECT
    res.beyond_reach_candidates = []
    assert scorecard._candidate_rows(res, lags, np.ones(11), 40.0, 0.1, None) == []
