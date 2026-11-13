"""Tests for benchmarks/check_goals.py (goal grading and wildcard expansion)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import check_goals  # noqa: E402

METRICS = {
    "config": {"dataset": "d", "git_commit": "x"},
    "final": {"by_bin": {"<1d": {"success": 0.7}, "1-3d": {"success": 0.9}}},
    "methods": {"a": {"overall": {"error_rate": 0.0}}, "b": {"overall": {"error_rate": 0.1}}},
}


def test_get_path_handles_dotted_bin_labels():
    """Greedy key matching resolves paths whose keys contain no dots, and misses return None."""
    assert check_goals.get_path(METRICS, "final.by_bin.<1d.success") == 0.7
    assert check_goals.get_path(METRICS, "final.by_bin.nope.success") is None


def test_expand_wildcard():
    """'*' expands to every numeric leaf at that level."""
    assert sorted(check_goals.expand(METRICS, "methods.*.overall.error_rate")) == [
        "methods.a.overall.error_rate", "methods.b.overall.error_rate"]


def _run(tmp_path, goals):
    """Write metrics/goals to tmp files and return check_goals' exit code."""
    (tmp_path / "m.json").write_text(json.dumps(METRICS))
    (tmp_path / "g.json").write_text(json.dumps({"goals": goals}))
    return check_goals.main([str(tmp_path / "m.json"), "--goals", str(tmp_path / "g.json")])


def test_wildcard_goal_fails_if_any_member_fails(tmp_path, capsys):
    """An active wildcard goal fails when one expanded metric misses, and names it."""
    goal = {"id": "G", "status": "active", "dataset": "d",
            "metric": "methods.*.overall.error_rate", "op": "<=", "target": 0.0}
    assert _run(tmp_path, [goal]) == 1
    assert "methods.b.overall.error_rate" in capsys.readouterr().out
    METRICS["methods"]["b"]["overall"]["error_rate"] = 0.0
    try:
        assert _run(tmp_path, [goal]) == 0
    finally:
        METRICS["methods"]["b"]["overall"]["error_rate"] = 0.1


def test_draft_goal_never_fails_run(tmp_path):
    """Draft goals are reported but don't set a failing exit code."""
    goal = {"id": "G", "status": "draft", "dataset": "d",
            "metric": "final.by_bin.<1d.success", "op": ">=", "target": 0.99}
    assert _run(tmp_path, [goal]) == 0


# ---- headline-card, relative and runtime goals (2026-11-13) ----
HEADLINE = {
    "meta": {"mode": "fullyear", "dataset": "headline:dev_medium:fullyear,population=dev_medium"},
    "cards": {
        "hand (T5)": {"kind": "hand", "values": {"R2_success_pop": 0.25, "R4_precision": {"0.1": 0.71}}},
        "ranker v0": {"kind": "ranker", "values": {"R2_success_pop": 0.27, "R4_precision": {"0.1": 0.53}}},
    },
}
HDS = HEADLINE["meta"]["dataset"]


def _run_files(tmp_path, run, goals, base=None, extra=()):
    """Write run/goals (and an optional baseline) to tmp files and return check_goals' exit code."""
    (tmp_path / "r.json").write_text(json.dumps(run))
    (tmp_path / "g.json").write_text(json.dumps({"goals": goals}))
    argv = [str(tmp_path / "r.json"), "--goals", str(tmp_path / "g.json"), *extra]
    if base is not None:
        (tmp_path / "b.json").write_text(json.dumps(base))
        argv += ["--baseline", str(tmp_path / "b.json")]
    return check_goals.main(argv)


def test_headline_goal_reads_the_named_systems_card(tmp_path):
    """A headline goal looks its metric up in the card its system names (kind or label)."""
    ok = {"id": "H", "status": "active", "dataset": HDS, "system": "ranker",
          "metric": "R2_success_pop", "op": ">=", "target": 0.26}
    assert _run_files(tmp_path, HEADLINE, [ok]) == 0
    assert _run_files(tmp_path, HEADLINE, [dict(ok, system="hand (T5)")]) == 1


def test_target_system_compares_two_cards(tmp_path, capsys):
    """target_system makes the target the same metric on another card (dotted prevalence key)."""
    g = {"id": "H3", "status": "active", "dataset": HDS, "system": "ranker", "target_system": "hand",
         "metric": "R4_precision.0.1", "op": ">="}
    assert _run_files(tmp_path, HEADLINE, [g]) == 1
    assert "0.710" in capsys.readouterr().out


def test_ambiguous_system_needs_override(tmp_path):
    """Two cards of one kind fail an active goal unless --system picks one."""
    run = json.loads(json.dumps(HEADLINE))
    run["cards"]["ranker v1"] = {"kind": "ranker", "values": {"R2_success_pop": 0.30}}
    g = {"id": "H", "status": "active", "dataset": HDS, "system": "ranker",
         "metric": "R2_success_pop", "op": ">=", "target": 0.28}
    assert _run_files(tmp_path, run, [g]) == 1
    assert _run_files(tmp_path, run, [g], extra=["--system", "ranker=ranker v1"]) == 0


def test_relative_goals_need_a_baseline_and_respect_direction(tmp_path):
    """max_drop (higher is better) and max_rise_frac (lower is better) compare with --baseline; no baseline = n/a."""
    drop = {"id": "H5", "status": "active", "dataset": HDS, "system": "ranker",
            "metric": "R2_success_pop", "relative": {"max_drop": 0.02}}
    assert _run_files(tmp_path, HEADLINE, [drop]) == 0                      # n/a never fails
    worse = json.loads(json.dumps(HEADLINE))
    worse["cards"]["ranker v0"]["values"]["R2_success_pop"] = 0.24
    assert _run_files(tmp_path, worse, [drop], base=HEADLINE) == 1
    assert _run_files(tmp_path, HEADLINE, [drop], base=worse) == 0

    prof = {"config": {"dataset": "profile:x:fullyear,features=1"}, "runtime": {"mean_total_s": 1.0}}
    slow = {"config": prof["config"], "runtime": {"mean_total_s": 1.2}}
    rise = {"id": "RT", "status": "active", "dataset": "profile:x:fullyear,features=1",
            "metric": "runtime.mean_total_s", "relative": {"max_rise_frac": 0.10}}
    assert _run_files(tmp_path, slow, [rise], base=prof) == 1
    assert _run_files(tmp_path, prof, [rise], base=slow) == 0


def test_headline_card_dataset_string():
    """headline_card tags its JSON with the dataset string the H goals match."""
    sys.path.insert(0, str(ROOT / "tools"))
    import headline_card
    assert headline_card.headline_dataset("datasets/dev_medium.csv", "fullyear", "dev_medium") == HDS


def test_profile_summary_is_check_goals_readable(tmp_path):
    """profile_pipeline.write_summary averages per-star medians and writes the runtime dataset string."""
    import types
    import pandas as pd
    import profile_pipeline
    df = pd.DataFrame({"star_id": ["a", "a", "b", "b"], "fit": [1.0, 3.0, 2.0, 2.0],
                       "total": [2.0, 4.0, 3.0, 3.0]})
    ids = ROOT / "benchmarks" / "profile_stars.txt"
    args = types.SimpleNamespace(star_ids_file=ids, synthetic=None, features=True, repeats=2,
                                 summary_json=tmp_path / "s.json")
    s = profile_pipeline.write_summary(df, ["fit", "total"], args, "fullyear")
    assert s["config"]["dataset"] == "profile:benchmarks/profile_stars.txt:fullyear,features=1"
    assert s["runtime"]["mean_total_s"] == 3.0 and s["runtime"]["mean_stage_s"]["fit"] == 2.0
    assert json.loads((tmp_path / "s.json").read_text()) == s
