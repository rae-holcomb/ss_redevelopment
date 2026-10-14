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
