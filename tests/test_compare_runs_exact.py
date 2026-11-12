"""tools/compare_runs_exact.py: exact, NaN-aware, runtime-blind run comparison (T12)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_runs_exact as cre  # noqa: E402


def _scorecard(d: Path, rows, metrics):
    """Write a minimal scorecard output directory."""
    d.mkdir()
    (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "metrics.json").write_text(json.dumps(metrics))


def test_identical_except_runtime(tmp_path):
    """Runtime keys and the config block are ignored; NaN == NaN."""
    r = dict(star_id="a", final_period=1.5, x=float("nan"), methods={"ls": dict(top1=2.0, seconds=0.1)})
    _scorecard(tmp_path / "A", [dict(r, seconds=1.0)], dict(final={"s": 0.25}, config=dict(git_commit="x")))
    _scorecard(tmp_path / "B", [dict(r, seconds=9.0, methods={"ls": dict(top1=2.0, seconds=5.0)})],
               dict(final={"s": 0.25}, config=dict(git_commit="y")))
    s = cre.compare_scorecards(tmp_path / "A", tmp_path / "B")
    assert s["n_diffs"] == 0 and s["values_compared"] == 5 and s["runtime_keys_skipped"] >= 3


def test_last_bit_difference_is_caught(tmp_path):
    """A one-ulp change in any nested value is a difference."""
    p = 1.5
    _scorecard(tmp_path / "A", [dict(star_id="a", cands=[dict(P=p)])], {})
    _scorecard(tmp_path / "B", [dict(star_id="a", cands=[dict(P=np.nextafter(p, 2))])], {})
    s = cre.compare_scorecards(tmp_path / "A", tmp_path / "B")
    assert s["n_diffs"] == 1 and "cands[0].P" in s["diffs"][0]


def test_feature_tables(tmp_path):
    """Shard comparison checks every candidate cell, ignoring the star's seconds."""
    for name, val in (("A", 0.1), ("B", 0.1), ("C", 0.1000000001)):
        (tmp_path / name / "shards").mkdir(parents=True)
        df = pd.DataFrame(dict(period=[1.0, 2.0], f=[val, np.nan], s=["x", None]))
        with open(tmp_path / name / "shards" / "a.pkl", "wb") as fh:
            pickle.dump(dict(star=dict(star_id="a", seconds=hash(name) % 7), candidates=df), fh)
    assert cre.compare_feature_tables(tmp_path / "A", tmp_path / "B")["n_diffs"] == 0
    assert cre.compare_feature_tables(tmp_path / "A", tmp_path / "C")["n_diffs"] == 1
