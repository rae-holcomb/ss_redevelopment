"""Contaminated-noise stars stay out of development and judging (Rae, 2026-11-16).

SMARTS stars whose real-TESS noise source was flagged `noise_exclude` (periodic or contaminating signal before
the rotation was injected) must never be used to develop or judge the pipeline. The *_clean manifests drop them,
the scorecard refuses a manifest that contains them, and the clean baselines are the frozen ones restricted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))
import build_clean_manifests as bcm  # noqa: E402
import scorecard as sc  # noqa: E402

NOISE = pd.read_csv(ROOT / "benchmarks" / "noise_sources_dev.csv")
EXCLUDED = set(NOISE.loc[NOISE["noise_exclude"].astype(bool), "sim_id"])


@pytest.mark.parametrize("name", bcm.MANIFESTS)
def test_clean_manifest_is_frozen_minus_excluded(name):
    """<name>_clean.csv = the frozen manifest's rows (same columns, order) whose sim_id is not noise_exclude."""
    frozen = pd.read_csv(ROOT / "datasets" / f"{name}.csv")
    clean = pd.read_csv(ROOT / "datasets" / f"{name}_clean.csv")
    assert not set(clean["sim_id"]) & EXCLUDED
    expect = frozen[~frozen["sim_id"].isin(EXCLUDED)].reset_index(drop=True)
    pd.testing.assert_frame_equal(clean, expect)
    assert len(clean) < len(frozen)


def test_committed_clean_manifests_are_current():
    """tools/build_clean_manifests.py --check passes (rebuild after the noise table changes)."""
    assert bcm.main(["--check"]) == 0


def _args(manifest, **kw):
    return SimpleNamespace(synthetic=None, manifest=ROOT / "datasets" / manifest, seed=0, data_root="data/smarts_dev",
                           ignore_manifest_sectors=True, coverage=sc.cov.FULL, **kw)


def test_scorecard_refuses_manifest_with_excluded_stars():
    """build_tasks (scorecard, profiling, feature tables, harnesses) stops on a noise_exclude star."""
    with pytest.raises(SystemExit, match="noise_exclude"):
        sc.build_tasks(_args("dev_medium.csv"))
    tasks = sc.build_tasks(_args("dev_medium_clean.csv"))
    assert len(tasks) == len(pd.read_csv(ROOT / "datasets" / "dev_medium_clean.csv"))
    assert not any(t["noise"]["noise_exclude"] for t in tasks)


def test_allow_flag_is_explicit_and_labelled():
    """--allow-noise-excluded lets an audit through and marks config.dataset."""
    args = _args("dev_small.csv", allow_noise_excluded=True)
    assert len(sc.build_tasks(args)) == len(pd.read_csv(ROOT / "datasets" / "dev_small.csv"))
    assert sc.dataset_string(args).endswith(",noise_excluded=allowed")
    assert "noise_excluded" not in sc.dataset_string(_args("dev_small_clean.csv"))


def test_profile_stars_are_clean():
    """The runtime-profile star list (goal RT2) holds only clean dev_medium stars."""
    ids = [x.strip() for x in (ROOT / "benchmarks" / "profile_stars.txt").read_text().splitlines()
           if x.strip() and not x.startswith("#")]
    clean = set(pd.read_csv(ROOT / "datasets" / "dev_medium_clean.csv")["star_id"])
    assert len(ids) == 15 and set(ids) <= clean


@pytest.mark.parametrize("mode,sectors", [("fullyear", "ignored"), ("assigned", "manifest")])
def test_clean_v7_baselines(mode, sectors):
    """The clean _v7 baselines cover exactly the clean manifest, under its dataset string."""
    m = json.loads((ROOT / f"benchmarks/baselines/dev_medium_clean_{mode}_v7.json").read_text())
    assert m["config"]["dataset"] == f"datasets/dev_medium_clean.csv,data_root=data/smarts_dev,sectors={sectors}"
    assert m["final"]["overall"]["n"] == len(pd.read_csv(ROOT / "datasets" / "dev_medium_clean.csv"))
    ps = pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_clean_{mode}_v7_tables/per_star.csv")
    assert not ps["noise_exclude"].astype(bool).any()


def test_goals_name_only_clean_dev_sets():
    """No goal is judged on a manifest that still holds noise_exclude stars."""
    goals = json.loads((ROOT / "benchmarks" / "goals.json").read_text())["goals"]
    for g in goals:
        ds = g.get("dataset", "")
        assert "dev_medium.csv" not in ds and "dev_small.csv" not in ds and "headline:dev_medium:" not in ds, g["id"]


def test_headline_card_restricts_to_manifest_stars():
    """headline_card scores only the manifest's stars and refuses a manifest with excluded stars."""
    import headline_card as hc
    frame = pd.DataFrame({"star_id": ["a", "b", "c"], "x": [1, 2, 3]})
    assert list(hc.restrict_to_stars(frame, {"a", "c"}, "t")["star_id"]) == ["a", "c"]
    with pytest.raises(SystemExit):
        hc.restrict_to_stars(frame, {"a", "z"}, "t")
