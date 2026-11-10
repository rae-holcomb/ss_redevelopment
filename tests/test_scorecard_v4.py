"""Scorecard v4 (T22 band-width breakdown, T23 searchable subset): evaluation helpers and metric groups."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import acf_utils  # noqa: E402
import evaluation as ev  # noqa: E402
import scorecard  # noqa: E402


def _lc(n=4000, dt=0.02, gap=None):
    """Evenly sampled time and unit flux, with an optional (start, stop) NaN gap of indices."""
    t = np.arange(n) * dt
    f = np.ones(n)
    if gap:
        f[gap[0]:gap[1]] = np.nan
    return t, f


def test_acf_reach_contiguous_is_70_percent_of_baseline():
    t, f = _lc()
    assert ev.acf_reach(t, f) == pytest.approx(0.7 * t[-1], abs=0.05)


def test_acf_reach_shrinks_with_big_gap():
    t, f = _lc()
    t2, f2 = _lc(gap=(1500, 3000))
    assert ev.acf_reach(t2, f2) < ev.acf_reach(t, f)


def test_acf_reach_ignores_pipeline_defaults(monkeypatch):
    t, f = _lc(gap=(1500, 2500))
    r0 = ev.acf_reach(t, f)
    monkeypatch.setattr(acf_utils.compute_acf, "__defaults__", (0.3, 0.9, False))
    assert ev.acf_reach(t, f) == r0


def test_acf_reach_too_few_points_is_nan():
    assert np.isnan(ev.acf_reach(np.arange(5.0), np.ones(5)))


@pytest.mark.parametrize("p_lo, reach, expect", [(5.0, 10.0, True), (5.1, 10.0, False), (None, 10.0, False),
                                                  (5.0, float("nan"), False)])
def test_is_searchable(p_lo, reach, expect):
    assert ev.is_searchable(p_lo, reach) is expect


def test_band_width_classes():
    assert scorecard.band_width_class(10, (10, 10)) == (0.0, "zero")
    assert scorecard.band_width_class(10, (10, 11.9))[1] == "lt_0p2"
    assert scorecard.band_width_class(10, (10, 13))[1] == "0p2_to_0p5"
    assert scorecard.band_width_class(10, (10, 15))[1] == "ge_0p5"
    assert scorecard.band_width_class(10, (10, float("inf")))[1] == "ge_0p5"
    assert scorecard.band_width_class(10, None) == (None, "unknown")


def _frame(extra):
    """Small per_star / per_method frames; `extra` adds v4 columns."""
    per_star = pd.DataFrame(dict(
        star_id=list("abcdef"), true_period=[1, 2, 5, 12, 20, 40.0], period_bin=["<1d", "1-3d", "3-10d", "10-30d", "10-30d", ">30d"],
        final_outcome=["correct", "diffrot", "none", "wrong", "none", "correct"],
        seconds=1.0, error=None))
    if extra:
        per_star["band_width_class"] = ["zero", "lt_0p2", "ge_0p5", "zero", "unknown", "zero"]
        per_star["coverage_class"] = ["single"] * 3 + ["gapped-2"] * 3
        per_star["searchable"] = [True, True, True, True, False, False]
    return per_star, pd.DataFrame()


def test_v3_metrics_unchanged_by_v4_columns():
    old = scorecard.compute_metrics(*_frame(False))
    new = scorecard.compute_metrics(*_frame(True))
    for k, v in old.items():
        if k == "final":
            for kk, vv in v.items():
                assert new["final"][kk] == vv
        else:
            assert new[k] == v


def test_v4_metric_groups():
    m = scorecard.compute_metrics(*_frame(True))["final"]
    assert m["by_band_width"]["zero"]["n"] == 3
    assert m["by_coverage_class"]["gapped-2"]["n"] == 3
    assert m["searchable"]["overall"]["n"] == 4
    assert m["searchable"]["overall"]["success"] == pytest.approx(0.5)
    assert m["unsearchable"]["overall"]["n"] == 2
    assert m["unsearchable"]["overall"]["none"] == pytest.approx(0.5)
    assert m["unsearchable"]["overall"]["success"] == pytest.approx(0.5)
    assert m["searchable"]["by_bin"]["10-30d"]["n"] == 1


def test_scorecard_version_and_columns():
    assert scorecard.SCORECARD_VERSION == 5
    t, f = _lc(gap=(1500, 2500))
    cols = scorecard._searchability_columns(t, f + 0.01 * np.sin(t), 3.0, (2.8, 3.6))
    assert set(cols) == {"acf_reach", "P_over_reach", "searchable", "band_width", "band_width_class", "flux_amp"}
    assert cols["searchable"] and cols["band_width_class"] == "0p2_to_0p5"
    assert cols["P_over_reach"] == pytest.approx(3.0 / cols["acf_reach"])
