"""Scorecard v6 (T26/P26): detectability join, `detectable` flag and detectable-subset metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))

import scorecard  # noqa: E402


def _per_star():
    """Four stars: two with table rows at their loaded sectors, one at a different draw, one synthetic-like."""
    return pd.DataFrame(dict(
        star_id=["a", "b", "c", "d"], sectors=["1-2-3", "5", "7-8", None],
        final_outcome=["correct", "none", "wrong", "none"], period_bin=["<1d", "1-3d", "1-3d", "<1d"],
        coverage_class=["full", "single", "gapped-2", "full"], searchable=[True, True, False, True]))


def _table():
    """Detectability rows keyed by (star_id, sectors); star c only has a row for another pattern."""
    return pd.DataFrame(dict(star_id=["a", "b", "c"], sectors=["1-2-3", "5", "1-2"], snr_rot=[12.0, 2.0, 40.0]))


def test_join_is_on_star_and_loaded_sectors():
    out = scorecard.add_detectability_columns(_per_star(), _table(), threshold=5.0)
    assert out["detectability"].tolist()[:2] == [12.0, 2.0]
    assert np.isnan(out["detectability"].iloc[2]) and np.isnan(out["detectability"].iloc[3])
    assert out["detectable"].tolist() == [True, False, None, None]


def test_join_leaves_existing_columns_untouched():
    ps = _per_star()
    out = scorecard.add_detectability_columns(ps, _table())
    pd.testing.assert_frame_equal(out[ps.columns], ps)
    assert scorecard.add_detectability_columns(ps, None) is ps


def test_threshold_is_inclusive_and_configurable():
    out = scorecard.add_detectability_columns(_per_star(), _table(), threshold=12.0)
    assert out["detectable"].tolist()[:2] == [True, False]


def test_v6_breakdowns_only_count_known_stars():
    ps = scorecard.add_detectability_columns(_per_star(), _table())
    f = scorecard._v6_detectability_breakdowns(ps)
    assert f["detectable"]["overall"]["n"] == 1 and f["detectable"]["overall"]["success"] == 1.0
    assert f["undetectable"]["overall"]["n"] == 1 and f["undetectable"]["overall"]["none"] == 1.0
    assert f["detectable_searchable"]["overall"]["n"] == 1


def test_v6_breakdowns_absent_without_table():
    assert scorecard._v6_detectability_breakdowns(_per_star()) == {}


def test_committed_table_covers_dev_sets_in_both_modes():
    t = pd.read_csv(ROOT / "benchmarks" / "detectability_dev.csv", dtype={"sectors": str})
    assert not t.duplicated(["star_id", "sectors"]).any()
    assert t["snr_rot"].notna().all() and (t["snr_rot"] > 0).all()
    for name in ("dev_small", "dev_medium"):
        man = pd.read_csv(ROOT / "datasets" / f"{name}.csv")
        modes = t[t.star_id.isin(man.star_id)].groupby("star_id")["modes"].apply(lambda m: "|".join(m))
        assert len(modes) == len(man)
        assert modes.str.contains("fullyear").all() and modes.str.contains("assigned").all()


def test_combine_merges_identical_keys():
    import build_detectability_table as b
    base = dict(sim_id=1, D_band_sqrt_ncyc=3.0, D_band=1.0, n_cycles_obs=9.0, D_peak=2.0, noise_tic=7,
                stack_which=2, error="")
    f1 = pd.DataFrame([dict(star_id="a", mode="fullyear", sectors="1-2", **base)])
    f2 = pd.DataFrame([dict(star_id="a", mode="assigned", sectors="1-2", **base),
                       dict(star_id="a", mode="assigned", sectors="2", **base)])
    t = b.combine([f1, f2])
    assert len(t) == 2
    assert t.set_index("sectors").loc["1-2", "modes"] == "assigned|fullyear"
