"""Tests for dev_medium_b (P57), the second noise-clean ~1,000-star dev set.

- builder logic on synthetic inputs (no data): pool filters, capacity-clipped allocation;
- the committed manifest (skips when absent): schema = dev_medium's, disjoint from dev_small /
  dev_medium / holdout / train_v0, SIM_ID >= 800k, every star in the noise table and none
  ``noise_exclude``, accepted by the scorecard's noise guard, no noise TIC shared with dev_medium;
- the drive (slow, skips when absent): ``build_dev_medium_b.py --check``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "claude_files"))

import build_dev_medium_b as bdb  # noqa: E402
import build_dev_sets as bds  # noqa: E402

DATASETS = ROOT / "datasets"
MANIFEST = DATASETS / bdb.OUT_FILES["manifest"]
NOISE = pd.read_csv(ROOT / "benchmarks" / "noise_sources_dev.csv")


def _b():
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST.name} not built")
    return pd.read_csv(MANIFEST, dtype={"native_sectors": str, "native_whole_sectors": str, "sectors": str})


# ---------------------------------------------------------------- builder logic

def test_filter_pool_drops_used_corrupt_excluded_and_shared_tics():
    pool = pd.DataFrame(dict(row=np.arange(8), stratum=["a"] * 8))
    flags = pd.DataFrame(dict(row=np.arange(8), sim_id=np.arange(8) + 800_000,
                              tic=[1, 2, 3, 4, 5, 1, 7, 8],
                              exclude_from_training=[False, False, False, True, False, False, False, False]))
    existing = {"dev_medium": pd.DataFrame(dict(row=[0])), "holdout": pd.DataFrame(dict(row=[1]))}
    out, c = bdb.filter_pool(pool, flags, existing, t2_excluded={2})
    # row 0, 1 used; 2 corrupt; 3 noise_exclude; 5 shares tic 1 with row 0
    assert list(out.row) == [4, 6, 7]
    assert (c["dropped_in_existing_set"], c["dropped_t2_excluded"], c["dropped_noise_exclude"],
            c["dropped_noise_tic_shared"]) == (2, 1, 1, 1)
    out2, _ = bdb.filter_pool(pool, flags, existing, t2_excluded={2}, allow_shared_tic=True)
    assert list(out2.row) == [4, 5, 6, 7]


def test_allocate_clips_short_bin_and_matches_t2_elsewhere(monkeypatch):
    # patch (and auto-restore) the module globals configure_bds() would set, so test_datasets is unaffected
    t2_allocate = bds.allocate
    monkeypatch.setattr(bds, "SET_ORDER", (bdb.NAME,))
    monkeypatch.setattr(bds, "SET_SIZES", {bdb.NAME: bdb.SIZE})
    caps = {f"{pb}|{a}|{s}": 1000 for pb in bds.AUDIT_PBIN_LABELS for a, s in bds.CELLS}
    full = bdb.allocate(caps, 1)[bdb.NAME]
    assert full == t2_allocate(caps, 1)[bdb.NAME]  # same as T2 when nothing is short
    short = dict(caps)
    for a, s in bds.CELLS:
        short[f"<1d|{a}|{s}"] = 5                              # 45 < the bin's ~172 target
    q = bdb.allocate(short, 1)[bdb.NAME]
    assert sum(v for k, v in q.items() if k.startswith("<1d|")) == 45
    assert sum(q.values()) == bdb.SIZE - (bds.bin_targets(bdb.SIZE)["<1d"] - 45)


# ---------------------------------------------------------------- committed manifest

def test_schema_matches_dev_medium():
    b = _b()
    assert list(b.columns) == bds.MANIFEST_COLUMNS == list(pd.read_csv(DATASETS / "dev_medium_clean.csv", nrows=1).columns)
    assert b.row.is_unique and b.sim_id.is_unique and b.path.is_unique
    assert 900 <= len(b) <= bdb.SIZE
    assert (b.sim_id >= bds.POOL_CUT).all()
    assert (b.max_abs_dev <= bds.CORRUPT_THRESH).all()
    assert (b.path == b.star_id + ".fits").all()


@pytest.mark.parametrize("other", ["dev_small", "dev_medium", "holdout", "train_v0"])
def test_disjoint_from_other_sets(other):
    b = _b()
    o = pd.read_csv(DATASETS / f"{other}.csv", usecols=["row", "sim_id"])
    assert not set(b.row) & set(o.row)
    assert not set(b.sim_id) & set(o.sim_id)


def test_noise_clean_and_in_noise_table():
    """Every star has a noise-table row (so the scorecard's guard really checks it) and none is noise_exclude."""
    b = _b()
    t = NOISE.set_index("sim_id")
    assert set(b.sim_id) <= set(t.index), "rebuild benchmarks/noise_sources_dev.csv (tools/build_noise_table.py)"
    assert not t.loc[list(b.sim_id), "noise_exclude"].astype(bool).any()
    assert t.loc[list(b.sim_id), "in_dev_medium_b"].astype(bool).all()
    assert int(t.in_dev_medium_b.sum()) == len(b)


def test_no_noise_tic_shared_with_dev_sets():
    """No galaxy-noise light curve serves both a dev_medium_b star and a dev_small / dev_medium star."""
    b = _b()
    t = NOISE.set_index("sim_id")
    b_tics = set(t.loc[list(b.sim_id), "tic"])
    other = t[t.in_dev_small.astype(bool) | t.in_dev_medium.astype(bool)]
    assert not b_tics & set(other.tic)


def test_scorecard_accepts_without_allow_flag():
    import scorecard as sc
    b = _b()
    tab = {int(r.sim_id): r._asdict() for r in NOISE.itertuples(index=False)}
    sc.check_no_noise_excluded(b, tab, MANIFEST)  # raises SystemExit if any star is noise_exclude


# ---------------------------------------------------------------- drive

@pytest.mark.slow
def test_build_check_from_drive_cache():
    """build_dev_medium_b.py --check reproduces the committed files from the drive's check cache."""
    _b()
    import smarts_data as sd
    root = sd.data_root(None)
    if not (root / sd.BUNDLE_DIR).is_dir() or not Path(root, sd.EXTRACT_DIR, bdb.BUILD_SUBDIR, "checks").is_dir():
        pytest.skip("SMARTS drive / P57 check cache not available")
    assert bdb.main(["--check", "--data-root", str(root)]) == 0
