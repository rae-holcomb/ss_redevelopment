"""Tests for the frozen T2 datasets (datasets/*.csv), their builder, and the scorecard's
dataset options.

Three groups:

- builder logic on synthetic inputs (no data needed): quotas, water-filling, rank
  reproducibility, coverage assignment feasibility and balance;
- scorecard dataset options (--data-root, --ignore-manifest-sectors), no data needed;
- the committed manifests (skip when absent) and the local FITS files / drive
  (skip when absent): disjointness, SIM_ID >= 800k, stratum counts vs the spec,
  pattern subset of native_sectors, schema, and seed reproducibility.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "claude_files"))

import build_dev_sets as bds  # noqa: E402
import coverage as cov  # noqa: E402

DATASETS = ROOT / "datasets"
SETS = {name: DATASETS / f"{name}.csv" for name in bds.SET_ORDER}


def _load(name):
    """Read one manifest, skipping the test if it hasn't been built."""
    if not SETS[name].exists():
        pytest.skip(f"{SETS[name]} not built")
    return pd.read_csv(SETS[name], dtype={"native_sectors": str, "native_whole_sectors": str, "sectors": str})


# ---------------------------------------------------------------- builder logic

def test_bin_targets_match_spec():
    """Equal shares for the five bins < 60 d, ~1/7 of the set in 60-180 d, sizes exact."""
    for name, n in bds.SET_SIZES.items():
        t = bds.bin_targets(n)
        assert sum(t.values()) == n
        short = [t[b] for b in bds.SHORT_BINS]
        assert max(short) - min(short) <= 1
        n_long = sum(t[b] for b in bds.LONG_BINS)
        assert n_long == round(n / 7)
        assert abs(t["60-120d"] - t["120-180d"]) <= 1


def test_waterfill_equal_and_capped():
    """Equal shares where capacity allows; capped cells' deficit moves to the others."""
    rng = np.random.default_rng(0)
    assert list(bds.waterfill(9, np.full(9, 100), rng)) == [1] * 9
    a = bds.waterfill(90, np.array([5, 100, 100, 100, 100, 100, 100, 100, 100]), rng)
    assert a.sum() == 90 and a[0] == 5 and set(a[1:]) <= {10, 11}
    with pytest.raises(ValueError):
        bds.waterfill(10, np.array([3, 3]), rng)


def _fake_pool(n=4000, seed=1):
    """Synthetic pool with the builder's stratum columns (no catalog needed)."""
    rng = np.random.default_rng(seed)
    pb = rng.choice(bds.AUDIT_PBIN_LABELS, n)
    a = rng.choice(bds.AMP_TIERS, n)
    s = rng.choice(bds.SHEAR_CLASSES, n)
    return pd.DataFrame({"row": rng.permutation(10 * n)[:n], "stratum": [f"{x}|{y}|{z}" for x, y, z in zip(pb, a, s)]})


def test_rank_pool_reproducible():
    """Same seed -> same per-stratum ranks; a different seed changes them."""
    p = _fake_pool()
    r1 = bds.rank_pool(p, 7)
    r2 = bds.rank_pool(p.sample(frac=1, random_state=3), 7)  # input order must not matter
    m = r1.merge(r2, on="row")
    assert (m.rank_x == m.rank_y).all()
    assert not (bds.rank_pool(p, 8).sort_values("row")["rank"].values == r1.sort_values("row")["rank"].values).all()
    for _, g in r1.groupby("stratum"):
        assert sorted(g["rank"]) == list(range(len(g)))


def test_allocate_respects_caps_and_sizes():
    """Quotas sum to each set size and never exceed a stratum's capacity across sets."""
    keys = [f"{pb}|{a}|{s}" for pb in bds.AUDIT_PBIN_LABELS for a, s in bds.CELLS]
    caps = {k: 1000 for k in keys}
    caps["<1d|low|anti"] = 30  # a small cell, like the real <1 d strata
    q = bds.allocate(caps, 0)
    assert q == bds.allocate(caps, 0)
    for name, n in bds.SET_SIZES.items():
        assert sum(q[name].values()) == n
    assert sum(q[name]["<1d|low|anti"] for name in q) <= 30


def test_assign_coverage_feasible_and_balanced():
    """Every assigned pattern is in the library, has the stated class and fits native_sectors;
    classes are balanced (within one) per period bin when every row is unconstrained."""
    lib = cov.load_library()
    rng = np.random.default_rng(0)
    rows = []
    for pb in bds.AUDIT_PBIN_LABELS[:3]:
        for i in range(24):
            full = i % 2 == 0
            native = list(range(1, 14)) if full else sorted(rng.choice(np.arange(1, 14), 10, replace=False))
            rows.append(dict(pbin=pb, native_whole_sectors=" ".join(map(str, native))))
    df = bds.assign_coverage(pd.DataFrame(rows), lib, seed=0, set_index=0)
    libset = dict(zip(lib.pattern, lib["class"]))
    for _, r in df.iterrows():
        assert libset[r.sectors] == r.coverage_class
        assert set(cov.parse_pattern(r.sectors)) <= set(map(int, r.native_whole_sectors.split()))
    assert not df.coverage_fallback.any()
    for _, g in df.groupby("pbin"):
        c = g.coverage_class.value_counts().reindex(cov.COVERAGE_CLASSES, fill_value=0)
        assert c.max() - c.min() <= 1
    again = bds.assign_coverage(pd.DataFrame(rows), lib, seed=0, set_index=0)
    assert (again.sectors == df.sectors).all()


# ---------------------------------------------------------------- scorecard dataset options

def _args(**kw):
    """Minimal argparse-like namespace for scorecard helpers."""
    d = dict(synthetic=None, manifest=None, seed=0, coverage="full", coverage_seed=0,
             coverage_weighting="uniform", coverage_library=None, data_root=None,
             ignore_manifest_sectors=False)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_scorecard_data_root_and_ignore_sectors(tmp_path, monkeypatch):
    """--data-root joins relative paths (repo-relative root), keeps absolute ones; the
    ignore switch drops the sectors column; both are recorded in the dataset string."""
    import scorecard
    monkeypatch.delenv("SMARTS_DATA_ROOT", raising=False)
    man = tmp_path / "m.csv"
    pd.DataFrame({"path": ["a.fits", "/abs/b.fits"], "sectors": ["1-2", "5"]}).to_csv(man, index=False)

    t = scorecard.build_tasks(_args(manifest=man))
    assert [x["path"] for x in t] == ["a.fits", "/abs/b.fits"]
    assert scorecard.dataset_string(_args(manifest=man)) == f"{man},sectors=manifest"

    t = scorecard.build_tasks(_args(manifest=man, data_root="data/smarts_dev"))
    assert t[0]["path"] == str(ROOT / "data/smarts_dev/a.fits") and t[1]["path"] == "/abs/b.fits"
    assert [x["load_kwargs"] for x in t] == [{"sectors": [1, 2]}, {"sectors": [5]}]

    t = scorecard.build_tasks(_args(manifest=man, data_root="data/smarts_dev", ignore_manifest_sectors=True))
    assert [x["load_kwargs"] for x in t] == [{}, {}]
    assert (scorecard.dataset_string(_args(manifest=man, data_root="data/smarts_dev", ignore_manifest_sectors=True))
            == f"{man},data_root=data/smarts_dev,sectors=ignored")

    monkeypatch.setenv("SMARTS_DATA_ROOT", "/Volumes/X")
    assert scorecard.build_tasks(_args(manifest=man))[0]["path"] == "/Volumes/X/a.fits"
    assert scorecard.dataset_string(_args(manifest=man)) == f"{man},data_root=/Volumes/X,sectors=manifest"
    # synthetic dataset strings are untouched by the new options
    assert scorecard.dataset_string(_args(synthetic=200, data_root="x")) == "synthetic:n=200,seed=0"


# ---------------------------------------------------------------- committed manifests

def test_manifest_schema():
    """Every manifest has exactly the builder's columns, no missing identifiers or truth."""
    for name in bds.SET_ORDER:
        m = _load(name)
        assert list(m.columns) == bds.MANIFEST_COLUMNS
        assert len(m) == bds.SET_SIZES[name]
        key = ["path", "star_id", "row", "sim_id", "bundle", "member", "true_period", "shear",
               "minlat", "maxlat", "P_lo", "P_hi", "native_sectors", "native_whole_sectors", "sectors",
               "coverage_class",
               "stratum", "sampling_weight", "coverage_weight"]
        assert not m[key].isna().any().any(), name
        assert m.row.is_unique and m.star_id.is_unique
        assert (m.star_id == m.row.map(lambda r: f"smarts-tess-v1.0-{r:06d}")).all()
        assert (m.member.str.split("/").str[-1] == m.star_id + ".fits").all()
        if name in bds.LOCAL_SETS:
            assert (m.path == m.star_id + ".fits").all()
        else:
            assert (m.path == bds.HOLDOUT_SUBDIR + "/" + m.star_id + ".fits").all()


def test_manifests_disjoint():
    """No catalog row (and so no file) appears in two sets."""
    rows = {name: set(_load(name).row) for name in bds.SET_ORDER}
    names = list(rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not rows[names[i]] & rows[names[j]], (names[i], names[j])


def test_manifests_sim_id_pool():
    """Every row is from the SIM_ID >= 800,000 (val/test galaxy) pool."""
    for name in bds.SET_ORDER:
        assert (_load(name).sim_id >= bds.POOL_CUT).all()


def test_stratum_counts_match_spec():
    """Per-bin counts equal bin_targets; amp x shear cells within a bin differ by <= 1
    unless a cell was capacity-limited (recorded in strata_counts.csv)."""
    counts_path = DATASETS / "strata_counts.csv"
    counts = pd.read_csv(counts_path) if counts_path.exists() else None
    for name in bds.SET_ORDER:
        m = _load(name)
        per_bin = m.pbin.value_counts().to_dict()
        assert per_bin == {k: v for k, v in bds.bin_targets(bds.SET_SIZES[name]).items() if v}
        assert set(m.stratum) == set(m.pbin + "|" + m.amp_tier + "|" + m.shear_class)
        if counts is not None:
            got = m.stratum.value_counts()
            want = counts.set_index("stratum")[f"quota_{name}"]
            assert (got.reindex(want.index, fill_value=0) == want).all()
        for pb, g in m.groupby("pbin"):
            c = g.groupby(["amp_tier", "shear_class"]).size().reindex(
                pd.MultiIndex.from_tuples(bds.CELLS), fill_value=0)
            if counts is None or not _capacity_limited(counts, pb, name):
                assert c.max() - c.min() <= 1, (name, pb, c.to_dict())


def _capacity_limited(counts: pd.DataFrame, pb: str, name: str) -> bool:
    """True if some cell of bin ``pb`` hit its capacity (all passing rows used across sets)."""
    g = counts[counts.pbin == pb]
    used = sum(g[f"n_{s}"] for s in bds.SET_ORDER)
    return bool(((g.n_checked - g.n_excluded) <= used).any() & (g.n_checked == g.n_pool).any())


def test_coverage_pattern_within_native_sectors():
    """Each assigned pattern is a library pattern of its class and a subset of native_whole_sectors
    (itself a subset of native_sectors);
    coverage classes are balanced within each period bin (counts differ by at most 1, apart
    from recorded fallbacks)."""
    lib = cov.load_library()
    libset = dict(zip(lib.pattern, lib["class"]))
    for name in bds.SET_ORDER:
        m = _load(name)
        for _, r in m.iterrows():
            pat = cov.parse_pattern(r.sectors)
            assert libset[cov.format_pattern(pat)] == r.coverage_class
            whole = set(map(int, r.native_whole_sectors.split()))
            assert whole <= set(map(int, r.native_sectors.split()))
            assert set(pat) <= whole, (name, r.star_id)
            assert r.n_sectors == len(pat)
        for pb, g in m.groupby("pbin"):
            c = g.coverage_class.value_counts().reindex(cov.COVERAGE_CLASSES, fill_value=0)
            assert c.max() - c.min() <= 2, (name, pb, c.to_dict())


def test_weights_reweight_to_corpus():
    """sampling_weight sums to the set size; coverage_weight reproduces the library's class mix."""
    lib = cov.load_library()
    real = lib.groupby("class").n_stars.sum() / lib.n_stars.sum()
    for name in bds.SET_ORDER:
        m = _load(name)
        assert m.sampling_weight.sum() == pytest.approx(len(m))
        w = m.groupby("coverage_class").coverage_weight.sum() / m.coverage_weight.sum()
        present = real.reindex(w.index)
        assert np.allclose(w, present / present.sum())


def test_exclusions_not_in_sets():
    """No excluded (corrupt / unloadable) row is in any set; all excluded rows have a reason."""
    p = DATASETS / "exclusions_log.csv"
    if not p.exists():
        pytest.skip("exclusions_log.csv not built")
    ex = pd.read_csv(p)
    assert ex.reason.notna().all()
    used = set().union(*(set(_load(n).row) for n in bds.SET_ORDER))
    assert not used & set(ex.row)
    for name in bds.SET_ORDER:
        assert (_load(name).max_abs_dev <= bds.CORRUPT_THRESH).all()


# ---------------------------------------------------------------- local data / drive

def test_local_dev_files_present_and_native_sectors():
    """dev_small files exist under data/smarts_dev; load_smarts_fits reproduces native_sectors and
    keeps exactly the assigned pattern (every row)."""
    m = _load("dev_small")
    paths = [bds.LOCAL_DIR / p for p in m.path]
    if not all(p.exists() for p in paths):
        pytest.skip("local dev files (data/smarts_dev) not present")
    from preprocessing import load_smarts_fits
    for (_, r), p in zip(m.iterrows(), paths):
        assert sorted(load_smarts_fits(p).sectors_used) == list(map(int, r.native_sectors.split()))
        assert tuple(load_smarts_fits(p, sectors=list(cov.parse_pattern(r.sectors))).sectors_used) \
            == cov.parse_pattern(r.sectors)


def _drive_root():
    """The SMARTS data root if the drive and the builder's check cache are present, else None."""
    import smarts_data as sd
    root = sd.data_root(None)
    if not (root / sd.BUNDLE_DIR).is_dir() or not bds.check_path(root, 1).parent.is_dir():
        return None
    return root


@pytest.mark.slow
def test_build_reproducible_from_seed(tmp_path, monkeypatch):
    """Rebuilding with the default seed (from the drive's cached checks) reproduces the committed
    manifests exactly. Skips when the drive or the manifests are absent."""
    for name in bds.SET_ORDER:
        _load(name)
    root = _drive_root()
    if root is None:
        pytest.skip("SMARTS drive / T2 check cache not available")
    monkeypatch.setattr(bds, "install_files", lambda df, name, root: 0)  # don't touch data dirs
    bds.main(["--data-root", str(root), "--out-dir", str(tmp_path), "--n-workers", "1"])
    for name in bds.SET_ORDER:
        a = pd.read_csv(SETS[name])
        b = pd.read_csv(tmp_path / f"{name}.csv")
        pd.testing.assert_frame_equal(a, b)
