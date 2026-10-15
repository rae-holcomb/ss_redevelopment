"""Tests for patchy sector coverage (task T16a).

Covers: the coverage library builder (claude_files/coverage.py +
tools/build_coverage_library.py), pattern sampling, coverage classes and
descriptors, synthetic `sectors=` (equivalence with load_smarts_fits's
conventions), the scorecard's --coverage plumbing, and bit-identity of the
default (full-coverage) outputs.
"""

import hashlib
import platform
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import coverage as cov  # noqa: E402
from preprocessing import recombine_segments, to_even_cadence  # noqa: E402
from synthetic import make_lightcurve, make_suite, tess_like_time_grid  # noqa: E402

TARGETS = ROOT / "target_df_shortform.csv"
N_CYCLE1_STARS = 3438  # stars in target_df_shortform.csv with >= 1 sector in 1-13


# ---------------------------------------------------------------- library

@pytest.fixture(scope="module")
def targets():
    """The real target table the library is built from."""
    return pd.read_csv(TARGETS, index_col=0)


@pytest.fixture(scope="module")
def library():
    """The committed default (all-cycles) coverage library."""
    return cov.load_library()


@pytest.fixture(scope="module")
def cycle1():
    """The committed Cycle-1 library."""
    return cov.load_library(cov.LIBRARY_DIR / "cycle1_patterns.csv")


def test_library_counts_sum_to_cycle1_stars(targets):
    """Every star with >= 1 Cycle-1 sector is counted exactly once."""
    lib = cov.build_library(targets, cycle=1)
    assert lib["n_stars"].sum() == N_CYCLE1_STARS
    assert len(lib) == 121  # same distinct patterns as the old sector_configs.txt
    assert lib["pattern"].is_unique
    np.testing.assert_allclose(lib["frac_stars"].sum(), 1.0)


def test_library_patterns_match_old_sector_configs(targets):
    """The deduplicated patterns are exactly those of sector_configs.txt (notebook cell 10)."""
    old = {cov.parse_pattern(line) for line in (ROOT / "sector_configs.txt").read_text().splitlines()
           if line.strip()}
    new = {cov.parse_pattern(p) for p in cov.build_library(targets)["pattern"]}
    assert new == old


def test_committed_libraries_are_up_to_date(targets, library):
    """Every datasets/coverage/*_patterns.csv equals a fresh build (rerun the builder if not)."""
    libs = []
    for cycle in cov.CYCLE_SECTORS:
        fresh = cov.build_library(targets, cycle=cycle)
        libs.append(fresh)
        committed = cov.load_library(cov.LIBRARY_DIR / f"cycle{cycle}_patterns.csv")
        pd.testing.assert_frame_equal(committed, fresh, check_dtype=False)
    pd.testing.assert_frame_equal(library, cov.combine_libraries(libs), check_dtype=False)


def test_cycle_libraries_count_every_star_season(targets, library):
    """Each cycle's counts sum to the stars observed in that cycle; the pooled library sums them."""
    from ast import literal_eval
    sec = targets["sec"].apply(literal_eval)
    total = 0
    for cycle, (first, last) in cov.CYCLE_SECTORS.items():
        n_obs = int(sec.apply(lambda s: any(first <= v <= last for v in s)).sum())
        lib = cov.build_library(targets, cycle=cycle)
        assert lib["n_stars"].sum() == n_obs
        assert lib["last_sector"].max() <= cov.N_SECTORS_YEAR
        total += n_obs
    assert library["n_stars"].sum() == total
    assert library["pattern"].is_unique


@pytest.mark.parametrize("raw,want,action", [
    ((3, 4), (3, 4), "none"),
    ((13,), (13,), "none"),
    ((14,), (13,), "shifted"),
    ((10, 12, 16), (7, 9, 13), "shifted"),
    ((1, 2, 3, 16), (1, 2, 3), "truncated"),        # window 1-13 holds 3 sectors, 4-16 only 1
    ((1, 14, 15, 16), (11, 12, 13), "truncated"),   # window 4-16 holds 3 sectors
    (tuple(range(1, 17)), tuple(range(1, 14)), "truncated"),
])
def test_fit_to_year(raw, want, action):
    """Patterns from 14-16-sector cycles are shifted, or truncated to their best 13-sector window."""
    assert cov.fit_to_year(raw) == (want, action)


def test_build_library_toy_counts():
    """Sectors > 13 are dropped, stars with no Cycle-1 sector are dropped, duplicates counted."""
    toy = pd.DataFrame({"sec": ["[1, 2, 28]", "[1, 2]", "[30, 31]", "[13]", "[1, 13, 40]"]})
    lib = cov.build_library(toy).set_index("pattern")
    assert lib.loc["1-2", "n_stars"] == 2
    assert lib.loc["13", "n_stars"] == 1
    assert lib.loc["1-13", "class"] == "gapped-2"
    assert lib["n_stars"].sum() == 4


def test_library_class_column_consistent(library):
    """Stored class / n_sectors / consecutive agree with the functions that define them."""
    for _, r in library.iterrows():
        pat = cov.parse_pattern(r["pattern"])
        assert r["class"] == cov.coverage_class(pat)
        assert r["n_sectors"] == len(pat)
        assert bool(r["consecutive"]) == cov.is_consecutive(pat)


# ---------------------------------------------------------------- patterns and classes

@pytest.mark.parametrize("pat,cls", [
    ((5,), "single"), ((5, 6), "consecutive-2"), ((5, 7), "gapped-2"), ((1, 13), "gapped-2"),
    ((1, 2, 3), "consecutive-3-5"), ((1, 5, 9, 12, 13), "gapped-3-5"), ((4, 5, 6, 7, 8), "consecutive-3-5"),
    (tuple(range(1, 7)), "consecutive-6-12"), ((1, 2, 3, 4, 5, 7), "gapped-6-12"),
    (tuple(range(1, 13)), "consecutive-6-12"), (tuple(range(1, 14)), "full"),
])
def test_coverage_class(pat, cls):
    """Class boundaries as documented in coverage.coverage_class."""
    assert cov.coverage_class(pat) == cls


def test_pattern_roundtrip_and_validation():
    """format/parse round-trip in all accepted spellings; bad patterns raise."""
    for text in ("1-2-13", "1,2,13", "1 2 13", "[1, 2, 13]", "13-2-1-2"):
        assert cov.parse_pattern(text) == (1, 2, 13)
    assert cov.format_pattern([13, 1, 2]) == "1-2-13"
    with pytest.raises(ValueError):
        cov.normalize_pattern([0, 1])
    with pytest.raises(ValueError):
        cov.normalize_pattern([14])
    with pytest.raises(ValueError):
        cov.normalize_pattern([])


# ---------------------------------------------------------------- sampling

def test_sample_full_is_none():
    """'full' means no sector selection (legacy behavior)."""
    assert cov.sample_patterns("full", 4) == [None] * 4


def test_sample_reproducible_and_seed_dependent(library):
    """Same seed -> same draw; different seed -> different draw."""
    a = cov.sample_patterns("real-weighted", 50, seed=1, library=library)
    b = cov.sample_patterns("real-weighted", 50, seed=1, library=library)
    c = cov.sample_patterns("real-weighted", 50, seed=2, library=library)
    assert a == b and a != c


@pytest.mark.parametrize("cls", [c for c in cov.COVERAGE_CLASSES if c != "full"])
def test_sample_class_membership(library, cls):
    """Every pattern drawn for a class belongs to it."""
    for pat in cov.sample_patterns(cls, 200, seed=0, library=library):
        assert cov.coverage_class(pat) == cls


def test_sample_real_weighted_frequencies(library):
    """real-weighted reproduces the real class mix (74% single, 16% consecutive-2, ...)."""
    draws = cov.sample_patterns("real-weighted", 20000, seed=0, library=library)
    got = pd.Series([cov.coverage_class(p) for p in draws]).value_counts(normalize=True)
    want = library.groupby("class")["n_stars"].sum() / library["n_stars"].sum()
    for cls, frac in want.items():
        assert abs(got.get(cls, 0.0) - frac) < 0.015, cls


def test_sample_class_uniform_vs_frequency(cycle1):
    """Within Cycle-1 gapped-2, 'uniform' gives each of the 10 patterns ~10%; 'frequency' favors 1-13 (21/49)."""
    n = 20000
    uni = pd.Series([cov.format_pattern(p) for p in
                     cov.sample_patterns("gapped-2", n, seed=0, library=cycle1, weighting="uniform")]
                    ).value_counts(normalize=True)
    assert len(uni) == 10 and (uni - 0.1).abs().max() < 0.02
    freq = pd.Series([cov.format_pattern(p) for p in
                      cov.sample_patterns("gapped-2", n, seed=0, library=cycle1, weighting="frequency")])
    assert abs((freq == "1-13").mean() - 21 / 49) < 0.02


@pytest.mark.parametrize("group", list(cov.CLASS_GROUPS))
def test_sample_class_group(library, group):
    """'3-5' / '6-12' draw from both the consecutive and gapped patterns of that size, under either weighting."""
    for weighting in ("frequency", "uniform"):
        draws = cov.sample_patterns(group, 4000, seed=0, library=library, weighting=weighting)
        classes = {cov.coverage_class(p) for p in draws}
        assert classes == set(cov.CLASS_GROUPS[group])


def test_sample_default_weighting_is_frequency(library):
    """Default within-class weighting is by star-season count (Rae, 2026-10-15)."""
    n = 20000
    sub = library[library["class"] == "gapped-2"]
    want = sub.set_index("pattern")["n_stars"] / sub["n_stars"].sum()
    got = pd.Series([cov.format_pattern(p) for p in cov.sample_patterns("gapped-2", n, seed=0, library=library)]
                    ).value_counts(normalize=True)
    top = want.idxmax()
    assert abs(got[top] - want[top]) < 0.02


def test_sample_explicit_pattern_and_errors(library):
    """pattern:<p> repeats p; unknown specs and weightings raise."""
    assert cov.sample_patterns("pattern:1-13", 3) == [(1, 13)] * 3
    with pytest.raises(ValueError):
        cov.sample_patterns("gapped", 3, library=library)
    with pytest.raises(ValueError):
        cov.sample_patterns("single", 3, library=library, weighting="bogus")


# ---------------------------------------------------------------- descriptors

def test_descriptors_toy():
    """Hand-checkable case: two 10-d stretches separated by a 20-d hole, 1-d cadence."""
    t = np.arange(0.0, 41.0)
    f = np.where((t <= 10) | (t >= 30), 0.0, np.nan)
    d = cov.coverage_descriptors(t, f, [1, 3])
    assert d["n_sectors"] == 2 and d["sectors"] == "1-3" and d["coverage_class"] == "gapped-2"
    assert d["total_baseline_days"] == 40.0
    assert d["longest_contiguous_days"] == 10.0
    assert d["largest_gap_days"] == 20.0
    np.testing.assert_allclose(d["duty_cycle"], 22 / 41)


def test_descriptors_short_gaps_do_not_split():
    """Orbit gaps (~1 d) don't end a contiguous stretch; a missing sector does."""
    lc = make_lightcurve(3.0, n_sectors=13, seed=1, sectors=[4, 5, 7])
    d = cov.coverage_descriptors(lc.time, lc.flux, [4, 5, 7])
    assert 1.0 < d["largest_gap_days"] < 30.0
    np.testing.assert_allclose(d["longest_contiguous_days"], 2 * 27.4 - 1.0, atol=0.05)
    np.testing.assert_allclose(d["total_baseline_days"], 4 * 27.4 - 1.0, atol=0.05)  # sectors 4..7


def test_descriptors_degenerate():
    """No or one finite point -> zeros rather than an exception."""
    d = cov.coverage_descriptors(np.arange(3.0), np.full(3, np.nan), [1])
    assert d["total_baseline_days"] == 0.0 and d["duty_cycle"] == 0.0


# ---------------------------------------------------------------- synthetic sectors=

def _orbit_segments(lc_full, n_sectors=13, sector_days=27.4):
    """Split a full-coverage synthetic light curve into per-orbit segments plus
    the diagnostics dict recombine_segments expects (gap points = NaN cadences)."""
    t, _, starts = tess_like_time_grid(n_sectors=n_sectors, sector_days=sector_days)
    orbit = np.searchsorted(starts, t, side="right") - 1
    segs = [(lc_full.time[orbit == k], lc_full.flux[orbit == k]) for k in range(2 * n_sectors)]
    diag = {"gap_mask": ~np.isfinite(lc_full.flux), "segment_flagged": np.zeros(2 * n_sectors, bool)}
    return segs, diag


@pytest.mark.parametrize("pattern", [[1], [7], [13], [3, 4], [5, 7], [1, 13], [1, 2, 13], [2, 6, 9, 10, 11]])
def test_synthetic_sectors_match_load_smarts_fits_conventions(pattern):
    """sectors=p gives the same time grid and NaN structure as load_smarts_fits's
    recombine_segments(drop_gap_points=True, sectors=p) + to_even_cadence, applied
    to the same star at full coverage; flux agrees up to the re-centering constant."""
    full = make_lightcurve(3.0, n_sectors=13, seed=4)
    sub = make_lightcurve(3.0, n_sectors=13, seed=4, sectors=pattern)
    segs, diag = _orbit_segments(full)
    tt, ff, used = recombine_segments(segs, diag, drop_gap_points=True, sectors=pattern)
    te, fe = to_even_cadence(tt, ff)
    assert used == sorted(pattern) == sub.params["sectors"]
    assert len(te) == len(sub.time)
    np.testing.assert_allclose(sub.time, te, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(np.isfinite(sub.flux), np.isfinite(fe))
    ok = np.isfinite(fe)
    np.testing.assert_allclose(sub.flux[ok], fe[ok] - fe[ok].mean(), rtol=0, atol=1e-15)
    assert abs(np.nanmean(sub.flux)) < 1e-12


def test_synthetic_sectors_through_fits_file(tmp_path):
    """End-to-end on a fake SMARTS FITS (gaps linearly interpolated, as SMARTS ships):
    load_smarts_fits(sectors=p) and make_lightcurve(sectors=p) agree on the kept
    sectors, the start time, the grid length and (to within gap-detection
    heuristics) the NaN structure."""
    from astropy.io import fits
    from preprocessing import load_smarts_fits
    full = make_lightcurve(3.0, n_sectors=13, seed=4, white_noise=0.002)
    last = np.flatnonzero(np.isfinite(full.flux))[-1]
    t, f = full.time[:last + 1], full.flux[:last + 1].copy()
    ok = np.isfinite(f)
    f[~ok] = np.interp(t[~ok], t[ok], f[ok])
    hdr = fits.Header()
    hdr["PERIOD"] = 3.0
    path = tmp_path / "fake_smarts.fits"
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.BinTableHDU.from_columns([fits.Column("time", "D", array=t),
                                                 fits.Column("flux", "D", array=f + 1.0)],
                                                name="LIGHTCURVE")]).writeto(path)
    for pattern in ([1, 2, 13], [3, 7]):
        pre = load_smarts_fits(path, sectors=pattern)
        sub = make_lightcurve(3.0, n_sectors=13, seed=4, white_noise=0.002, sectors=pattern)
        assert pre.sectors_used == pattern
        assert abs(pre.time[0] - sub.time[0]) < 0.1
        assert abs(len(pre.time) - len(sub.time)) <= 50
        n = min(len(pre.time), len(sub.time))
        assert np.mean(np.isfinite(pre.flux[:n]) != np.isfinite(sub.flux[:n])) < 0.01


def test_synthetic_sectors_subset_of_full_star():
    """A patchy light curve is the same star: identical flux (up to a constant) on kept cadences."""
    full = make_lightcurve(5.0, n_sectors=13, seed=9, red_noise_sigma=0.001, ramp_amplitude=0.002,
                           spot_lifetime_days=20.0)
    sub = make_lightcurve(5.0, n_sectors=13, seed=9, red_noise_sigma=0.001, ramp_amplitude=0.002,
                          spot_lifetime_days=20.0, sectors=[2, 8])
    i0 = int(np.flatnonzero(np.isclose(full.time, sub.time[0]))[0])
    seg = full.flux[i0:i0 + len(sub.time)]
    ok = np.isfinite(sub.flux)
    assert np.all(np.isfinite(seg[ok]))
    diff = seg[ok] - sub.flux[ok]
    np.testing.assert_allclose(diff, diff[0], rtol=0, atol=1e-15)


def test_synthetic_sectors_validation():
    """Out-of-range or empty sector lists raise."""
    with pytest.raises(ValueError):
        make_lightcurve(3.0, n_sectors=13, sectors=[14])
    with pytest.raises(ValueError):
        make_lightcurve(3.0, n_sectors=2, sectors=[3])
    with pytest.raises(ValueError):
        make_lightcurve(3.0, n_sectors=13, sectors=[])


def test_make_suite_sectors_keeps_population():
    """Coverage doesn't change the star population: every non-coverage param is identical."""
    pats = cov.sample_patterns("real-weighted", 6, seed=3)
    base = make_suite(6, seed=0)
    patchy = make_suite(6, seed=0, sectors=pats)
    one = make_suite(6, seed=0, sectors=[5])
    for b, p, o, pat in zip(base, patchy, one, pats):
        assert b.star_id == p.star_id
        for k in ("period", "amplitude", "shear", "minlat_deg", "maxlat_deg", "inclination_deg", "seed"):
            assert b.params[k] == p.params[k] == o.params[k]
        assert p.params["sectors"] == list(pat)
        assert o.params["sectors"] == [5]
        assert "sectors" not in b.params
    with pytest.raises(ValueError):
        make_suite(6, seed=0, sectors=pats[:3])


# ---------------------------------------------------------------- default bit-identity

# sha256 over (time, flux) bytes of make_suite(5, seed=0), computed with the
# synthetic.py at main 589cb8d (before T16a). Pins the default output exactly.
DEFAULT_SUITE_SHA256 = "b9cb395678b35c2f9ed6d9a66b48be42850221d03f60e7f6c724f1963c0e7585"


@pytest.mark.skipif(platform.machine() != "x86_64",
                    reason="digest recorded on x86_64; libm cos/exp may differ in the last bit elsewhere")
def test_default_suite_bit_identical_to_pre_t16a():
    """sectors=None (default) leaves make_suite's output bit-identical to pre-T16a code."""
    h = hashlib.sha256()
    for lc in make_suite(5, seed=0):
        h.update(lc.time.tobytes())
        h.update(lc.flux.tobytes())
    assert h.hexdigest() == DEFAULT_SUITE_SHA256


def test_default_suite_matches_baseline_population():
    """The default suite reproduces the baseline per_star.csv star parameters exactly."""
    base = pd.read_csv(ROOT / "benchmarks/baselines/synth200_s0_tables/per_star.csv",
                       float_precision="round_trip").set_index("star_id")
    for lc in make_suite(10, seed=0):
        row = base.loc[lc.star_id]
        assert row["true_period"] == lc.true_period
        for k in ("amplitude", "shear", "minlat_deg", "maxlat_deg", "inclination_deg"):
            assert row[f"meta_{k}"] == lc.params[k]


@pytest.mark.slow
def test_default_scorecard_rows_bit_identical_to_baseline():
    """score_one_star on default (full) synthetic stars reproduces the committed
    baseline's final periods and outcomes exactly (3 stars; the full 200-star
    check is in the T16a report via tools/compare_runs.py)."""
    import determinism
    if determinism.blas_backend() != "openblas":
        pytest.xfail("baseline was produced under OpenBLAS (ss_openblas); other BLAS differ in the last bits")
    import scorecard
    from threadpoolctl import threadpool_limits  # 1 BLAS thread, as in the scorecard workers (T14)
    base = pd.read_csv(ROOT / "benchmarks/baselines/synth200_s0_tables/per_star.csv",
                       float_precision="round_trip").set_index("star_id")
    args = types.SimpleNamespace(synthetic=3, seed=0, coverage="full")
    for task in scorecard.build_tasks(args):
        with threadpool_limits(1):
            rec = scorecard.score_one_star(task, scorecard.DEFAULT_PIPELINE_METHODS, cov_rel_tol())
        row = base.loc[rec["star_id"]]
        assert rec["final_outcome"] == row["final_outcome"]
        if pd.isna(row["final_period"]):
            assert rec["final_period"] is None
        else:
            assert rec["final_period"] == row["final_period"]
        assert rec["coverage_class"] == "full" and rec["n_sectors"] == 13


def cov_rel_tol():
    """The project REL_TOL (imported, never hardcoded)."""
    from evaluation import REL_TOL
    return REL_TOL


# ---------------------------------------------------------------- scorecard plumbing

def _args(**kw):
    """Minimal argparse-like namespace for scorecard helpers."""
    d = dict(synthetic=None, manifest=None, seed=0, coverage="full", coverage_seed=0,
             coverage_weighting="frequency", coverage_library=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_dataset_string():
    """Full coverage keeps the v2 dataset string (baselines/goals still apply); others record the spec."""
    import scorecard
    assert scorecard.dataset_string(_args(synthetic=200)) == "synthetic:n=200,seed=0"
    assert (scorecard.dataset_string(_args(synthetic=100, coverage="single", coverage_seed=3))
            == "synthetic:n=100,seed=0,coverage=single,coverage_weighting=frequency,coverage_seed=3,"
               "coverage_library=all_cycles")
    assert (scorecard.dataset_string(_args(synthetic=100, coverage="gapped-2", coverage_weighting="uniform",
                                           coverage_library=cov.LIBRARY_DIR / "cycle2_patterns.csv"))
            == "synthetic:n=100,seed=0,coverage=gapped-2,coverage_weighting=uniform,coverage_seed=0,"
               "coverage_library=cycle2")
    assert (scorecard.dataset_string(_args(synthetic=100, coverage="real-weighted"))
            == "synthetic:n=100,seed=0,coverage=real-weighted,coverage_seed=0,coverage_library=all_cycles")
    assert (scorecard.dataset_string(_args(synthetic=10, coverage="pattern:1-13"))
            == "synthetic:n=10,seed=0,coverage=pattern:1-13")


def test_build_tasks_synthetic_coverage():
    """--coverage applies a sampled pattern per synthetic star, same draw as sample_patterns."""
    import scorecard
    tasks = scorecard.build_tasks(_args(synthetic=8, coverage="consecutive-2", coverage_seed=5))
    want = cov.sample_patterns("consecutive-2", 8, seed=5)
    assert [tuple(t["lc"].params["sectors"]) for t in tasks] == want


def test_build_tasks_manifest_coverage(tmp_path):
    """FITS manifests get load_kwargs['sectors'] from --coverage or a per-row `sectors` column."""
    import scorecard
    man = tmp_path / "m.csv"
    pd.DataFrame({"path": ["a.fits", "b.fits", "c.fits"]}).to_csv(man, index=False)
    full = scorecard.build_tasks(_args(manifest=man))
    assert all(t["load_kwargs"] == {} for t in full)
    single = scorecard.build_tasks(_args(manifest=man, coverage="single", coverage_seed=1))
    assert [tuple(t["load_kwargs"]["sectors"]) for t in single] == cov.sample_patterns("single", 3, seed=1)

    man2 = tmp_path / "m2.csv"
    pd.DataFrame({"path": ["a.fits", "b.fits", "c.fits"], "sectors": ["1-13", "", "4"]}).to_csv(man2, index=False)
    rows = scorecard.build_tasks(_args(manifest=man2))
    assert [t["load_kwargs"] for t in rows] == [{"sectors": [1, 13]}, {}, {"sectors": [4]}]
    with pytest.raises(SystemExit):
        scorecard.build_tasks(_args(manifest=man2, coverage="single"))


@pytest.mark.slow
def test_scorecard_fits_manifest_with_coverage(tmp_path):
    """The real-data path end to end, without real data: a fake SMARTS FITS in a
    manifest, run with --coverage pattern:..., records the pattern and descriptors."""
    import json
    from astropy.io import fits
    import scorecard
    lc = make_lightcurve(3.0, amplitude=0.01, white_noise=0.001, n_sectors=13, seed=4)
    last = np.flatnonzero(np.isfinite(lc.flux))[-1]
    t, f = lc.time[:last + 1], lc.flux[:last + 1].copy()
    ok = np.isfinite(f)
    f[~ok] = np.interp(t[~ok], t[ok], f[ok])
    hdr = fits.Header()
    hdr.update(PERIOD=3.0, DIFFROT=0.0, MINLAT=0.0, MAXLAT=50.0)
    path = tmp_path / "fake.fits"
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.BinTableHDU.from_columns([fits.Column("time", "D", array=t),
                                                 fits.Column("flux", "D", array=f + 1.0)],
                                                name="LIGHTCURVE")]).writeto(path)
    man = tmp_path / "man.csv"
    pd.DataFrame({"path": [str(path)], "star_id": ["fake"]}).to_csv(man, index=False)
    out = tmp_path / "run"
    scorecard.main(["--manifest", str(man), "--coverage", "pattern:3-4", "--n-workers", "1",
                    "--outdir", str(out)])
    ps = pd.read_csv(out / "per_star.csv")
    assert ps.loc[0, "sectors"] == "3-4" and ps.loc[0, "coverage_class"] == "consecutive-2"
    assert ps.loc[0, "meta_sectors_requested"] == "3-4"
    assert ps.loc[0, "total_baseline_days"] < 2 * 27.4
    m = json.loads((out / "metrics.json").read_text())
    assert m["config"]["dataset"].endswith(",coverage=pattern:3-4")
    assert m["config"]["scorecard_version"] == 3
    assert ps.loc[0, "final_outcome"] in ("correct", "diffrot")
