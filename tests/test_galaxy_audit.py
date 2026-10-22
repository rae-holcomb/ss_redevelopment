"""Tests for the T20 galaxy-ID tools (tools/audit/galaxy_pass.py, galaxy_groups.py); no drive needed."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "audit"))
import galaxy_groups as gg  # noqa: E402
import galaxy_pass as gp  # noqa: E402
from lc_spotcheck import interp_runs  # noqa: E402

T_GRID = (np.arange(gp.N_CAD) / 48.0).astype(np.float32)


def _galaxy(rng, sigma=7e-4):
    """Fake galaxy noise light curve with two linearly filled gaps and a constant end fill."""
    g = 1 + sigma * rng.standard_normal(gp.N_CAD) + 2e-3 * np.sin(T_GRID / 20.0)
    for a, b in ((3000, 3100), (9000, 9050)):
        g[a:b] = np.linspace(g[a - 1], g[b], b - a + 2)[1:-1]
    g[-380:] = g[-381]
    return g


def _fits_bytes(flux):
    """Bytes of a FITS file laid out like a SMARTS-TESS DR1 file."""
    prim = fits.PrimaryHDU()
    prim.header["PERIOD"] = 1.0
    lc = fits.BinTableHDU.from_columns(
        [fits.Column("time", "E", array=T_GRID), fits.Column("flux", "E", array=flux.astype(np.float32))],
        name="LIGHTCURVE")
    wav = fits.ImageHDU(np.zeros((64, 64), np.uint8), name="WAVELET")
    buf = io.BytesIO()
    fits.HDUList([prim, lc, wav]).writeto(buf)
    return buf.getvalue()


def test_interp_diff_mask_matches_t1_rule():
    """The vectorized mask equals T1's loop: diffs max(s-1, 0)..e for every interp run (s, e)."""
    f = _galaxy(np.random.default_rng(0)).astype(np.float32).astype(float)
    ref = np.zeros(len(f) - 1, bool)
    for s, e in interp_runs(f, min_run=3):
        ref[max(s - 1, 0):e + 1] = True
    assert ref.sum() > 400
    assert np.array_equal(gp.interp_diff_mask(f), ref)


def test_parse_lightcurve_fast_path_and_fallback():
    """Fast fixed-offset parse equals astropy; a non-standard layout falls back to astropy."""
    flux = _galaxy(np.random.default_rng(1))
    b = _fits_bytes(flux)
    assert len(b) == gp.FILE_SIZE
    t, f = gp.parse_lightcurve(b)
    assert np.array_equal(f, flux.astype(np.float32)) and np.array_equal(t, T_GRID)
    t2, f2 = gp.parse_lightcurve(b + b"\0" * 2880)  # wrong size -> astropy path
    assert np.array_equal(f2, f) and np.array_equal(t2, t)


def test_file_record_flags_corrupt_orbit():
    """A garbage orbit is flagged with the right orbit number; fills are zeroed in the fingerprint."""
    f = _galaxy(np.random.default_rng(2))
    orbit = gp.orbit_index(T_GRID)
    f[orbit == 5] = 3.0 + np.arange((orbit == 5).sum()) * 1e-3
    rec, fp = gp.file_record(f.astype(np.float32), orbit)
    assert rec["orbits_hit"] == "5" and rec["max_abs_dev"] > 2
    assert rec["trail_const"] == 381 and fp.dtype == np.int8 and len(fp) == gp.N_FP
    assert np.all(fp[3000:3100] == 0) and np.all(fp[-379:] == 0)
    assert 500 < rec["p2p_ppm"] < 900


def test_grouping_recovers_shared_noise_galaxies():
    """Files built from 6 galaxies + different sinusoids regroup exactly into 6 galaxies."""
    rng = np.random.default_rng(3)
    gals = [_galaxy(rng) for _ in range(6)]
    orbit = gp.orbit_index(T_GRID)
    truth, bundles = [], []
    for b in range(3):
        fps = []
        for _ in range(12):
            g = int(rng.integers(6))
            P, A = rng.uniform(0.2, 30), 10 ** rng.uniform(-4, -2.5)
            flux = gals[g] * (1 + A * np.sin(2 * np.pi * T_GRID / P + rng.uniform(0, 6.3)))
            fps.append(gp.file_record(flux.astype(np.float32), orbit)[1])
            truth.append(g)
        bundles.append(np.stack(fps))
    S, cnt = gg.build_templates(bundles, assign=0.6, link=0.9, log=lambda *a, **k: None)
    S, cnt, _ = gg.merge_templates(S, cnt, 0.97, 0.6)
    gid, _, c1, c2 = gg.assign_all(bundles, gg._normalize(S), 0.6)
    truth = np.array(truth)
    assert (gid >= 0).all() and c1.min() > 0.9 and c2.max() < 0.3
    # one-to-one mapping between found and true galaxies
    pairs = set(zip(gid.tolist(), truth.tolist()))
    assert len(pairs) == len(set(truth)) == len(set(gid.tolist()))


def test_scatter_sum_matches_add_at():
    """Sparse one-hot row sums equal np.add.at."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 7)).astype(np.float32)
    idx = rng.integers(0, 5, 50)
    ref = np.zeros((5, 7), np.float32)
    np.add.at(ref, idx, X)
    assert np.allclose(gg.scatter_sum(idx, X, 5), ref, atol=1e-5)


@pytest.mark.parametrize("n", [1, 2])
def test_best_two_small(n):
    """best_two handles 0, 1 and 2 templates."""
    C = np.arange(3 * n, dtype=np.float32).reshape(3, n)
    k, c1, c2 = gg.best_two(C)
    assert (k == n - 1).all() and np.allclose(c1, C[:, -1])
    assert np.allclose(c2, C[:, -2] if n == 2 else -1)
    k0, _, _ = gg.best_two(np.zeros((3, 0), np.float32))
    assert (k0 == -1).all()


def test_sibling_light_curves_stay_separate():
    """Two noise light curves correlated at ~0.92 get separate galaxy IDs and a sibling link,
    even when one of them appears only once per bundle."""
    rng = np.random.default_rng(5)
    a = _galaxy(rng)
    b = a.copy()
    sel = np.abs(np.diff(a, prepend=a[0])) > 0  # perturb real (non-fill) cadences only
    b[sel] += 3e-4 * rng.standard_normal(sel.sum())
    orbit = gp.orbit_index(T_GRID)
    bundles, truth = [], []
    for k in range(4):
        fps = []
        for g in [0] * 8 + [1]:
            base = a if g == 0 else b
            sig = 1e-4 * np.sin(2 * np.pi * T_GRID / rng.uniform(1, 50))
            fps.append(gp.file_record((base + sig).astype(np.float32), orbit)[1])
            truth.append(g)
        bundles.append(np.stack(fps))
    xa, xb = (gg.unit_rows(gp.file_record(x.astype(np.float32), orbit)[1][None]) for x in (a, b))
    assert 0.85 < (xa @ xb.T).item() < 0.96
    S, cnt = gg.build_templates(bundles, assign=0.6, link=0.97, log=lambda *a, **k: None)
    S, cnt, _ = gg.merge_templates(S, cnt, 0.97, 0.6)
    gid, _, _, _ = gg.assign_all(bundles, gg._normalize(S), 0.6)
    truth = np.array(truth)
    assert len(set(gid[truth == 0])) == 1 and len(set(gid[truth == 1])) == 1
    assert gid[truth == 0][0] != gid[truth == 1][0]
    sib = gg.sibling_pairs(gg._normalize(S), 0.3)
    assert len(sib) == 1 and sib["corr"].iloc[0] > 0.85


def test_injection_model_ratios_pick_the_right_model():
    """residual_ratios gives ~1 for the true model and far from 1 for the other."""
    import injection_model as im
    rng = np.random.default_rng(6)
    g = 1 + 5e-4 * rng.standard_normal(gp.N_CAD)
    g[5000:6000] = 20 + rng.standard_normal(1000)          # corrupt orbit
    s = 1 + 5e-3 * np.sin(2 * np.pi * T_GRID / 3.3)
    add = im.residual_ratios(g + (s - 1), g, 0.3)
    mul = im.residual_ratios(g * s, g, 0.3)
    assert abs(add["add_ratio"] - 1) < 0.2 and add["mul_ratio"] < 0.2
    assert abs(mul["mul_ratio"] - 1) < 0.2 and mul["add_ratio"] > 5


def test_stack_fits_roundtrip_through_loader(tmp_path):
    """A stack written by galaxy_periods.write_fits loads with load_smarts_fits."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_files"))
    import galaxy_periods as gper
    from preprocessing import load_smarts_fits
    f = _galaxy(np.random.default_rng(7))
    p = tmp_path / "x.fits"
    gper.write_fits(p, f, 123)
    lc = load_smarts_fits(p)
    assert lc.true_period == 1.0 and np.isfinite(lc.flux).sum() > 0.7 * gp.N_CAD


def test_robust_scatter_matches_sigma():
    """robust_scatter ~ sigma for Gaussian noise, unaffected by a few huge outliers."""
    import galaxy_stack as gs
    x = np.random.default_rng(8).standard_normal(20000)
    x[:50] = 1e6
    assert abs(gs.robust_scatter(x) - 1) < 0.05
