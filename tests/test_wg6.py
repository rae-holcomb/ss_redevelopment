"""WG6 loader (tiny FITS-in-zip fixture, no drive), manifest builder, committed manifest, scorecard dataset strings."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import wg6  # noqa: E402

N_SEASON, DT = 200, 12.1 / 1440  # tiny seasons: 200 points at the Roman cadence
P_SIG = 0.5


def _fits_bytes(with_jd=True, nan_idx=(5, 7), n_seasons=3):
    """A WG6-like FITS (improved layout): F146 with JD/mag/mag_error over `n_seasons` seasons 100 d apart."""
    jd = np.concatenate([2461631.5 + s * 100.0 + np.arange(N_SEASON) * DT for s in range(n_seasons)])
    mag = 18.0 - 2.5 * np.log10(1.0 + 0.05 * np.sin(2 * np.pi * (jd - jd[0]) / P_SIG))
    mag[list(nan_idx)] = np.nan
    err = np.full_like(mag, 0.003)
    cols = ([fits.Column(name="JD", format="D", array=jd)] if with_jd else []) + [
        fits.Column(name="mag", format="D", array=mag), fits.Column(name="mag_error", format="D", array=err)]
    prim = fits.PrimaryHDU()
    prim.header["NAME"], prim.header["VARTYPE"] = "OGLE-BLG-TEST-0001", "rrlyr"
    h = fits.HDUList([prim, fits.BinTableHDU.from_columns(cols, name="F146"),
                      fits.BinTableHDU.from_columns(cols, name="F087")])
    b = io.BytesIO()
    h.writeto(b)
    return b.getvalue(), jd


@pytest.fixture()
def wg6_zip(tmp_path):
    raw, jd = _fits_bytes()
    z = tmp_path / "RGES_filters_TEST_lightcurves_final.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dir/star.fits", raw)
    return z, jd


def test_load_season_from_zip(wg6_zip):
    z, jd = wg6_zip
    lc2 = wg6.load_wg6_roman(z, "dir/star.fits", season=2)
    assert len(lc2.time) == N_SEASON and lc2.meta["jd0"] == pytest.approx(jd[N_SEASON])
    lc = wg6.load_wg6_roman(z, "dir/star.fits", season=1, true_period=P_SIG)
    assert len(lc.time) == N_SEASON and lc.time[0] == 0.0
    assert np.allclose(np.diff(lc.time), DT, atol=1e-9)           # uniform grid, days from season start
    assert lc.true_period == P_SIG and lc.sectors_used == [1]
    assert lc.meta["name"] == "OGLE-BLG-TEST-0001" and lc.meta["n_seasons_in_file"] == 3
    assert lc.meta["jd0"] == pytest.approx(jd[0])
    assert np.isnan(lc.flux[5]) and np.isnan(lc.flux[7]) and lc.meta["n_nan"] == 2   # NaN slots stay in place


def test_flux_is_zero_centred_relative_flux_with_sigma(wg6_zip):
    z, _ = wg6_zip
    lc = wg6.load_wg6_roman(z, "dir/star.fits", season=1)
    assert np.nanmedian(lc.flux) == pytest.approx(0.0, abs=5e-3)
    assert np.nanmax(lc.flux) == pytest.approx(0.045, rel=0.15)
    assert np.allclose(lc.flux_err[np.isfinite(lc.flux)], np.log(10) / 2.5 * 0.003, rtol=0.06)   # sigma_flux ~ ln10/2.5 sigma_mag
    # the injected period is recoverable
    t, f = lc.time[10:], lc.flux[10:]
    freq = np.linspace(0.2, 10, 4000)
    pw = [abs(np.sum(f * np.exp(-2j * np.pi * fr * t))) for fr in freq]
    assert 1 / freq[int(np.argmax(pw))] == pytest.approx(P_SIG, rel=0.03)


def test_all_seasons_and_errors(wg6_zip):
    z, _ = wg6_zip
    lc = wg6.load_wg6_roman(z, "dir/star.fits")
    assert len(lc.time) == 3 * N_SEASON and lc.sectors_used == [1, 2, 3]
    with pytest.raises(ValueError, match="member"):
        wg6.load_wg6_roman(z)
    with pytest.raises(ValueError, match="season"):
        wg6.load_wg6_roman(z, "dir/star.fits", season=4)
    with pytest.raises(ValueError, match="band"):
        wg6.load_wg6_roman(z, "dir/star.fits", band="F999")


def test_plain_fits_path_and_simple_noise_times_dir(tmp_path):
    raw, jd = _fits_bytes(with_jd=False)
    f = tmp_path / "simple.fits"
    f.write_bytes(raw)
    with pytest.raises(ValueError, match="times_dir"):
        wg6.load_wg6_roman(f)
    np.save(tmp_path / "roman_times_shortcadence.npy", jd)
    lc = wg6.load_wg6_roman(f, times_dir=tmp_path, season=3)
    assert len(lc.time) == N_SEASON


def test_relative_path_resolved_against_root(wg6_zip, monkeypatch):
    z, _ = wg6_zip
    monkeypatch.setenv("WG6_RESAMPLED_ROOT", str(z.parent))
    lc = wg6.load_wg6_roman(z.name, "dir/star.fits", season=1)
    assert len(lc.time) == N_SEASON


def test_mag_to_flux_nan_safe():
    f, e = wg6.mag_to_flux(np.array([18.0, np.nan, 18.0, 17.0]), np.full(4, 0.01))
    assert np.nanmedian(f) == pytest.approx(1.0) and np.isnan(f[1]) and np.isnan(e[1])
    f, e = wg6.mag_to_flux(np.full(3, np.nan))
    assert np.isnan(f).all() and e is None


def test_load_wg6_raw_hjd_fix(tmp_path):
    t = np.array([4.95e6 + 0.1 * i for i in range(30)] + [2455000.0])  # doubled-offset Cepheid HJDs + one normal
    mag = 16.0 + 0.1 * np.sin(np.arange(31))
    cols = [fits.Column(name="HJD", format="D", array=t), fits.Column(name="mag", format="D", array=mag),
            fits.Column(name="mag_error", format="D", array=np.full(31, 0.01))]
    prim = fits.PrimaryHDU()
    prim.header["NAME"] = "OGLE-BLG-CEP-0001"
    p = tmp_path / "raw.fits"
    fits.HDUList([prim, fits.BinTableHDU.from_columns(cols, name="LC_I")]).writeto(p)
    lc = wg6.load_wg6_raw(p)
    assert lc.meta["jd0"] == pytest.approx(2455000.0) and len(lc.time) == 31 and lc.time[-1] < 3e6
    assert np.nanmedian(lc.flux) == pytest.approx(0.0, abs=0.02)


# --- manifest builder -----------------------------------------------------------------------------------------------

def _toy_inventory_truth():
    inv, tr = [], []
    for i in range(40):
        for cls in ("ECL", "ELL", "T2CEP", "RRLYR", "HB", "DSCT", "LPV"):
            sid = f"OGLE-BLG-{cls}-{i:04d}"
            inv.append(dict(variant="improved", cls=cls, zip="z.zip", member=f"{sid}.fits", star_id=sid, cv_ind=np.nan,
                            fl_template=np.nan, fl_k=np.nan, wfmag=np.nan))
            tr.append(dict(star_id=sid, cls=cls, P_true=1.0 + i, p_convention="orbital period", subtype="x", in_raw=True))
    for o in range(60):
        tmpl = "flat" if o < 10 else str(1000 + o % 10)
        for wf in (17, 20, 21):
            sid = f"FL_{o}"
            inv.append(dict(variant="improved", cls="FL", zip="fl.zip", member=f"{sid}_{wf}.fits", star_id=sid, cv_ind=np.nan,
                            fl_template=tmpl, fl_k=o, wfmag=wf))
        tr.append(dict(star_id=f"FL_{o}", cls="FL", P_true=np.nan if tmpl == "flat" else 2.0 + o % 10,
                       p_convention="rotation period", subtype="flare", in_raw=tmpl != "flat"))
    return pd.DataFrame(inv), pd.DataFrame(tr)


def test_build_manifest_schema_split_and_eb_flags():
    import build_wg6_manifest as b
    inv, tr = _toy_inventory_truth()
    man = b.build(inv, tr, n_per_class=30, n_stress=20, fl_mags=(17, 20), seasons=(1, 2, 3))
    assert list(man.columns) == b.COLUMNS and man.star_id.is_unique
    assert set(man.season) == {1, 2, 3} and set(man.eval_set) == {"dev", "holdout"} and set(man.variant) == {"improved"}
    b.check_split(man)                                                       # no star / template on both sides
    fl = man[man.cls == "FL"]
    assert set(fl.wfmag.astype(int)) == {17, 20}
    assert fl[fl.fl_template == "flat"].is_rotator.eq(False).all() and fl[fl.fl_template != "flat"].is_rotator.all()
    ecl = man[man.cls == "ECL"]
    assert ecl.is_eb.all() and np.allclose(ecl.P_phot_expected, ecl.P_true / 2)
    assert np.allclose(man[man.cls == "T2CEP"].P_phot_expected, man[man.cls == "T2CEP"].P_true) and not man[man.cls == "T2CEP"].is_eb.any()
    assert man[man.cls == "ELL"].obj_id.nunique() == 30
    # deterministic
    again = b.build(inv.sample(frac=1, random_state=3), tr, n_per_class=30, n_stress=20)   # row order must not matter
    pd.testing.assert_frame_equal(again, man)


def test_split_check_catches_leak():
    import build_wg6_manifest as b
    m = pd.DataFrame(dict(obj_id=["a", "a"], fl_template=["t", "t"], eval_set=["dev", "holdout"]))
    with pytest.raises(AssertionError):
        b.check_split(m)


# --- committed manifest + scorecard dataset strings (no drive needed) ----------------------------------------------------

MAN = ROOT / "datasets" / "wg6_roman_v0.csv"


def test_committed_manifest_is_consistent():
    import build_wg6_manifest as b
    man = pd.read_csv(MAN, low_memory=False)
    assert list(man.columns) == b.COLUMNS and man.star_id.is_unique
    assert set(man.variant) == {"improved"} and set(man.season) == {1, 2, 3}
    b.check_split(man)
    assert (man.beyond_reach.astype(bool) == (man.P_true > 35.5)).all()
    assert man[man.is_eb.astype(bool)].cls.isin(["ECL", "ELL"]).all()
    assert (man.is_rotator.astype(bool) <= (man.cls == "FL")).all()          # v0 rotators are FL only
    n = man.groupby("cls").obj_id.nunique()
    assert n[["ECL", "ELL", "RRLYR"]].eq(300).all() and n["T2CEP"] == 100      # T2CEP has only 100 improved stars


def test_scorecard_wg6_dataset_strings():
    import scorecard as sc
    assert sc.parse_wg6_dataset("wg6r_fl_v0_improved_s1") == ("fl", 1)
    assert sc.parse_wg6_dataset(Path("wg6r_contam_v0_improved_s3")) == ("contam", 3)
    for bad in ("wg6r_fl_v0_improved_s4", "wg6r_rot_v0_improved_s1", "datasets/dev_small.csv"):
        assert sc.parse_wg6_dataset(bad) is None
    fl = sc.build_wg6_tasks("wg6r_fl_v0_improved_s2")
    co = sc.build_wg6_tasks("wg6r_contam_v0_improved_s2")
    man = pd.read_csv(MAN, low_memory=False)
    assert len(fl) == ((man.cls == "FL") & man.is_rotator & ~man.beyond_reach & (man.eval_set == "dev") & (man.season == 2)).sum()
    assert all(t["kind"] == "wg6r" and t["season"] == 2 and t["true_period"] > 0 for t in fl + co)
    assert all(not t["meta"]["is_rotator"] for t in co) and {"ECL", "ELL"} <= {t["meta"]["cls"] for t in co}
    assert not ({t["star_id"] for t in fl} & {t["star_id"] for t in co})
    holdout = set(man[man.eval_set == "holdout"].star_id)
    assert not ({t["star_id"] for t in fl + co} & holdout)                   # never selects the holdout
    be = sc.build_wg6_tasks("wg6r_beyond_v0_improved_s2")
    assert be and all(t["meta"]["P_true"] > 35.5 for t in be)
    assert not ({t["star_id"] for t in be} & {t["star_id"] for t in fl + co})          # disjoint from the main sets
    assert all(t["meta"]["P_true"] <= 35.5 for t in fl + co)
    assert {"HB", "LPV"} <= {t["meta"]["cls"] for t in be}
    assert len(fl) + len([t for t in be if t["meta"]["is_rotator"]]) == ((man.cls == "FL") & man.is_rotator & (man.eval_set == "dev") & (man.season == 2)).sum()
    ell = next(t for t in co if t["meta"]["cls"] == "ELL")
    assert ell["true_period"] == pytest.approx(ell["meta"]["P_true"] / 2)    # EBs scored at P_orb/2


def test_manifest_members_exist_on_drive():
    from wg6 import resampled_root
    root = resampled_root()
    if not root.is_dir():
        pytest.skip("Data_Vault drive (WG6_RESAMPLED_ROOT) not mounted")
    man = pd.read_csv(MAN, low_memory=False)
    sub = man.drop_duplicates(["zip", "member"]).sample(5, random_state=0)
    for r in sub.itertuples():
        lc = wg6.load_wg6_roman(root / r.zip, r.member, season=r.season)
        assert 8000 < len(lc.time) < 9000
