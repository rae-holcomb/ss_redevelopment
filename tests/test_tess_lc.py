"""SPOC/QLP loader on tiny synthetic FITS fixtures (no network, no drive), plus the frozen validation manifest."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

import tess_lc
from preprocessing import NoUsableSectorsError, PreprocessedLightCurve

ROOT = Path(__file__).resolve().parents[1]
P_SIG = 3.0
CAD = 1800 / 86400.0


def write_lc(path, sector, t0, n=600, cadence=CAD, qlp=False, level=1000.0, amp=0.02, flagged=(), nan_idx=(), pdc_scale=2.0):
    """SPOC-like (TIME, SAP_FLUX, PDCSAP_FLUX, QUALITY) or QLP-like (TIME, SAP_FLUX, KSPSAP_FLUX, QUALITY) light curve."""
    t = t0 + np.arange(n) * cadence
    sap = level * (1 + amp * np.sin(2 * np.pi * t / P_SIG))
    q = np.zeros(n, np.int32)
    for i in flagged:
        q[i] = 4  # bit 3 (value 4): in the default bitmask 175
    sap = sap.copy()
    for i in nan_idx:
        sap[i] = np.nan
    cols = [fits.Column(name="TIME", format="D", array=t), fits.Column(name="SAP_FLUX", format="D", array=sap)]
    if qlp:
        cols.append(fits.Column(name="KSPSAP_FLUX", format="D", array=np.ones(n)))
    else:
        cols.append(fits.Column(name="PDCSAP_FLUX", format="D", array=sap / pdc_scale))
    cols.append(fits.Column(name="QUALITY", format="J", array=q))
    prim = fits.PrimaryHDU()
    prim.header["SECTOR"] = sector
    prim.header["ORIGIN"] = "MIT/QLP" if qlp else "SPOC"
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    fits.HDUList([prim, hdu]).writeto(path, overwrite=True)
    return t


@pytest.fixture
def star(tmp_path):
    p1, p2 = tmp_path / "s14_lc.fits", tmp_path / "s15_lc.fits"
    write_lc(p1, 14, 1000.0, flagged=(10, 11), nan_idx=(20,), level=1000.0)
    write_lc(p2, 15, 1030.0, level=5000.0)  # other sector, different level: per-sector median normalisation
    return [p2, p1]  # deliberately out of order


def test_container_and_conventions(star):
    lc = tess_lc.load_tess_star(star, true_period=P_SIG, tic=123)
    assert isinstance(lc, PreprocessedLightCurve)
    assert lc.true_period == P_SIG and lc.sectors_used == [14, 15]
    assert lc.time[0] == 0.0 and np.allclose(np.diff(lc.time), CAD)
    assert lc.meta["t0_btjd"] == pytest.approx(1000.0)
    # zero-centred in each sector, whatever the raw level; the 30-day gap stays NaN
    assert abs(np.nanmedian(lc.flux[:600])) < 0.02 and abs(np.nanmedian(lc.flux[-600:])) < 0.02
    assert np.isnan(lc.flux[600:1440]).all()
    assert np.nanmax(lc.flux) == pytest.approx(0.02, abs=2e-3)


def test_quality_flags_and_nans_become_nan_slots(star):
    lc = tess_lc.load_tess_star(star)
    assert np.isnan(lc.flux[[10, 11, 20]]).all() and np.isfinite(lc.flux[[9, 12, 19, 21]]).all()
    assert lc.diagnostics["n_flagged"] == 2 and lc.diagnostics["n_nan"] == 1
    none = tess_lc.load_tess_star(star, quality="none")  # flags ignored (the NaN flux is still dropped)
    assert np.isfinite(none.flux[[10, 11]]).all() and np.isnan(none.flux[20])
    assert np.isnan(tess_lc.load_tess_star(star, quality=4).flux[10])  # explicit bitmask


def test_flux_column_choice(star):
    sap = tess_lc.load_tess_star(star, flux_column="SAP_FLUX")
    pdc = tess_lc.load_tess_star(star, flux_column="PDCSAP_FLUX")
    assert sap.meta["flux_column"] == "SAP_FLUX" and pdc.meta["flux_column"] == "PDCSAP_FLUX"
    # constant rescaling cancels in the relative flux, so identical here
    np.testing.assert_allclose(sap.flux, pdc.flux, equal_nan=True, atol=1e-12)


def test_short_window_columns_refused(star):
    for col in ("KSPSAP_FLUX", "cal_psf_flux", "CORR_FLUX"):
        with pytest.raises(ValueError, match="erases rotation"):
            tess_lc.load_tess_star(star, flux_column=col)
    with pytest.raises(ValueError, match="flux_column must be"):
        tess_lc.load_tess_star(star, flux_column="NOPE")


def test_qlp_default_mask_and_no_pdcsap(tmp_path):
    p = tmp_path / "hlsp_qlp_tess_ffi_s0014-0000000000000123_tess_v01_llc.fits"
    write_lc(p, 14, 2000.0, qlp=True, flagged=(5,))
    lc = tess_lc.load_tess_star([p])
    assert lc.meta["product"] == "qlp" and np.isnan(lc.flux[5]) and np.isfinite(lc.flux[6])
    with pytest.raises(ValueError, match="no PDCSAP_FLUX"):
        tess_lc.load_tess_star([p], flux_column="PDCSAP_FLUX")


def test_sector_selection_and_empty(star):
    lc = tess_lc.load_tess_star(star, sectors=[15])
    assert lc.sectors_used == [15] and lc.time.size == 600
    with pytest.raises(NoUsableSectorsError):
        tess_lc.load_tess_star(star, sectors=[99])


def test_ten_minute_sector_binned_to_30_min(tmp_path):
    p = tmp_path / "s30_lc.fits"
    write_lc(p, 30, 3000.0, n=1800, cadence=600 / 86400.0)
    lc = tess_lc.load_tess_star([p], timestep=1800)
    assert lc.time.size in (600, 601) and np.isfinite(lc.flux).all()  # the last bin may be a partial one
    assert np.allclose(np.diff(lc.time), CAD)


def test_mixed_products_rejected(tmp_path):
    a, b = tmp_path / "a_lc.fits", tmp_path / "b_qlp_llc.fits"
    write_lc(a, 14, 1000.0)
    write_lc(b, 15, 1030.0, qlp=True)
    with pytest.raises(ValueError, match="mixed products"):
        tess_lc.load_tess_star([a, b])


def test_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TESS_VALIDATION_ROOT", str(tmp_path))
    assert tess_lc.tess_validation_root() == tmp_path


def test_pipeline_runs_on_loaded_star(tmp_path):
    """The loaded grid is directly usable by the ACF code (NaN slots honoured)."""
    import acf_utils
    p = tmp_path / "s14_lc.fits"
    write_lc(p, 14, 1000.0, n=1200, amp=0.05)
    lc = tess_lc.load_tess_star([p])
    assert lc.flux.shape == lc.time.shape and np.isfinite(lc.flux).all()
    assert hasattr(acf_utils, "compute_acf")


# --- frozen manifest -------------------------------------------------------------------------------------------------

def test_manifest_reproduces_and_matches_decisions():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_tess_validation.py"), "--version", "v0", "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    m = pd.read_csv(ROOT / "datasets" / "tess_validation_v0.csv")
    assert len(m) == 480 and m.star_id.is_unique
    assert m.label.value_counts().to_dict() == {"rotator": 268, "nonrotator": 100, "contaminant_eb": 50,
                                                "expected_undetectable": 40, "nonrotator_candidate": 22}
    eu = m[m.label == "expected_undetectable"]
    assert (eu.gaia_variable == 0).all()
    assert m[m.label == "rotator"].truth_period.max() <= 71  # the 70-180 d block is skipped (Rae, 2026-11-19)


# --- downloader (fake fetch; no network) -----------------------------------------------------------------------------

def _dl():
    sys.path.insert(0, str(ROOT / "tools"))
    import download_tess_validation as d
    return d


def test_downloader_policy_resume_atomic(tmp_path):
    d = _dl()
    plan = pd.DataFrame([
        dict(star_id="TIC1", tic=1, product="spoc", sector=14, filename="a_lc.fits", relpath="spoc/TIC1/a_lc.fits", size_bytes=5, data_uri="u1"),
        dict(star_id="TIC1", tic=1, product="qlp", sector=14, filename="a_llc.fits", relpath="qlp/TIC1/a_llc.fits", size_bytes=5, data_uri="u2"),
        dict(star_id="TIC2", tic=2, product="qlp", sector=14, filename="b_llc.fits", relpath="qlp/TIC2/b_llc.fits", size_bytes=5, data_uri="u3")])
    fb = d.apply_qlp_policy(plan, "fallback")
    assert list(fb["relpath"]) == ["spoc/TIC1/a_lc.fits", "qlp/TIC2/b_llc.fits"]  # QLP only where SPOC is absent
    assert len(d.apply_qlp_policy(plan, "all")) == 3 and set(d.apply_qlp_policy(plan, "none")["product"]) == {"spoc"}
    calls = []

    def fetch(uri, dest):
        calls.append(uri)
        if uri == "u3" and calls.count("u3") < 2:
            raise RuntimeError("transient")
        Path(dest).write_bytes(b"12345")

    log = d.Log(tmp_path / "log.txt")
    res = d.download(fb, tmp_path, log, fetch=fetch, sleep=lambda s: None)
    assert list(res["status"]) == ["done", "done"] and calls.count("u3") == 2
    assert not list(tmp_path.rglob("*.part"))
    res2 = d.download(fb, tmp_path, log, fetch=fetch, sleep=lambda s: None)  # resume: nothing refetched
    assert list(res2["status"]) == ["skipped", "skipped"] and calls == ["u1", "u3", "u3"]
    assert "FAIL" in (tmp_path / "log.txt").read_text()


def test_downloader_mount_check(tmp_path):
    d = _dl()
    with pytest.raises(SystemExit, match="plug in the Data_Vault drive"):
        d.check_mount(Path("/Volumes/__no_such_drive__/tess_validation"))
    d.check_mount(tmp_path)  # not under /Volumes: fine


# --- scorecard dataset strings (task building and loading only; no pipeline run) ----------------------------------------

def test_scorecard_tessval_strings_and_tasks(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import scorecard as sc
    assert sc.parse_tessval_dataset("tessval_v1_sap") == ("SAP_FLUX", ("rotator",))
    assert sc.parse_tessval_dataset("tessval_v1_pdcsap_eb") == ("PDCSAP_FLUX", ("contaminant_eb",))
    assert sc.parse_tessval_dataset("tessval_v1_sap_neg")[1] == ("nonrotator", "nonrotator_candidate")
    assert sc.parse_tessval_dataset("tessval_v1_sap_eu")[1] == ("expected_undetectable",)
    assert sc.parse_tessval_dataset("tessval_v0_sap") is None and sc.parse_tessval_dataset("dev_small.csv") is None
    assert sc.parse_tessval_dataset("tessval_v1_sap_holdout") is None  # no dataset string selects the locked holdout
    man = pd.DataFrame(dict(star_id=["TIC1", "TIC2", "TIC3", "TIC4"], tic=[1, 2, 3, 4], label=["rotator"] * 4, source=["x"] * 4,
                            truth_period=[3.0, 5.0, 7.0, 9.0], pbin=["3-10"] * 4, star_type=["dwarf"] * 4,
                            pulsation_class=[np.nan] * 4, Tmag=[10.0] * 4,
                            split=["validation", "validation", "validation", "holdout"]))
    mp = tmp_path / "man.csv"
    man.to_csv(mp, index=False)
    monkeypatch.setattr(sc, "TESSVAL_MANIFEST", mp)
    rows = []
    for tic, prod, tag in [(1, "spoc", "a_lc"), (2, "qlp", "b_llc"), (4, "spoc", "d_lc")]:  # TIC3: nothing downloaded
        rel = f"{prod}/TIC{tic}/{tag}.fits"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        write_lc(tmp_path / rel, 14, 1000.0, qlp=(prod == "qlp"))
        rows.append(dict(tic=tic, product=prod, relpath=rel, status="done"))
    pd.DataFrame(rows).to_csv(tmp_path / "files_index.csv", index=False)
    sap = sc.build_tessval_tasks("tessval_v1_sap", tmp_path)
    assert [t["star_id"] for t in sap] == ["TIC1", "TIC2"]  # TIC4 is the holdout: never selected
    assert sap[1]["meta"]["product"] == "qlp"
    pdc = sc.build_tessval_tasks("tessval_v1_pdcsap", tmp_path)
    assert [t["star_id"] for t in pdc] == ["TIC1"]  # PDCSAP is SPOC-only (paired subset)
    sid, t, f, p, prange, meta, secs = sc._load(sap[0])
    assert sid == "TIC1" and p == 3.0 and prange is None and secs == [14] and np.isfinite(f).any()


def test_real_manifest_holdout_never_in_tasks():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import scorecard as sc
    m = pd.read_csv(sc.TESSVAL_MANIFEST)
    hold = set(m.loc[m.split == "holdout", "star_id"])
    assert hold and m.split.isin(["validation", "holdout"]).all()
    assert "holdout" not in sc.TESSVAL_DATASET_RE.pattern


# --- new outcomes (evaluation.py, additive) and the scorecard tessval metrics block --------------------------------------

def test_classify_negative_and_eb_outcome():
    import evaluation as ev
    assert ev.classify_negative(None) == ev.DECLINE and ev.classify_negative(float("nan"), "rc_giant") == ev.DECLINE
    assert ev.classify_negative(0.35, "rc_giant") == ev.PULSATION
    assert ev.classify_negative(0.6, "rc_giant") == ev.ANSWER and ev.classify_negative(0.35, None) == ev.ANSWER
    assert ev.classify_negative(2.0, "gaia_pulsator") == ev.PULSATION and ev.classify_negative(3.5, "gaia_pulsator") == ev.ANSWER
    assert ev.classify_negative(12.0, "gaia_pulsator") == ev.ANSWER
    assert [ev.eb_outcome(o) for o in ("correct", "half", "double", "wrong", "none")] == ["lock", "lock", "lock", "other", "decline"]
    lo, hi = ev.wilson_interval(0, 40)
    assert lo == 0.0 and 0.05 < hi < 0.12
    assert np.isnan(ev.wilson_interval(0, 0)[0])


def test_tessval_block_and_no_change_to_other_runs():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import scorecard as sc
    base = pd.DataFrame(dict(star_id=list("abcd"), final_outcome=["correct", "wrong", "none", "correct"], period_bin=["<1d"] * 4,
                             final_period=[1.0, 2.0, None, 1.0], error=[None] * 4, seconds=[1.0] * 4))
    assert sc._tessval_block(base) == {}  # a non-tessval run gets no tessval key
    neg = pd.DataFrame(dict(
        star_id=list("abcdefg"), error=[None] * 6 + ["boom"],
        meta_tessval_label=["nonrotator"] * 4 + ["expected_undetectable"] * 2 + ["nonrotator"],
        meta_star_type=["rc_giant", "rc_giant", "dwarf", "dwarf", "dwarf", "dwarf", "dwarf"],
        meta_pulsation_class=["rc_giant", "rc_giant", "", "", "", "", ""],
        final_period=[0.3, 9.0, None, 5.0, 4.0, None, 1.0], final_outcome=["none"] * 7))
    b = sc._tessval_block(neg)
    nr = b["nonrotator"]["overall"]
    assert nr["n"] == 4 and nr["n_errors"] == 1 and nr["pulsation"] == 0.25 and nr["answer"] == 0.5 and nr["decline"] == 0.25
    assert b["nonrotator"]["by_star_type"]["rc_giant"]["pulsation"] == 0.5
    assert b["expected_undetectable"]["overall"]["answer"] == 0.5
    eb = pd.DataFrame(dict(star_id=list("abc"), error=[None] * 3, meta_tessval_label=["contaminant_eb"] * 3, meta_pbin=["1-3"] * 3,
                           final_outcome=["half", "wrong", "none"], final_period=[1, 2, None]))
    e = sc._tessval_block(eb)["contaminant_eb"]["overall"]
    assert e["lock"] == pytest.approx(1 / 3) and e["half"] == pytest.approx(1 / 3) and e["other"] == pytest.approx(1 / 3)
