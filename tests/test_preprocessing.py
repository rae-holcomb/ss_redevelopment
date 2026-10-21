"""Tests for preprocessing.load_smarts_fits's no-usable-data error (task T21).

When nothing survives sector selection (every requested sector flagged or
beyond the end of the file, or every orbit flagged with `sectors=None`),
load_smarts_fits must raise a named NoUsableSectorsError (a ValueError) that
says which sectors were requested, flagged or absent, and which file, before
regridding. Before T21 it crashed inside to_even_cadence with "attempt to get
argmax of an empty sequence".
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import preprocessing
from preprocessing import NoUsableSectorsError, load_smarts_fits
from synthetic import make_lightcurve

ROOT = Path(__file__).resolve().parents[1]
NOISY_SECTOR = 5  # sector whose scatter we inflate so split_lightcurve flags it


def _write_fake_smarts(path, noisy_sector=None):
    """Write a SMARTS-style FITS (gaps linearly interpolated, flux centred on 1).

    If `noisy_sector` is given, large white noise is added over that sector's
    two orbits so that split_lightcurve flags both of them.
    """
    full = make_lightcurve(3.0, n_sectors=13, seed=4, white_noise=0.002)
    last = np.flatnonzero(np.isfinite(full.flux))[-1]
    t, f = full.time[:last + 1], full.flux[:last + 1].copy()
    ok = np.isfinite(f)
    f[~ok] = np.interp(t[~ok], t[ok], f[ok])
    if noisy_sector is not None:
        segs, _ = preprocessing.split_lightcurve(t, f, n_segments=27)
        i0 = 2 * (noisy_sector - 1)
        m = (t >= segs[i0][0][0]) & (t <= segs[i0 + 1][0][-1])
        f[m] += np.random.default_rng(0).normal(0, 0.05, m.sum())
    hdr = fits.Header()
    hdr["PERIOD"] = 3.0
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.BinTableHDU.from_columns([fits.Column("time", "D", array=t),
                                                 fits.Column("flux", "D", array=f + 1.0)],
                                                name="LIGHTCURVE")]).writeto(path)
    return path


@pytest.fixture(scope="module")
def noisy_fits(tmp_path_factory):
    """Fake SMARTS file whose sector NOISY_SECTOR is flagged by split_lightcurve."""
    return _write_fake_smarts(tmp_path_factory.mktemp("t21") / "noisy.fits", NOISY_SECTOR)


def test_fixture_flags_the_noisy_sector(noisy_fits):
    """Sanity check on the fixture: both orbits of NOISY_SECTOR are flagged."""
    pre = load_smarts_fits(noisy_fits)
    flagged = pre.diagnostics["segment_flagged"]
    assert flagged[2 * NOISY_SECTOR - 2] and flagged[2 * NOISY_SECTOR - 1]


def test_all_requested_sectors_flagged_raises_named_error(noisy_fits):
    """Requesting only a flagged sector raises NoUsableSectorsError naming it and the file."""
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(noisy_fits, sectors=[NOISY_SECTOR])
    err = ei.value
    assert isinstance(err, ValueError)
    assert err.requested == [NOISY_SECTOR]
    assert err.flagged == [NOISY_SECTOR]
    assert err.absent == []
    msg = str(err)
    assert "noisy.fits" in msg and f"[{NOISY_SECTOR}]" in msg and "flagged" in msg


def test_sector_beyond_end_of_file_raises_named_error(noisy_fits):
    """A requested sector with no orbits in the file is reported as absent."""
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(noisy_fits, sectors=[99])
    assert ei.value.absent == [99]
    assert ei.value.flagged == []
    assert "99" in str(ei.value) and "absent" in str(ei.value)


def test_mixed_flagged_and_absent(noisy_fits):
    """Flagged and absent sectors are listed separately (and deduplicated, sorted)."""
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(noisy_fits, sectors=[99, NOISY_SECTOR, NOISY_SECTOR])
    assert ei.value.requested == [NOISY_SECTOR, 99]
    assert ei.value.flagged == [NOISY_SECTOR]
    assert ei.value.absent == [99]


def test_empty_sector_list_raises(noisy_fits):
    """sectors=[] requests nothing, so nothing survives."""
    with pytest.raises(NoUsableSectorsError):
        load_smarts_fits(noisy_fits, sectors=[])


def test_partial_survival_still_loads(noisy_fits):
    """If at least one requested sector survives, the load succeeds as before."""
    pre = load_smarts_fits(noisy_fits, sectors=[NOISY_SECTOR, 7])
    assert pre.sectors_used == [7]
    assert np.isfinite(pre.flux).sum() > 100


def test_every_orbit_flagged_with_sectors_none_raises(noisy_fits, monkeypatch):
    """sectors=None with every orbit flagged (and drop_flagged_segments=True) raises too."""
    real_split = preprocessing.split_lightcurve

    def all_flagged(*args, **kwargs):
        """split_lightcurve, but with every orbit flagged."""
        segs, diag = real_split(*args, **kwargs)
        diag["segment_flagged"] = np.ones(len(segs), dtype=bool)
        return segs, diag

    monkeypatch.setattr(preprocessing, "split_lightcurve", all_flagged)
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(noisy_fits)
    assert ei.value.requested is None
    assert "every orbit" in str(ei.value)
    # Without dropping flagged orbits the same file still loads.
    assert load_smarts_fits(noisy_fits, drop_flagged_segments=False).sectors_used


def test_error_pickles(noisy_fits):
    """The error survives a pickle round trip (it may cross a process-pool boundary)."""
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(noisy_fits, sectors=[99, NOISY_SECTOR])
    back = pickle.loads(pickle.dumps(ei.value))
    assert str(back) == str(ei.value)
    assert (back.requested, back.flagged, back.absent) == ([NOISY_SECTOR, 99], [NOISY_SECTOR], [99])


REAL_FILE = ROOT / "data" / "smarts_sample" / "smarts-tess-v1.0-828115.fits"


@pytest.mark.skipif(not REAL_FILE.exists(), reason="data/smarts_sample not present")
def test_real_file_all_flagged_request():
    """Real SMARTS file whose sectors 1 and 11 are flagged: the T2 crash case."""
    with pytest.raises(NoUsableSectorsError) as ei:
        load_smarts_fits(REAL_FILE, sectors=[1, 11])
    assert ei.value.flagged == [1, 11]
