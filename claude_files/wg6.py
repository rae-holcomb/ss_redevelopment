"""WG6 (RGES PIT) variable-star light-curve loaders: per-mission adapters returning the ``load_smarts_fits`` container.

Two products (see ``docs/WG6_REFERENCE.md``):

- the **resampled Roman-cadence set** (``load_wg6_roman``): F087/F146/F213 tables on the WG6 GBTDS grid (6 seasons of
  71.00 d, F146 every 12.1 min). Read **in place from the zips** (no extraction), improved- or simple-noise variant;
- the **raw archival set** (``load_wg6_raw``): OGLE I-band ground photometry (with the Cepheid double-offset HJD fix).

Both return a ``WG6LightCurve``, a ``PreprocessedLightCurve`` (``time``, ``flux``, ``true_period``, ... as from
``preprocessing.load_smarts_fits``) plus ``flux_err`` and a ``meta`` dict. Conventions that match the SMARTS loader, so the
pipeline and scorecard treat both the same way:

- ``flux`` is relative flux **minus 1** (zero-centred, median 0), not magnitudes;
- ``time`` is in days. The Roman loader returns days since the start of the cut season, on the file's own uniform grid;
  slots where the file has NaN (FL dropouts) stay in place as NaN flux.

Assumptions recorded here (Rae, 2026-11-13): development uses only the three *early* GBTDS seasons. WG6's grid has six
consecutive 71-d seasons with no early/late split, so **seasons 1-3 (1-indexed) are treated as the early ones**.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from preprocessing import PreprocessedLightCurve  # noqa: F401  (container re-used from the SMARTS loader)

from dataclasses import dataclass, field

DEFAULT_RESAMPLED_ROOT = "/Volumes/Data_Vault/wg6_roman_data_resampled"
EARLY_SEASONS = (1, 2, 3)        # assumption: first 3 of WG6's 6 consecutive seasons = the "early" GBTDS seasons
SEASON_GAP_DAYS = 1.0            # a gap above this starts a new season (in-season gaps are < 1 h; between seasons 109-115 d)
BANDS = ("F087", "F146", "F213")
_MAG_TO_FLUX = np.log(10.0) / 2.5  # sigma_flux = _MAG_TO_FLUX * sigma_mag * flux

_ZIP_CACHE: dict = {}


@dataclass
class WG6LightCurve(PreprocessedLightCurve):
    """``PreprocessedLightCurve`` plus per-point ``flux_err`` and a ``meta`` dict (name, vartype, band, season, ...)."""
    flux_err: Optional[np.ndarray] = field(default=None, repr=False)
    meta: dict = field(default_factory=dict)


def resampled_root(root: Union[str, Path, None] = None) -> Path:
    """Root of the resampled data: argument, else ``$WG6_RESAMPLED_ROOT``, else the Data_Vault default path."""
    return Path(root or os.environ.get("WG6_RESAMPLED_ROOT") or DEFAULT_RESAMPLED_ROOT).expanduser()


def _open_zip(path: Path) -> zipfile.ZipFile:
    """Per-process cached ZipFile handle (the FL zips hold thousands of members; re-reading the directory is slow)."""
    key = str(path)
    zf = _ZIP_CACHE.get(key)
    if zf is None or zf.fp is None:
        zf = _ZIP_CACHE[key] = zipfile.ZipFile(path)
    return zf


def _read_fits_bytes(zip_or_path: Union[str, Path], member: Optional[str], root: Union[str, Path, None]) -> tuple:
    """Return (bytes_or_path, label). A ``.zip`` needs ``member``; otherwise ``zip_or_path`` is a FITS file.

    A relative ``zip_or_path`` that doesn't exist as given is resolved against the resampled root.
    """
    p = Path(zip_or_path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = resampled_root(root) / p
    if p.suffix.lower() == ".zip":
        if member is None:
            raise ValueError(f"{p.name} is a zip: pass member=<path of the FITS inside it>")
        return _open_zip(p).read(member), f"{p.name}::{member}"
    return p.read_bytes(), str(p)


def split_seasons(t: np.ndarray, gap_days: float = SEASON_GAP_DAYS) -> list:
    """Split a time array at gaps larger than ``gap_days``; returns a list of ``slice`` objects, one per contiguous season."""
    t = np.asarray(t, float)
    if t.size == 0:
        return []
    cut = np.where(np.diff(t) > gap_days)[0] + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, t.size]
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def mag_to_flux(mag: np.ndarray, mag_err: Optional[np.ndarray] = None) -> tuple:
    """Magnitudes -> relative flux with median 1 (NaN-safe median), and the matching flux error.

    ``flux = 10**(-0.4 (mag - nanmedian(mag)))``; ``sigma_flux = ln(10)/2.5 * sigma_mag * flux``. NaNs propagate (the
    slot stays, flux is NaN). If every value is NaN the result is all-NaN.
    """
    mag = np.asarray(mag, float)
    with np.errstate(all="ignore"):
        med = np.nanmedian(mag) if np.isfinite(mag).any() else np.nan
        flux = 10.0 ** (-0.4 * (mag - med))
        err = None if mag_err is None else _MAG_TO_FLUX * np.asarray(mag_err, float) * flux
    return flux, err


def load_wg6_roman(zip_or_path: Union[str, Path], member: Optional[str] = None, band: str = "F146",
                   season: Optional[int] = None, times_dir: Union[str, Path, None] = None,
                   true_period: Optional[float] = None, root: Union[str, Path, None] = None) -> WG6LightCurve:
    """Load one WG6 Roman-resampled light curve (one band, optionally one 71-d season) as a pipeline-ready container.

    Parameters
    ----------
    zip_or_path : a ``.zip`` (then ``member`` names the FITS inside; read in memory, never extracted) or a FITS path.
        A relative path is resolved against ``root`` / ``$WG6_RESAMPLED_ROOT`` if it doesn't exist as given.
    member : FITS member path inside the zip.
    band : ``"F146"`` (12.1 min, the pipeline band), ``"F087"`` or ``"F213"`` (6 h).
    season : 1-indexed season to cut (1-3 are the "early" GBTDS seasons by our assumption; 1-6 exist). ``None`` keeps
        all six; the zero-point of ``time`` is then the first point.
    times_dir : directory holding ``roman_times_{shortcadence,longcadence,longcadence2}.npy``; only needed for the
        simple-noise files, which have no ``JD`` column (improved files carry it).
    true_period : stored on the container (the file carries no truth; join it from the manifest).

    Returns
    -------
    WG6LightCurve with ``time`` (days from the first point of the cut), ``flux`` (relative flux - 1, NaN where the file
    has NaN), ``flux_err`` (relative-flux sigma), ``meta`` (name, vartype, band, season, jd0, median_mag, n_nan, ...).

    Raises
    ------
    ValueError for an unknown band, a zip without ``member``, a missing season, or a simple-noise file without ``times_dir``.
    """
    from astropy.io import fits
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    raw, label = _read_fits_bytes(zip_or_path, member, root)
    with fits.open(io.BytesIO(raw)) as h:
        if band not in h:
            raise ValueError(f"{label} has no {band} extension (has {[x.name for x in h][1:]})")
        hdr = h[0].header
        d = h[band].data
        names = d.columns.names
        mag = np.asarray(d["mag"], float)
        err = np.asarray(d["mag_error"], float)
        if "JD" in names:
            jd = np.asarray(d["JD"], float)
        else:
            if times_dir is None:
                raise ValueError(f"{label} has no JD column (simple-noise file): pass times_dir=")
            fn = {"F146": "roman_times_shortcadence.npy", "F087": "roman_times_longcadence.npy",
                  "F213": "roman_times_longcadence2.npy"}[band]
            jd = np.asarray(np.load(Path(times_dir) / fn), float)
            if jd.size != mag.size:
                raise ValueError(f"{label}: {mag.size} rows but {fn} has {jd.size}")
        name, vartype = str(hdr.get("NAME", "")).strip(), str(hdr.get("VARTYPE", "")).strip()

    n_seasons = len(split_seasons(jd))
    if season is not None:
        seg = split_seasons(jd)
        if not 1 <= int(season) <= len(seg):
            raise ValueError(f"{label}: season {season} not in 1..{len(seg)}")
        sl = seg[int(season) - 1]
        jd, mag, err = jd[sl], mag[sl], err[sl]
    flux, ferr = mag_to_flux(mag, err)
    n_nan = int(np.isnan(flux).sum())
    meta = dict(name=name, vartype=vartype, band=band, season=season, n_seasons_in_file=n_seasons, jd0=float(jd[0]),
                median_mag=float(np.nanmedian(mag)) if np.isfinite(mag).any() else float("nan"), n_nan=n_nan,
                median_mag_err=float(np.nanmedian(err)) if np.isfinite(err).any() else float("nan"))
    return WG6LightCurve(time=jd - jd[0], flux=flux - 1.0, true_period=None if true_period is None else float(true_period),
                         sectors_used=[season] if season is not None else list(range(1, n_seasons + 1)),
                         segments=[], diagnostics={"n_nan": n_nan}, source_path=label,
                         flux_err=ferr, meta=meta)


def fix_ogle_time(t: np.ndarray) -> np.ndarray:
    """Undo the doubled +2450000 offset on Cepheid / type-II Cepheid raw HJDs (WG6_REFERENCE quirk 5.4).

    Values above 3e6 are ``HJD - 2450000`` already shifted twice, so 2450000 is subtracted once; others are unchanged.
    """
    t = np.asarray(t, float)
    return np.where(t > 3e6, t - 2450000.0, t)


def load_wg6_raw(path: Union[str, Path], band: str = "LC_I", true_period: Optional[float] = None) -> WG6LightCurve:
    """Load one WG6 *raw* archival OGLE light curve (extension ``band``, default ``LC_I``) with the HJD fix.

    Returns the same container as ``load_wg6_roman``: ``time`` in days from the first point (uneven, ground-based
    sampling, **not** pipeline-ready until resampled: see WG6_REFERENCE §6), ``flux`` = relative flux - 1 (median 0),
    ``flux_err`` from ``mag_error``. Points with non-finite magnitude are dropped. TESS flare files use extension names
    ``0``, ``1``, ... and columns ``BTJD``: pass ``band="1"`` etc. (the time column is picked by name).
    """
    from astropy.io import fits
    with fits.open(path) as h:
        hdu = h[band] if band in h else h[int(band)]
        d = hdu.data
        names = d.columns.names
        tcol = "HJD" if "HJD" in names else "BTJD"
        t = fix_ogle_time(np.asarray(d[tcol], float))
        mag = np.asarray(d["mag"], float)
        err = np.asarray(d["mag_error" if "mag_error" in names else "mag_err"], float)
        hdr = h[0].header
        name, vartype = str(hdr.get("NAME", "")).strip(), str(hdr.get("VARTYPE", "")).strip()
    ok = np.isfinite(t) & np.isfinite(mag)
    t, mag, err = t[ok], mag[ok], err[ok]
    order = np.argsort(t)
    t, mag, err = t[order], mag[order], err[order]
    flux, ferr = mag_to_flux(mag, err)
    meta = dict(name=name, vartype=vartype, band=band, jd0=float(t[0]) if t.size else float("nan"),
                median_mag=float(np.nanmedian(mag)) if t.size else float("nan"))
    return WG6LightCurve(time=t - (t[0] if t.size else 0.0), flux=flux - 1.0,
                         true_period=None if true_period is None else float(true_period),
                         sectors_used=[], segments=[], diagnostics={}, source_path=str(path), flux_err=ferr, meta=meta)
