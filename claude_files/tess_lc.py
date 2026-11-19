"""Loader for real TESS light curves (TESS-SPOC FFI and QLP HLSP FITS) -> ``PreprocessedLightCurve``.

Counterpart of ``preprocessing.load_smarts_fits`` for the P27 real-TESS validation set
(``datasets/tess_validation_v0.csv``). It returns the same container with the same conventions:
time in days, flux = relative flux - 1 (zero-centred, per-sector median), an even grid (default 30 min) with NaN in
every empty slot, so ``acf_utils`` and the rest of the pipeline run unchanged.

One *star* is many files (one per sector), so ``load_tess_star`` takes a list of paths and stitches them;
``load_tess_sector`` reads one.

Flux column (``flux_column``)
-----------------------------
``"SAP_FLUX"`` (default; closest to the SMARTS noise, Rae 2026-11-12) or ``"PDCSAP_FLUX"`` (TESS-SPOC only; for the paired
comparison). Columns that are detrended on short windows (QLP ``KSPSAP_FLUX``, TGLC ``cal_*``) erase rotation beyond a few
days and are refused outright.

Quality masking (``quality``)
-----------------------------
``"default"`` (SPOC: the lightkurve default bitmask 175 = bits 1,2,3,4,6,8; QLP: any non-zero ``QUALITY``), ``"hard"`` (SPOC
bitmask 7407), ``"hardest"`` (any non-zero flag), ``"none"``, or an int bitmask applied to ``QUALITY``. Cadences with a
non-finite time or flux are dropped as well. Flagged cadences become NaN slots, never interpolated.

Assumptions to confirm on one real file of each product before the first scored run (nothing here touched the network):
HDU ``LIGHTCURVE``, columns ``TIME`` (BTJD, days), ``SAP_FLUX``, ``PDCSAP_FLUX`` (SPOC only), ``QUALITY``; primary header
keyword ``SECTOR``. ``tests/test_tess_lc.py`` builds fixtures with exactly this layout.

Regridding
----------
FFI cadences are 30 min (S1-26), 10 min (S27-55) or 200 s (S56+); the pipeline grid is ``timestep`` seconds (default 1800).
Points falling in the same grid bin are **averaged** (unlike ``to_even_cadence``, which keeps the last one), so 10-min and
200-s sectors are binned down rather than subsampled. 30-min input maps one-to-one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from preprocessing import NoUsableSectorsError, PreprocessedLightCurve

FLUX_COLUMNS = ("SAP_FLUX", "PDCSAP_FLUX")
REFUSED_COLUMNS = ("KSPSAP_FLUX", "CAL_PSF_FLUX", "CAL_APER_FLUX", "CORR_FLUX", "FLUX")  # short-window detrended
SPOC_BITMASKS = {"default": 175, "hard": 7407, "hardest": 0xFFFFFFFF, "none": 0}
DEFAULT_ROOT = "/Volumes/Data_Vault/tess_validation"


@dataclass
class TessLightCurve(PreprocessedLightCurve):
    """``PreprocessedLightCurve`` plus ``meta`` (tic, product, flux_column, t0_btjd, per-sector counts)."""
    meta: dict = field(default_factory=dict)


def tess_validation_root(root: Union[str, Path, None] = None) -> Path:
    """Root of the downloaded validation light curves: argument, else ``$TESS_VALIDATION_ROOT``, else the Data_Vault default."""
    return Path(root or os.environ.get("TESS_VALIDATION_ROOT") or DEFAULT_ROOT).expanduser()


def _detect_product(header, path) -> str:
    """``"qlp"`` or ``"spoc"``, from the primary header (ORIGIN/AUTHOR/PROCVER text) and then the file name."""
    blob = " ".join(str(header.get(k, "")) for k in ("ORIGIN", "AUTHOR", "CREATOR", "PROCVER")).lower()
    name = Path(str(path)).name.lower()
    if "qlp" in blob or "qlp" in name:
        return "qlp"
    return "spoc"


def _bad_mask(quality: np.ndarray, quality_arg: Union[str, int], product: str) -> np.ndarray:
    """True where a cadence is flagged bad under ``quality_arg``."""
    q = np.asarray(quality).astype(np.int64)
    if isinstance(quality_arg, (int, np.integer)):
        return (q & int(quality_arg)) != 0
    if quality_arg not in SPOC_BITMASKS:
        raise ValueError(f"quality must be one of {sorted(SPOC_BITMASKS)} or an int bitmask, got {quality_arg!r}")
    if quality_arg == "default" and product == "qlp":
        return q != 0
    return (q & SPOC_BITMASKS[quality_arg]) != 0


def load_tess_sector(path: Union[str, Path], flux_column: str = "SAP_FLUX", quality: Union[str, int] = "default",
                     lightcurve_hdu: str = "LIGHTCURVE", product: str = "auto") -> dict:
    """Read one SPOC/QLP light-curve FITS: ``dict(time, flux, sector, product, n_total, n_flagged, n_nan, median)``.

    ``flux`` is relative flux - 1 (divided by the median of the unflagged finite cadences); flagged and non-finite cadences
    are removed (not NaN-filled: the regrid does that). ``time`` is BTJD in days.
    """
    from astropy.io import fits
    col = str(flux_column).upper()
    if col in REFUSED_COLUMNS:
        raise ValueError(f"{flux_column} is detrended on a short window and erases rotation: use SAP_FLUX or PDCSAP_FLUX")
    if col not in FLUX_COLUMNS:
        raise ValueError(f"flux_column must be one of {FLUX_COLUMNS}, got {flux_column!r}")
    with fits.open(path) as h:
        prod = _detect_product(h[0].header, path) if product == "auto" else product
        sector = h[0].header.get("SECTOR")
        if sector is None:
            sector = h[1].header.get("SECTOR")
        d = h[lightcurve_hdu].data
        names = {n.upper(): n for n in d.columns.names}
        if col not in names:
            raise ValueError(f"{Path(path).name} ({prod}) has no {col} column (has {sorted(names)})")
        time = np.asarray(d[names["TIME"]], float)
        flux = np.asarray(d[names[col]], float)
        qual = np.asarray(d[names["QUALITY"]]) if "QUALITY" in names else np.zeros(time.size, int)
    bad = _bad_mask(qual, quality, prod)
    finite = np.isfinite(time) & np.isfinite(flux)
    keep = ~bad & finite
    med = float(np.median(flux[keep])) if keep.any() else float("nan")
    if keep.any() and not med > 0:
        raise ValueError(f"{Path(path).name}: non-positive median {col} ({med}); cannot normalise")
    return dict(time=time[keep], flux=flux[keep] / med - 1.0 if keep.any() else flux[keep], sector=None if sector is None else int(sector),
                product=prod, n_total=int(time.size), n_flagged=int(bad.sum()), n_nan=int((~finite).sum()), median=med)


def _regrid_mean(time: np.ndarray, flux: np.ndarray, timestep: float) -> tuple:
    """Even grid of ``timestep`` seconds from ``time[0]``; each bin is the mean of its points, NaN where empty."""
    dt = timestep / 86400.0
    n = int(round((time[-1] - time[0]) / dt)) + 1
    idx = np.round((time - time[0]) / dt).astype(int)
    ok = (idx >= 0) & (idx < n)
    s = np.zeros(n)
    c = np.zeros(n, int)
    np.add.at(s, idx[ok], flux[ok])
    np.add.at(c, idx[ok], 1)
    out = np.full(n, np.nan)
    np.divide(s, c, out=out, where=c > 0)
    return time[0] + np.arange(n) * dt, out


def load_tess_star(paths: Sequence[Union[str, Path]], flux_column: str = "SAP_FLUX", quality: Union[str, int] = "default",
                   sectors: Optional[Sequence[int]] = None, timestep: float = 1800, true_period: Optional[float] = None,
                   tic: Optional[int] = None, lightcurve_hdu: str = "LIGHTCURVE", product: str = "auto") -> TessLightCurve:
    """One call: the per-sector SPOC/QLP FITS files of one star -> pipeline-ready ``PreprocessedLightCurve``.

    Each sector is median-normalised on its own (relative flux - 1), the sectors are concatenated in time order, and the
    result is put on an even ``timestep`` grid (NaN in empty slots, so sector gaps stay as gaps). ``time`` runs in days from
    the first cadence (``meta['t0_btjd']`` is its BTJD).

    Parameters
    ----------
    paths : the star's FITS files (any order; they must be the same product).
    sectors : optional sector numbers to keep (e.g. to test a coverage pattern); a file without a SECTOR keyword is kept
        only when ``sectors`` is None.
    true_period : stored on the container (the files carry none; join it from the manifest).
    tic : stored in ``meta`` only.

    Raises
    ------
    NoUsableSectorsError (a ValueError) if no cadence survives the sector selection and quality mask.
    ValueError for a refused/unknown flux column, mixed products, or ``PDCSAP_FLUX`` requested of a QLP file.
    """
    want = None if sectors is None else {int(s) for s in sectors}
    secs = []
    seen = set()
    for p in sorted(map(str, paths)):
        s = load_tess_sector(p, flux_column=flux_column, quality=quality, lightcurve_hdu=lightcurve_hdu, product=product)
        if want is not None and s["sector"] not in want:
            continue
        if s["sector"] is not None and s["sector"] in seen:
            continue  # same sector twice (re-delivery): keep the first
        seen.add(s["sector"])
        s["path"] = p
        secs.append(s)
    prods = {s["product"] for s in secs}
    if len(prods) > 1:
        raise ValueError(f"mixed products in one star: {sorted(prods)}")
    usable = [s for s in secs if s["time"].size]
    if not usable:
        raise NoUsableSectorsError(str(paths[0]) if len(paths) else "<no files>", None if want is None else sorted(want),
                                   [s["sector"] for s in secs], [], len(secs), len(secs))
    usable.sort(key=lambda s: s["time"][0])
    t = np.concatenate([s["time"] for s in usable])
    f = np.concatenate([s["flux"] for s in usable])
    order = np.argsort(t, kind="stable")
    t, f = t[order], f[order]
    time_even, flux_even = _regrid_mean(t, f, timestep)
    t0 = float(t[0])
    segments = [(s["time"] - t0, s["flux"]) for s in usable]
    meta = dict(tic=tic, product=usable[0]["product"], flux_column=str(flux_column).upper(), quality=quality,
                t0_btjd=t0, timestep_s=float(timestep), n_files=len(secs),
                per_sector={s["sector"]: dict(n=int(s["time"].size), n_flagged=s["n_flagged"], n_nan=s["n_nan"]) for s in secs})
    diag = dict(n_cadences_kept=int(t.size), n_flagged=int(sum(s["n_flagged"] for s in secs)),
                n_nan=int(sum(s["n_nan"] for s in secs)), n_total=int(sum(s["n_total"] for s in secs)))
    return TessLightCurve(
        time=time_even - t0, flux=flux_even, true_period=None if true_period is None else float(true_period),
        sectors_used=[s["sector"] for s in usable if s["sector"] is not None], segments=segments, diagnostics=diag,
        source_path=";".join(s["path"] for s in usable), n_orbits_total=len(secs),
        n_orbits_flagged=int(len(secs) - len(usable)), meta=meta)
