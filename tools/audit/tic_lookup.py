"""Public-catalog lookups for the 1,719 SMARTS noise sources (TIC + Gaia DR3), cached on disk.

Run with an env that has astroquery (``codeastro``); ``ss_openblas`` has none
and nothing should be installed there. Analysis happens later in
``ss_openblas`` from the CSVs written here. Read-only public queries, a few
large requests rather than thousands of small ones:

1. ``tic_targets.csv``: TIC v8.2 rows of every noise source (batched ID query).
2. ``gaia_xmatch.csv``: Gaia DR3 sources (VizieR I/355/gaiadr3) within ``--radius``
   arcsec (default 63", 3 TESS pixels) of each source, G < ``--gmax``, with
   astrometry (parallax, proper motion, astrometric excess noise, RUWE), DSC
   star/galaxy/quasar probabilities and the variability flag; batched
   multi-position cone queries, cached per batch (resumable).
3. ``gaia_vari.csv``: Gaia DR3 variability classification (VizieR I/358/vclassre)
   of every matched source flagged VARIABLE.

Each step is skipped if its output exists (delete it to refresh).

Usage::

    /opt/anaconda3/envs/codeastro/bin/python -I tools/audit/tic_lookup.py \\
        --tic-table /Volumes/Data_Vault/smarts_data/extracted/galaxy_ids/tic_table.csv \\
        --outdir /Volumes/Data_Vault/smarts_data/extracted/tic_lookup
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

TIC_COLS = ["ID", "ra", "dec", "Tmag", "objType", "typeSrc", "GAIA", "TWOMASS", "plx", "e_plx", "pmRA",
            "pmDEC", "Jmag", "Kmag", "GAIAmag", "Teff", "lumclass", "numcont", "contratio",
            "disposition", "duplicate_id", "gallat", "eclat"]


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically (temp file in the same directory, then rename)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part.csv")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def tic_targets(ids: list, chunk: int = 300, pause: float = 1.0) -> pd.DataFrame:
    """TIC catalog rows for ``ids`` (batched ``query_criteria`` calls, ``pause`` s apart)."""
    from astroquery.mast import Catalogs
    out = []
    for i in range(0, len(ids), chunk):
        t = Catalogs.query_criteria(catalog="Tic", ID=[int(x) for x in ids[i:i + chunk]])
        out.append(t[TIC_COLS].to_pandas())
        print(f"  TIC {i + len(t)}/{len(ids)}", flush=True)
        time.sleep(pause)
    return pd.concat(out, ignore_index=True)


GAIA_COLS = ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "Plx", "RPlx", "pmRA", "pmDE", "epsi",
             "sepsi", "RUWE", "PQSO", "PGal", "PSS", "VarFlag", "NSS", "_r"]


def _vizier(columns, **kw):
    """A VizieR client on the https CDS mirror, no row limit."""
    from astroquery.vizier import Vizier
    Vizier.VIZIER_SERVER = "vizier.cds.unistra.fr"
    return Vizier(columns=columns, row_limit=-1, **kw)


def gaia_xmatch(tg: pd.DataFrame, radius: float, gmax: float, cache: Path, batch: int = 50,
                pause: float = 1.0) -> pd.DataFrame:
    """Gaia DR3 (VizieR I/355/gaiadr3) sources within ``radius`` arcsec of each target.

    Multi-position VizieR cone queries, ``batch`` targets per request (~14 s each),
    cached as ``cache/batch_NNNN.csv`` (resumable). Tried first and abandoned
    (2026-10-21): an ESA-archive uploaded-table join (HTTP 408 timeout), per-target
    ESA sync cones (~7 s each, ~3.3 h), CDS XMatch (port 80 unreachable from here).
    Column meanings: ``epsi`` astrometric excess noise (mas), ``sepsi`` its
    significance, ``PSS``/``PGal``/``PQSO`` DSC star/galaxy/quasar probabilities,
    ``_r`` separation (arcsec).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    cache.mkdir(parents=True, exist_ok=True)
    v = _vizier(GAIA_COLS, column_filters={"Gmag": f"<{gmax}"})
    out = []
    for b0 in range(0, len(tg), batch):
        f = cache / f"batch_{b0:04d}.csv"
        d = tg.iloc[b0:b0 + batch]
        if not f.exists():
            for attempt in range(3):
                try:
                    r = v.query_region(SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg),
                                       radius=radius * u.arcsec, catalog="I/355/gaiadr3")
                    break
                except Exception as e:  # transient service errors: back off and retry
                    print(f"  retry batch {b0} ({type(e).__name__})", flush=True)
                    time.sleep(15 * (attempt + 1))
            else:
                raise RuntimeError(f"VizieR query failed for batch {b0}")
            df = r[0].to_pandas() if len(r) else pd.DataFrame(columns=["_q"] + GAIA_COLS)
            df["tic"] = d.ID.values[df.pop("_q").values - 1]
            _atomic_csv(df, f)
            time.sleep(pause)
        out.append(pd.read_csv(f))
        print(f"  Gaia {min(b0 + batch, len(tg))}/{len(tg)}", flush=True)
    return pd.concat(out, ignore_index=True).rename(columns={"_r": "sep_arcsec"})


def gaia_vari(source_ids: np.ndarray, batch: int = 200, pause: float = 1.0) -> pd.DataFrame:
    """Gaia DR3 variability classes (VizieR I/358/vclassre) for ``source_ids``."""
    v = _vizier(["Source", "Class", "ClassSc"])
    out = []
    for b0 in range(0, len(source_ids), batch):
        ids = ",".join(str(int(x)) for x in source_ids[b0:b0 + batch])
        r = v.query_constraints(catalog="I/358/vclassre", Source=f"={ids}")
        if len(r):
            out.append(r[0].to_pandas())
        time.sleep(pause)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Source", "Class", "ClassSc"])


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tic-table", required=True, help="tic_table.csv from galaxy_tic_map.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--radius", type=float, default=63.0, help="arcsec (3 TESS pixels)")
    ap.add_argument("--gmax", type=float, default=19.0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.tic_table).tic.tolist()

    p = out / "tic_targets.csv"
    if not p.exists():
        _atomic_csv(tic_targets(ids), p)
    tg = pd.read_csv(p)
    print(f"TIC: {len(tg)} rows for {len(ids)} ids")

    p = out / "gaia_xmatch.csv"
    if not p.exists():
        _atomic_csv(gaia_xmatch(tg, args.radius, args.gmax, out / "gaia_cones"), p)
    gx = pd.read_csv(p)
    print(f"Gaia: {len(gx)} sources within {args.radius}\" of {gx.tic.nunique()} targets")

    p = out / "gaia_vari.csv"
    if not p.exists():
        var = gx.loc[gx.VarFlag == "VARIABLE", "Source"].unique()
        _atomic_csv(gaia_vari(var) if len(var) else pd.DataFrame(), p)
    print(f"Gaia vari: {len(pd.read_csv(p))} classified variables")


if __name__ == "__main__":
    main()
