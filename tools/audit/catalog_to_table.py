"""Convert the SMARTS-TESS DR1 catalog FITS table into a compact columnar file on the drive.

The catalog FITS (see ``extract_catalog.py``) has three HDUs:

0. PRIMARY: HLSP metadata only.
1. BINTABLE, 1,000,000 rows x 12 columns: the simulation input parameters.
2. IMAGE uint8 (64, 64, 1,000,000): per-simulation 64x64 wavelet images
   (the CNN inputs from Claytor+2022), ~4.1 GB. Not converted here.

The table's column names differ from the per-file FITS header keys, so they
are renamed to the header keys (``CATALOG_TO_HEADER``) to keep one
vocabulary across the repo. Output (on the drive, atomic, skipped if present
unless ``--force``)::

    <root>/extracted/catalog/smarts_catalog.npz      (compressed, ~fast to load)
    <root>/extracted/catalog/smarts_catalog.csv.gz   (portable)

Load with ``load_catalog()``. ~1 min.

Usage::

    python tools/audit/catalog_to_table.py [--data-root ...] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smarts_data as sd  # noqa: E402

#: Catalog TTYPE -> per-file FITS header keyword (or new name where no key exists).
CATALOG_TO_HEADER = {
    "Simulation Number": "SIM_ID",
    "Activity Rate": "ACTIVITY",
    "Cycle Length": "CYCLE",
    "Cycle Overlap": "OVERLAP",
    "Inclination": "INCL",
    "Spot Min": "MINLAT",
    "Spot Max": "MAXLAT",
    "Period": "PERIOD",
    "Omega": "OMEGA",
    "Shear": "DIFFROT",
    "Decay Time": "TSPOT",
    "Butterfly": "BFLY",
}


def catalog_npz_path(root: Path) -> Path:
    """Path of the compressed npz catalog table on the drive."""
    return Path(root, sd.EXTRACT_DIR, "catalog", "smarts_catalog.npz")


def read_catalog_fits(fits_path: Path) -> dict:
    """Read HDU 1 of the catalog FITS into a dict of numpy columns (header-key names).

    ``BFLY`` (stored as 5-char strings 'True '/'False') becomes bool;
    ``SIM_ID`` is int64; everything else float64.
    """
    from astropy.io import fits
    with fits.open(fits_path, memmap=True) as hdul:
        tab = hdul[1].data
        cols = {}
        for src, dst in CATALOG_TO_HEADER.items():
            arr = np.asarray(tab[src])
            if dst == "BFLY":
                s = np.char.strip(np.char.decode(arr) if arr.dtype.kind == "S" else arr)
                bad = ~np.isin(s, ["True", "False"])
                if bad.any():
                    raise ValueError(f"unexpected BFLY values: {np.unique(s[bad])[:10]}")
                arr = s == "True"
            elif dst == "SIM_ID":
                arr = arr.astype(np.int64)
            else:
                arr = arr.astype(np.float64)
            cols[dst] = arr
    return cols


def load_catalog(root: Path | None = None):
    """Load the converted catalog as a pandas DataFrame (header-key column names)."""
    import pandas as pd
    root = sd.require_drive(sd.data_root(root))
    with np.load(catalog_npz_path(root)) as z:
        return pd.DataFrame({k: z[k] for k in z.files})


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--force", action="store_true", help="rewrite outputs even if present")
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    fits_path = Path(root, sd.EXTRACT_DIR, "catalog", sd.CATALOG_FITS)
    if not fits_path.exists():
        sys.exit(f"{fits_path} missing: run tools/audit/extract_catalog.py first")

    npz = catalog_npz_path(root)
    csv = npz.with_suffix(".csv.gz")
    if npz.exists() and csv.exists() and not args.force:
        print(f"already converted: {npz}, {csv}")
        return
    cols = read_catalog_fits(fits_path)
    with sd.atomic_path(npz) as tmp:
        np.savez_compressed(tmp, **cols)
    import pandas as pd
    with sd.atomic_path(csv) as tmp:
        pd.DataFrame(cols).to_csv(tmp, index=False, float_format="%.10g", compression="gzip")
    for p in (npz, csv):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
