"""Extract the SMARTS-TESS DR1 catalog tarball onto the external drive (resumable, atomic).

The catalog tarball holds a single ~4.2 GB FITS file
(``hlsp_smarts_tess_ffi_all_tess_v1.0_sim.fits``). It is extracted to
``<root>/extracted/catalog/`` on the drive; nothing is written to the laptop.
Takes ~1-3 min on the Data_Vault drive. Skips the work if the file already
exists with the size recorded in the tar header.

Usage::

    python tools/audit/extract_catalog.py [--data-root /Volumes/Data_Vault/smarts_data]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smarts_data as sd  # noqa: E402


def extract_catalog(root: Path) -> Path:
    """Stream the catalog FITS out of its tarball into ``<root>/extracted/catalog/``.

    Returns the path of the extracted FITS. Idempotent: a complete existing
    copy (size matches the tar member) is left alone.
    """
    tarball = root / sd.BUNDLE_DIR / sd.CATALOG_TARBALL
    out_dir = sd.extract_dir(root, "catalog")
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            final = out_dir / Path(member.name).name
            if final.exists() and final.stat().st_size == member.size:
                print(f"already extracted: {final} ({member.size} bytes)")
                continue
            print(f"extracting {member.name} ({member.size / 1e9:.2f} GB) -> {final}")
            src = tf.extractfile(member)
            with sd.atomic_path(final) as tmp, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
    return out_dir / sd.CATALOG_FITS


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    print(extract_catalog(root))


if __name__ == "__main__":
    main()
