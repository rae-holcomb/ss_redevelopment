"""Header-only inventory of every WG6 light-curve file (T28 scoping, 2026-11-12).

Reads only the PRIMARY header of each ``*_multiband_lc.fits`` under
``$WG6_DATA_ROOT/extracted/<set>/`` (plus the pre-extracted ``RRLYR_lightcurves/``)
and writes one row per file: set, relative path, NAME, VARTYPE, RA, DEC, the
``NDATA_<band>`` counts, ``SECTOR*``/``UKSRC*`` keys, and any header key not seen in
the standard WG6 header (so quirks surface).

Output: one CSV per set in ``--out`` (default ``reports/2026-11-12_wg6-dataset-scoping/data/inventory``),
written atomically; a set whose CSV exists is skipped (``--redo`` to force). Serial, light I/O.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from astropy.io import fits

DEFAULT_ROOT = "/Volumes/Data_Vault/wg6_roman_data_raw"
STANDARD_KEYS = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "NAME", "RA", "DEC", "VARTYPE",
                 "VSOURCE", "ISOURCE", "HSOURCE", "KSOURCE", "TSOURCE"}


def resolve_root(cli_root):
    """Return the WG6 root (CLI > env WG6_DATA_ROOT > default); exit if the drive is absent."""
    root = Path(cli_root or os.environ.get("WG6_DATA_ROOT", DEFAULT_ROOT))
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive (or set WG6_DATA_ROOT).")
    return root


def set_dirs(root: Path) -> dict[str, Path]:
    """Map set name -> folder of FITS files (only completed extractions, i.e. with ``.done``)."""
    out = {}
    if (root / "RRLYR_lightcurves").is_dir():
        out["RRLYR_lightcurves"] = root / "RRLYR_lightcurves"
    for d in sorted((root / "extracted").iterdir()):
        if d.is_dir() and not d.name.startswith((".", "_")) and (d / ".done").exists():
            out[d.name] = d
    return out


def header_row(path: Path, root: Path, set_name: str) -> dict:
    """Return the inventory row for one FITS file (primary header only)."""
    row = {"set": set_name, "relpath": str(path.relative_to(root)), "size_B": path.stat().st_size}
    try:
        h = fits.getheader(path, 0)
    except Exception as e:  # corrupt / non-FITS files are recorded, not fatal
        row["error"] = repr(e)[:200]
        return row
    for k in ("NAME", "VARTYPE", "RA", "DEC"):
        row[k] = h.get(k)
    nd = {k: h[k] for k in h if k.startswith("NDATA_")}
    row["bands"] = " ".join(k[6:] for k in nd)
    row["ndata"] = " ".join(str(v) for v in nd.values())
    row["n_total"] = sum(int(v) for v in nd.values())
    row["sectors"] = " ".join(str(h[k]) for k in h if k.startswith("SECTOR"))
    row["uksrc"] = " ".join(str(h[k]) for k in h if k.startswith("UKSRC"))
    extra = [k for k in h if k not in STANDARD_KEYS and not k.startswith(("NDATA_", "SECTOR", "UKSRC", "CCD"))]
    row["extra_keys"] = " ".join(f"{k}={h[k]}" for k in extra)[:500]
    return row


def main(argv=None):
    """CLI: write one inventory CSV per WG6 set."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="reports/2026-11-12_wg6-dataset-scoping/data/inventory")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["set", "relpath", "size_B", "NAME", "VARTYPE", "RA", "DEC", "bands", "ndata", "n_total",
            "sectors", "uksrc", "extra_keys", "error"]
    for name, d in set_dirs(root).items():
        dest = out / f"{name}.csv"
        if (dest.exists() or dest.with_suffix(".csv.gz").exists()) and not args.redo:
            print(f"skip {name}")
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fz", ".gz")
                       and not p.name.startswith("._"))
        others = sorted(p.name for p in d.rglob("*") if p.is_file() and p not in set(files) and p.name != ".done")
        print(f"{name}: {len(files)} FITS, {len(others)} other files {others[:5]}", flush=True)
        tmp = dest.with_suffix(".csv.part")
        seen = set()
        if tmp.exists():  # resume inside a set: keep complete rows already written
            with open(tmp, newline="") as fh:
                seen = {r["relpath"] for r in csv.DictReader(fh) if r.get("size_B")}
        with open(tmp, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if fh.tell() == 0:
                w.writeheader()
            for p in files:
                if str(p.relative_to(root)) in seen:
                    continue
                w.writerow(header_row(p, root, name))
                fh.flush()
        os.replace(tmp, dest)


if __name__ == "__main__":
    main()
