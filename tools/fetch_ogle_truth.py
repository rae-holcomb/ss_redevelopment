"""Download the OGLE-IV OCVS bulge catalog tables (truth periods) for the WG6 classes.

Rae approved this download on 2026-11-11 (relayed by the coordinator session): catalog tables only, no photometry.
Files go to ``$WG6_DATA_ROOT/extracted/_truth/<cls>/<file>``, one class folder each. Each file is written to
``.part`` then renamed, so the download is resumable (files already present are skipped). Every URL, size and date goes
to ``_truth/manifest.csv``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/"
FILES = {
    "": ["README"],
    "cep": ["README", "ident.dat", "cepF.dat", "cep1O.dat", "cepF1O.dat", "cep1O2O.dat", "cep1O2O3O.dat", "cep2O3O.dat"],
    "t2cep": ["README", "ident.dat", "t2cep.dat"],
    "rrlyr": ["README", "ident.dat", "RRab.dat", "RRc.dat", "RRd.dat", "aRRd.dat"],
    "dsct": ["README", "ident.dat", "dsct.dat"],
    "ecl": ["README", "ident.dat", "ecl.dat", "ell.dat"],
    "rot": ["README", "ident.dat", "rot.dat", "remarks.txt"],
    "lpv": ["README", "ident.dat", "Miras.dat"],
    "hb": ["README", "ident.dat", "hb.dat"],
}


def main(argv=None):
    """CLI: fetch every table in FILES that isn't already on the drive."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("WG6_DATA_ROOT", "/Volumes/Data_Vault/wg6_roman_data"))
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive (or set WG6_DATA_ROOT).")
    out = root / "extracted" / "_truth"
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"
    rows = {}
    if man_path.exists():
        rows = {r["url"]: r for r in csv.DictReader(open(man_path))}
    for cls, names in FILES.items():
        d = out / (cls or "_root")
        d.mkdir(exist_ok=True)
        for n in names:
            url = BASE + (f"{cls}/" if cls else "") + n
            dest = d / n
            if dest.exists() and url in rows:
                print(f"skip {url}")
                continue
            part = dest.with_name(n + ".part")
            print(f"get  {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(part, dest)
            rows[url] = {"url": url, "path": str(dest.relative_to(root)), "bytes": dest.stat().st_size,
                         "downloaded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")}
            tmp = man_path.with_suffix(".part")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, ["url", "path", "bytes", "downloaded_utc"])
                w.writeheader(); w.writerows(rows.values())
            os.replace(tmp, man_path)


if __name__ == "__main__":
    main()
