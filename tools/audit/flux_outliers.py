"""Flag SMARTS files whose raw flux has extreme excursions (corrupt galaxy-noise orbits).

Some TESS galaxy noise light curves contain whole orbits of garbage (relative
flux far from 1, even negative). ``load_smarts_fits`` flags and drops those
orbits, but anything reading the raw flux or the WAVELET HDU (Claytor's CNN
input) sees them. For each file this records max |flux - 1|, the number of
cadences with |flux - 1| > ``--thresh``, and the orbits (27 equal time slices
of the 365-d grid) that contain them.

Resumable: appends one CSV row per file to ``--out`` and skips files already
listed there (an interrupted run loses at most the row being written).

``loader_residuals()`` then checks how much of the corruption survives
``load_smarts_fits`` for the flagged files (see the report's Reproduce section).

Usage::

    python tools/audit/flux_outliers.py --inputs DIR_OR_GLOB ... --out reports/<folder>/data/flux_outliers.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from noise_fingerprint import collect_paths  # noqa: E402

FIELDS = ["file", "row", "max_abs_dev", "n_dev_gt_thresh", "orbits_hit", "flux_min", "flux_max"]


def scan(path: Path, thresh: float) -> dict:
    """Outlier summary for one FITS file."""
    with fits.open(path) as h:
        t = np.asarray(h["LIGHTCURVE"].data["time"], dtype=float)
        f = np.asarray(h["LIGHTCURVE"].data["flux"], dtype=float)
    dev = np.abs(f - 1)
    bad = dev > thresh
    orbit = np.minimum((t / (t.max() + 1e-9) * 27).astype(int) + 1, 27)
    return dict(file=path.name, row=int(path.name[-11:-5]), max_abs_dev=float(dev.max()),
                n_dev_gt_thresh=int(bad.sum()),
                orbits_hit=" ".join(map(str, np.unique(orbit[bad]))),
                flux_min=float(f.min()), flux_max=float(f.max()))


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.1, help="|flux - 1| threshold (relative flux)")
    args = ap.parse_args()
    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["file"] for r in csv.DictReader(fh)}
    paths = [p for p in collect_paths(args.inputs) if p.name not in done]
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for i, p in enumerate(paths):
            if not p.parent.exists():
                sys.exit(f"{p.parent} vanished (drive unplugged?); rerun to resume")
            w.writerow(scan(p, args.thresh))
            if i % 100 == 0:
                fh.flush()
    print(f"scanned {len(paths)} new files -> {out}")


def loader_residuals(candidates_csv: Path, search_dirs: list, out_csv: Path) -> pd.DataFrame:
    """For each flagged file, max |flux| left after ``load_smarts_fits`` (0-centred flux).

    Shows whether the pipeline's orbit flagging removes the corrupt noise.
    Files are looked up by name in ``search_dirs``.
    """
    sys.path.insert(0, str(HERE.parents[1] / "claude_files"))
    from preprocessing import load_smarts_fits
    cand = pd.read_csv(candidates_csv)
    rows = []
    for member in cand.member:
        name = Path(member).name
        path = next((Path(d) / name for d in search_dirs if (Path(d) / name).exists()), None)
        if path is None:
            continue
        lc = load_smarts_fits(path)
        rows.append(dict(member=member, loaded_max_abs=float(np.nanmax(np.abs(lc.flux))),
                         n_orbits_flagged=lc.n_orbits_flagged, n_sectors_used=len(lc.sectors_used)))
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    main()
