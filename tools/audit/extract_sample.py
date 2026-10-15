"""Stream-extract a small, deterministic sample of SMARTS FITS from chosen period bundles.

For each bundle, picks ``--per-bundle`` catalog rows spread across log(ACTIVITY)
(one random row from each activity quantile slice of that bundle, seeded), then
streams the bundle once, extracting only those members. Full bundles are never
unpacked. Files go to the drive (``<root>/extracted/sample/``) and are copied to
the small local working set ``data/smarts_sample/`` (gitignored), with a
manifest ``data/smarts_sample/manifest.csv`` whose ``relpath`` column is
relative to the data root (never an absolute /Volumes path).

Resumable: files already present (right size) are skipped; the manifest is
rewritten atomically at the end. ~6 s per bundle.

Usage::

    python tools/audit/extract_sample.py --bundles 1 2 5 10 20 40 80 120 150 180 --per-bundle 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smarts_data as sd  # noqa: E402
from bundle_index import scan_bundle  # noqa: E402
from catalog_to_table import load_catalog  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LOCAL_SAMPLE = REPO / "data" / "smarts_sample"


def bundle_rows(cat: pd.DataFrame, n: int) -> pd.DataFrame:
    """Catalog rows (index = file row number) whose period falls in bundle ``n``: (n-1, n]."""
    lo = -np.inf if n == 1 else n - 1
    return cat[(cat.PERIOD > lo) & (cat.PERIOD <= n)]


def choose_rows(cat: pd.DataFrame, n: int, k: int, seed: int) -> list:
    """Pick ``k`` rows of bundle ``n``, one per log-activity quantile slice (deterministic)."""
    sub = bundle_rows(cat, n).sort_values("ACTIVITY")
    rng = np.random.default_rng(seed + n)
    picks = [int(rng.choice(chunk)) for chunk in np.array_split(sub.index.values, k)]
    return sorted(picks)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--bundles", type=int, nargs="+",
                    default=[1, 2, 5, 10, 20, 40, 80, 120, 150, 180])
    ap.add_argument("--per-bundle", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-local-copy", action="store_true",
                    help="extract to the drive only; skip data/smarts_sample/")
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    cat = load_catalog(root)
    drive_dir = sd.extract_dir(root, "sample")

    records = []
    for n in args.bundles:
        rows = choose_rows(cat, n, args.per_bundle, args.seed)
        sd.require_drive(root)
        idx = scan_bundle(root, n, extract_rows=rows, extract_to=drive_dir)
        in_bundle = set(idx.row)
        expected = set(bundle_rows(cat, n).index)
        print(f"bundle {n:03d}: {len(idx)} members; catalog predicts {len(expected)}; "
              f"symmetric difference {len(in_bundle ^ expected)}; extracted rows {rows}", flush=True)
        for r in rows:
            name = f"smarts-tess-v1.0-{r:06d}.fits"
            records.append(dict(row=r, bundle=n, file=name,
                                relpath=f"{sd.EXTRACT_DIR}/sample/{name}",
                                member=f"{n:03d}/{name}",
                                size=(drive_dir / name).stat().st_size,
                                bundle_count=len(idx),
                                bundle_count_catalog=len(expected),
                                bundle_mismatch=len(in_bundle ^ expected)))
    man = pd.DataFrame(records)
    man = man.join(cat, on="row")
    with sd.atomic_path(drive_dir / "manifest.csv") as tmp:
        man.to_csv(tmp, index=False)
    if not args.no_local_copy:
        LOCAL_SAMPLE.mkdir(parents=True, exist_ok=True)
        for f in man.file:
            dst = LOCAL_SAMPLE / f
            if not (dst.exists() and dst.stat().st_size == (drive_dir / f).stat().st_size):
                with sd.atomic_path(dst) as tmp:
                    shutil.copyfile(drive_dir / f, tmp)
        with sd.atomic_path(LOCAL_SAMPLE / "manifest.csv") as tmp:
            man.to_csv(tmp, index=False)
    print(f"{len(man)} files; {man['size'].sum() / 1e6:.1f} MB; manifest at {drive_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
