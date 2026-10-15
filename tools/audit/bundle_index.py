"""Index the SMARTS-TESS period bundles (member names and sizes), and stream-extract members.

Each bundle ``hlsp_smarts_tess_ffi_NNN_tess_v1.0_sim.tar.gz`` holds a directory
``NNN/`` of ``smarts-tess-v1.0-RRRRRR.fits`` files, where RRRRRR is the 0-based
row of the (period-sorted) catalog, *not* its "Simulation Number" column.
Bundle NNN holds catalog periods in (NNN-1, NNN] (bundle 001: [0.1, 1]).

Listing one bundle means decompressing it (~5-6 s per ~0.5 GB bundle on the
Data_Vault drive), so a full index of all 180 bundles takes ~17 min. The index
is cached per bundle on the drive at ``<root>/extracted/bundle_index/NNN.csv``
(columns: member, row, size), written atomically, so the full run is
resumable: rerunning skips bundles already indexed.

Usage::

    # index a few bundles
    python tools/audit/bundle_index.py --bundles 1 2 5
    # full index (~17 min; Rae runs this locally): resumable, just rerun after an unplug
    python tools/audit/bundle_index.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smarts_data as sd  # noqa: E402

_ROW_RE = re.compile(r"smarts-tess-v1\.0-(\d{6})\.fits$")


def index_path(root: Path, n: int) -> Path:
    """Cached index CSV of bundle ``n`` on the drive."""
    return Path(root, sd.EXTRACT_DIR, "bundle_index", f"{n:03d}.csv")


def scan_bundle(root: Path, n: int, extract_rows=None, extract_to: Path | None = None) -> pd.DataFrame:
    """One streaming pass over bundle ``n``: list every member, optionally extracting some.

    Parameters
    ----------
    root : SMARTS data root.
    n : bundle number (1..180).
    extract_rows : optional iterable of catalog row numbers to extract.
    extract_to : directory to write extracted FITS into (flat, atomic writes;
        files already present with the right size are skipped).

    Returns
    -------
    DataFrame with columns member (path inside the tar), row (int), size (bytes).
    Also writes the cached index CSV if it doesn't exist yet.
    """
    want = set(int(r) for r in extract_rows) if extract_rows is not None else set()
    if want and extract_to is None:
        raise ValueError("extract_to is required when extract_rows is given")
    rows = []
    with tarfile.open(root / sd.BUNDLE_DIR / sd.bundle_name(n), "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            match = _ROW_RE.search(m.name)
            row = int(match.group(1)) if match else -1
            rows.append((m.name, row, m.size))
            if row in want:
                final = Path(extract_to) / Path(m.name).name
                if final.exists() and final.stat().st_size == m.size:
                    continue
                with sd.atomic_path(final) as tmp:
                    tmp.write_bytes(tf.extractfile(m).read())
    df = pd.DataFrame(rows, columns=["member", "row", "size"])
    ip = index_path(root, n)
    if not ip.exists():
        with sd.atomic_path(ip) as tmp:
            df.to_csv(tmp, index=False)
    return df


def load_index(root: Path, n: int) -> pd.DataFrame:
    """Return bundle ``n``'s index, scanning (and caching) it if needed."""
    ip = index_path(root, n)
    if ip.exists():
        return pd.read_csv(ip)
    return scan_bundle(root, n)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundles", type=int, nargs="+")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    bundles = range(1, sd.N_BUNDLES + 1) if args.all else args.bundles
    t0 = time.time()
    for n in bundles:
        if index_path(root, n).exists():
            continue
        sd.require_drive(root)  # fail cleanly if the drive vanished mid-run
        df = load_index(root, n)
        print(f"bundle {n:03d}: {len(df)} files, rows {df.row.min()}-{df.row.max()}, "
              f"{df['size'].sum() / 1e9:.2f} GB  [{time.time() - t0:.0f} s]", flush=True)
    if args.all:
        allidx = pd.concat([load_index(root, n).assign(bundle=n)
                            for n in range(1, sd.N_BUNDLES + 1)])
        out = Path(root, sd.EXTRACT_DIR, "bundle_index", "all_bundles.csv.gz")
        with sd.atomic_path(out) as tmp:
            allidx.to_csv(tmp, index=False, compression="gzip")
        print(f"{len(allidx)} files, {allidx['size'].sum() / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
