"""Build the committed per-star detectability table the scorecard joins (v6 `detectable` flag; T26/P26).

For each dev manifest (dev_small, dev_medium) and each coverage mode the scorecard can run it in
(full-year = all native sectors; assigned = the manifest's `sectors` column), computes the T26
detectability statistic ``snr_rot = D_band * sqrt(N_cyc)`` with ``tools/audit/detectability.py``.
That needs the T20 noise stacks on the Data_Vault drive (read-only) and the local dev files.

The table is keyed by (star_id, sectors): `sectors` is the pattern the loader actually used,
in the same "1-2-13" form as the scorecard's per-star `sectors` column, so any run whose loaded
coverage matches a row gets the value; other coverage draws get no value (flag left empty).

Output: ``benchmarks/detectability_dev.csv`` (atomic write). Per-(manifest, mode) intermediate
CSVs and resumable checkpoints live on the drive under ``extracted/t26_detectability/``.

Usage (~3 min per 1,000 stars per mode at 2 workers)::

    $PY tools/build_detectability_table.py --n-workers 2
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "audit"))
sys.path.insert(0, str(REPO / "claude_files"))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

import detectability as det  # noqa: E402
import smarts_data as sd  # noqa: E402

OUT = REPO / "benchmarks" / "detectability_dev.csv"
MANIFESTS = ("dev_small", "dev_medium")
MODES = ("fullyear", "assigned")
COLUMNS = ["star_id", "sim_id", "sectors", "modes", "snr_rot", "D_band", "n_cycles_obs", "D_peak",
           "noise_tic", "stack_which"]


def combine(frames: list) -> pd.DataFrame:
    """Concatenate per-(manifest, mode) outputs into one table keyed by (star_id, sectors).

    Rows with the same key (e.g. an assigned pattern that is the full native set) are identical
    computations, so they are merged and `modes` lists every mode that produced the key.
    """
    t = pd.concat(frames, ignore_index=True)
    bad = t.error.fillna("") != ""
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} stars failed: {t.loc[bad, ['star_id', 'mode', 'error']].head()}")
    t = t.rename(columns={"D_band_sqrt_ncyc": "snr_rot"})
    modes = t.groupby(["star_id", "sectors"])["mode"].apply(lambda m: "|".join(sorted(set(m)))).rename("modes")
    t = t.drop_duplicates(["star_id", "sectors"]).merge(modes.reset_index(), on=["star_id", "sectors"])
    return t[COLUMNS].sort_values(["star_id", "sectors"]).reset_index(drop=True)


def main():
    """CLI: compute every (manifest, mode) and write benchmarks/detectability_dev.csv."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="SMARTS data root ($SMARTS_DATA_ROOT)")
    ap.add_argument("--data-dir", default=str(REPO / "data" / "smarts_dev"))
    ap.add_argument("--n-workers", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    root = sd.data_root(a.data_root)
    det.check_stacks_readable(root)
    inter = sd.extract_dir(root, "t26_detectability", "tables")
    frames = [det.run(str(REPO / "datasets" / f"{m}.csv"), mode, inter / f"{m}_{mode}.csv", a.data_dir, root,
                      n_workers=a.n_workers, resume=True)
              for m in MANIFESTS for mode in MODES]
    t = combine(frames)
    with sd.atomic_path(Path(a.out)) as tmp:
        t.to_csv(tmp, index=False)
    print(f"wrote {a.out}: {len(t)} rows, {t.star_id.nunique()} stars")


if __name__ == "__main__":
    main()
