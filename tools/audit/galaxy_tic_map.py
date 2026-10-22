"""Map every SMARTS file (catalog row) to its noise light curve's TIC ID, and check it against T20's fingerprints.

Zach Claytor's ``galaxy_files.csv`` maps the *original* simulation number
(= catalog ``SIM_ID``) to the noise file ``TIC<id>_full.lc.fits`` (see
``docs/SMARTS_REFERENCE.md`` s7). This writes, on the drive:

- ``extracted/galaxy_ids/galaxy_map.csv.gz``: row, sim_id, tic, galaxy (0..1718,
  TICs sorted ascending), pool (lt800k / ge800k), bundle;
- ``extracted/galaxy_ids/tic_table.csv``: per TIC: galaxy, n_files, n_lt800k, n_ge800k.

``--fingerprint-groups DIR`` additionally compares a ``galaxy_groups.py``
output (blind fingerprint clustering) with the TIC map: impure fingerprint
groups, TICs split over several groups, and the TIC IDs of sibling pairs.

Usage::

    $PY tools/audit/galaxy_tic_map.py [--fingerprint-groups DIR --report-dir reports/<folder>/data]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smarts_data as sd  # noqa: E402

GALAXY_FILES = "galaxy_files.csv"
OUT_DIR = "galaxy_ids"


def bundle_of_period(p: np.ndarray) -> np.ndarray:
    """Bundle number holding catalog period ``p`` (bundle NNN = (NNN-1, NNN] d; bundle 1 = [0.1, 1])."""
    return np.clip(np.ceil(p).astype(int), 1, sd.N_BUNDLES)


def build_map(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-file (row -> TIC) map and per-TIC table from the catalog and galaxy_files.csv."""
    cat = np.load(Path(root, sd.EXTRACT_DIR, "catalog", "smarts_catalog.npz"))
    sim = cat["SIM_ID"].astype(np.int64)
    gf = pd.read_csv(Path(root, GALAXY_FILES))
    tic_by_sim = np.zeros(len(gf), np.int64)
    tic_by_sim[gf["Simulation Number"].values] = gf.file.str.extract(r"TIC(\d+)_full")[0].astype(np.int64).values
    tic = tic_by_sim[sim]
    tics, galaxy = np.unique(tic, return_inverse=True)
    m = pd.DataFrame(dict(row=np.arange(len(sim)), sim_id=sim, tic=tic, galaxy=galaxy,
                          pool=np.where(sim >= 800_000, "ge800k", "lt800k"),
                          bundle=bundle_of_period(cat["PERIOD"])))
    t = m.groupby("galaxy").agg(tic=("tic", "first"), n_files=("row", "size"),
                                n_lt800k=("pool", lambda s: int((s == "lt800k").sum())),
                                n_ge800k=("pool", lambda s: int((s == "ge800k").sum()))).reset_index()
    return m, t


def compare_fingerprints(m: pd.DataFrame, gdir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Cross-tab of fingerprint galaxy_id vs TIC: summary, impure groups, split TICs (and sibling TICs)."""
    a = pd.read_csv(gdir / "galaxy_assign.csv.gz").merge(m[["row", "tic", "sim_id"]], on="row",
                                                          suffixes=("", "_map"))
    assert (a.sim_id == a.sim_id_map).all(), "SIM_ID mismatch between groups and catalog"
    x = a[a.galaxy_id >= 0].groupby(["galaxy_id", "tic"]).size().rename("n").reset_index()
    per_g = x.groupby("galaxy_id").agg(n_tics=("tic", "size"), n=("n", "sum"), n_major=("n", "max"))
    per_t = x.groupby("tic").agg(n_groups=("galaxy_id", "size"), n=("n", "sum"), n_major=("n", "max"))
    impure = x[x.galaxy_id.isin(per_g.index[per_g.n_tics > 1])].sort_values(["galaxy_id", "n"],
                                                                          ascending=[True, False])
    split = x[x.tic.isin(per_t.index[per_t.n_groups > 1])].sort_values(["tic", "n"], ascending=[True, False])
    sib = pd.read_csv(gdir / "galaxy_siblings.csv")
    maj = x.sort_values("n").groupby("galaxy_id").tail(1).set_index("galaxy_id").tic
    sib = sib.assign(tic_a=maj.reindex(sib.galaxy_a).values, tic_b=maj.reindex(sib.galaxy_b).values)
    summ = dict(n_files=len(a), n_assigned=int((a.galaxy_id >= 0).sum()),
                n_groups=int(len(per_g)), n_groups_single_tic=int((per_g.n_tics == 1).sum()),
                files_in_majority_tic=float(per_g.n_major.sum() / per_g.n.sum()),
                n_tics=int(len(per_t)), n_tics_split=int((per_t.n_groups > 1).sum()),
                n_sibling_pairs=int(len(sib)), n_sibling_pairs_same_tic=int((sib.tic_a == sib.tic_b).sum()),
                sibling_tic_abs_diff_median=float(np.median(np.abs(sib.tic_a - sib.tic_b))))
    return summ, pd.concat([impure.assign(kind="impure_group"), split.assign(kind="split_tic")]), sib


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fingerprint-groups", default=None, help="galaxy_groups.py output directory")
    ap.add_argument("--report-dir", default=None, help="where to write the comparison tables")
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    m, t = build_map(root)
    out = sd.extract_dir(root, OUT_DIR)
    with sd.atomic_path(out / "galaxy_map.csv.gz") as tmp:
        m.to_csv(tmp, index=False, compression="gzip")
    with sd.atomic_path(out / "tic_table.csv") as tmp:
        t.to_csv(tmp, index=False)
    print(f"{len(m)} files -> {len(t)} TICs; shared by both pools: "
          f"{int(((t.n_lt800k > 0) & (t.n_ge800k > 0)).sum())}")
    if args.fingerprint_groups:
        summ, mism, sib = compare_fingerprints(m, Path(args.fingerprint_groups))
        print(json.dumps(summ, indent=1))
        rd = Path(args.report_dir or out)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "fingerprint_vs_tic.json").write_text(json.dumps(summ, indent=1))
        mism.to_csv(rd / "fingerprint_vs_tic_mismatches.csv", index=False)
        sib.to_csv(rd / "sibling_pairs_tic.csv", index=False)


if __name__ == "__main__":
    main()
