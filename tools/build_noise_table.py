#!/usr/bin/env python3
"""
build_noise_table.py -- per-star noise-source table for scorecard v5 (tasks T24/T25).

Joins each star of the dev_small and dev_medium manifests (never the holdout) to its SMARTS
noise-source light curve via SIM_ID -> noise TIC, using the T20 flags file

    <SMARTS_DATA_ROOT>/extracted/galaxy_ids/noise_source_flags.csv.gz

and writes a small committed table, `benchmarks/noise_sources_dev.csv` (one row per sim_id),
so the scorecard never needs the external drive. Columns:

    sim_id, tic, noise_class (T20 `classification`), noise_periodic, noise_exclude
    (`exclude_from_training`), noise_P_ref (days; pipeline period on the noise stack, else LS
    peak; present even for `quiet` sources, where it is a weak LS peak), noise_amp_ppm,
    noise_persist, noise_near_lmc, noise_quiet_coherent, in_dev_small, in_dev_medium.

`noise_quiet_coherent` marks the ~34 `quiet` sources that carry coherent short-period signals
(0.1-2.2 d, ~300 ppm, S/N > 20 in both halves of the year) that T20's `periodic` cut misses.
The list is `benchmarks/noise_quiet_coherent.csv`, copied verbatim from
`reports/2026-11-09_smarts-note-review/checks/quiet_coherent.csv` on branch
claude/smarts-note-review-2d28d8 (commit fd543bf, 2026-11-09; generator c07c_quiet_coherent.py).
It is matched by TIC.

Usage
-----
    python tools/build_noise_table.py                      # drive root: $SMARTS_DATA_ROOT or /Volumes/Data_Vault/smarts_data
    python tools/build_noise_table.py --root "/Volumes/Data_Vault 1/smarts_data"
The output is written atomically. Stops with a clear message if the drive is not mounted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = "/Volumes/Data_Vault/smarts_data"
FLAGS_REL = "extracted/galaxy_ids/noise_source_flags.csv.gz"
MANIFESTS = {"in_dev_small": ROOT / "datasets" / "dev_small.csv",
             "in_dev_medium": ROOT / "datasets" / "dev_medium.csv"}


def build(flags: pd.DataFrame, coherent_tics, manifests: dict) -> pd.DataFrame:
    """Return the per-sim_id noise table for the stars in `manifests`.

    Parameters
    ----------
    flags : DataFrame with T20's noise_source_flags columns (sim_id, tic, classification, ...).
    coherent_tics : iterable of TICs of the quiet-but-coherent sources.
    manifests : {flag column name: DataFrame with a `sim_id` column}.
    """
    ids = pd.concat([m[["sim_id"]] for m in manifests.values()]).drop_duplicates()
    t = ids.merge(flags, on="sim_id", how="left")
    if t["tic"].isna().any():
        raise SystemExit(f"{int(t['tic'].isna().sum())} manifest sim_ids missing from the flags file")
    out = pd.DataFrame({
        "sim_id": t["sim_id"], "tic": t["tic"].astype("int64"),
        "noise_class": t["classification"], "noise_periodic": t["periodic"].astype(bool),
        "noise_exclude": t["exclude_from_training"].astype(bool), "noise_P_ref": t["P_ref"],
        "noise_amp_ppm": t["amp1_ppm_all"], "noise_persist": t["persist"],
        "noise_near_lmc": t["near_lmc"].astype(bool)})
    out["noise_quiet_coherent"] = out["tic"].isin(set(int(x) for x in coherent_tics))
    for col, m in manifests.items():
        out[col] = out["sim_id"].isin(set(m["sim_id"]))
    return out.sort_values("sim_id").reset_index(drop=True)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("SMARTS_DATA_ROOT", DEFAULT_ROOT))
    ap.add_argument("--coherent", type=Path, default=ROOT / "benchmarks" / "noise_quiet_coherent.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "noise_sources_dev.csv")
    a = ap.parse_args(argv)
    flags_path = Path(a.root) / FLAGS_REL
    if not flags_path.exists():
        sys.exit(f"{flags_path} not found: plug in the Data_Vault drive (or pass --root / set SMARTS_DATA_ROOT).")
    flags = pd.read_csv(flags_path)
    table = build(flags, pd.read_csv(a.coherent)["tic"],
                  {k: pd.read_csv(v) for k, v in MANIFESTS.items()})
    tmp = a.out.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(a.out)
    print(f"wrote {a.out}: {len(table)} stars; classes {table.noise_class.value_counts().to_dict()}; "
          f"quiet_coherent {int(table.noise_quiet_coherent.sum())}")


if __name__ == "__main__":
    main()
