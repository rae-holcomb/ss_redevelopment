#!/usr/bin/env python3
"""
build_clean_manifests.py -- the frozen dev manifests minus contaminated-noise stars (Rae, 2026-11-16).

SMARTS injects simulated rotation into real TESS light curves ("noise sources"). T20 flagged sources that
already carried a periodic or contaminating signal before injection (`exclude_from_training`, the
`noise_exclude` column of `benchmarks/noise_sources_dev.csv`). Rae: never develop or judge the pipeline on
those stars. This writes, for each frozen dev manifest,

    datasets/<name>_clean.csv   = datasets/<name>.csv rows whose sim_id is NOT noise_exclude (same columns,
                                  same order)

The frozen manifests themselves are untouched. The scorecard refuses a manifest that holds noise_exclude stars
(`scorecard.check_no_noise_excluded`) unless --allow-noise-excluded is passed.

The holdout has no sim_id column and no noise-table rows by design; filter it the same way when Rae opens it.

Usage
-----
    python tools/build_clean_manifests.py            # rewrites datasets/dev_small_clean.csv, dev_medium_clean.csv
    python tools/build_clean_manifests.py --check    # exit 1 if the committed files are stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOISE_TABLE = ROOT / "benchmarks" / "noise_sources_dev.csv"
MANIFESTS = ("dev_small", "dev_medium")


def clean_manifest(man: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Rows of `man` whose sim_id is in the noise table and not noise_exclude; order and columns kept."""
    flags = noise.set_index("sim_id")["noise_exclude"].astype(bool)
    missing = set(man["sim_id"]) - set(flags.index)
    if missing:
        raise SystemExit(f"{len(missing)} manifest sim_ids have no noise-table row (rebuild "
                         "benchmarks/noise_sources_dev.csv with tools/build_noise_table.py)")
    return man[~man["sim_id"].map(flags)].reset_index(drop=True)


def main(argv=None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="only verify the committed *_clean.csv files")
    args = p.parse_args(argv)
    noise = pd.read_csv(NOISE_TABLE)
    stale = False
    for name in MANIFESTS:
        src = ROOT / "datasets" / f"{name}.csv"
        out = ROOT / "datasets" / f"{name}_clean.csv"
        man = pd.read_csv(src)
        clean = clean_manifest(man, noise)
        text = clean.to_csv(index=False)
        if args.check:
            ok = out.exists() and out.read_text() == text
            stale |= not ok
            print(f"{out.name}: {'ok' if ok else 'STALE'}")
            continue
        tmp = out.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(out)
        print(f"{out.relative_to(ROOT)}: {len(clean)} of {len(man)} rows ({len(man) - len(clean)} noise_exclude dropped)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
