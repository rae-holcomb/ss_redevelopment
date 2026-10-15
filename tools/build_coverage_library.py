#!/usr/bin/env python3
"""
build_coverage_library.py -- count real-star TESS sector patterns per observing cycle (task T16a).

Reads `target_df_shortform.csv` (one row per TIC star, with every sector it falls on in
`sec`, computed with tess-point at the star's position; see datasets/coverage/README.md).
For each TESS cycle (year) 1-7 it takes each star's sectors within that cycle (one
"star-season"), expresses them relative to the cycle's first sector, fits them into the
13-sector SMARTS year (coverage.fit_to_year: cycles 4-6 have 14-16 sectors), and counts
how many stars share each pattern. Cycle 1 reproduces the old deduplicated
`sector_configs.txt` patterns, now with frequencies.

Outputs (datasets/coverage/):
    cycle<N>_patterns.csv    one library per cycle (N = 1..7)
    all_cycles_patterns.csv  cycles pooled (n_stars = star-seasons); the default library

Columns: pattern ("1-2-13"), n_stars, frac_stars, n_sectors, consecutive, class,
first_sector, last_sector, span_sectors, max_missing_run, n_shifted, n_truncated,
and cycle (per-cycle files) or n_cycles + cycles (pooled file).

Usage
-----
    python tools/build_coverage_library.py
    python tools/build_coverage_library.py --targets target_df_shortform.csv --outdir datasets/coverage
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

import pandas as pd  # noqa: E402

import coverage  # noqa: E402


def main(argv=None):
    """CLI entry point: build every per-cycle library and the pooled one."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=Path, default=ROOT / "target_df_shortform.csv")
    p.add_argument("--outdir", type=Path, default=coverage.LIBRARY_DIR)
    args = p.parse_args(argv)

    targets = pd.read_csv(args.targets, index_col=0)
    sha = hashlib.sha256(args.targets.read_bytes()).hexdigest()[:16]
    args.outdir.mkdir(parents=True, exist_ok=True)
    libs = []
    for cycle, (first, last) in coverage.CYCLE_SECTORS.items():
        lib = coverage.build_library(targets, cycle=cycle)
        libs.append(lib)
        n_any = int(lib["n_stars"].sum())
        header = (f"# Built by tools/build_coverage_library.py from {args.targets.name} (sha256 {sha}...): "
                  f"TESS cycle {cycle} = sectors {first}-{last} ({last - first + 1} sectors), mapped to SMARTS "
                  f"sectors 1-13 by coverage.fit_to_year. {len(targets)} stars, {n_any} observed in this cycle, "
                  f"{len(lib)} distinct patterns. See datasets/coverage/README.md.")
        _write(args.outdir / f"cycle{cycle}_patterns.csv", lib, header)
        print(f"cycle {cycle} (S{first}-{last}): {n_any} stars, {len(lib)} patterns, "
              f"{int(lib.n_shifted.sum())} shifted, {int(lib.n_truncated.sum())} truncated")
    pooled = coverage.combine_libraries(libs)
    header = (f"# Built by tools/build_coverage_library.py from {args.targets.name} (sha256 {sha}...): "
              f"TESS cycles 1-7 pooled; n_stars counts star-seasons ({int(pooled.n_stars.sum())}), "
              f"{len(pooled)} distinct patterns. Default --coverage library. See datasets/coverage/README.md.")
    _write(args.outdir / "all_cycles_patterns.csv", pooled, header)
    n = pooled["n_stars"].sum()
    print(f"\npooled: {n} star-seasons, {len(pooled)} patterns")
    print(pooled.groupby("class", sort=False).agg(patterns=("pattern", "size"), star_seasons=("n_stars", "sum"))
                .assign(frac=lambda d: d.star_seasons / n).to_string())


def _write(path: Path, lib: pd.DataFrame, header: str):
    """Write a library CSV preceded by a one-line '#' provenance comment."""
    with open(path, "w") as fh:
        fh.write(header + "\n")
        lib.to_csv(fh, index=False)


if __name__ == "__main__":
    main()
