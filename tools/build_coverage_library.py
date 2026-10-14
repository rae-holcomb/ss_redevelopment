#!/usr/bin/env python3
"""
build_coverage_library.py -- count real-star TESS Cycle-1 sector patterns (task T16a).

Reads `target_df_shortform.csv` (one row per TIC star, with its observed
sector list in `sec`), keeps each star's Cycle-1 sectors (1-13, the span a
SMARTS light curve emulates), and counts how many stars share each pattern.
This replaces `sector_configs.txt`, which held the same 121 patterns
deduplicated, without frequencies (ss_claude_sandbox.ipynb cell 10).

Output: datasets/coverage/cycle1_patterns.csv with columns
    pattern          "1-2-13" (sorted 1-indexed sector numbers)
    n_stars          stars with exactly this Cycle-1 pattern
    frac_stars       n_stars / stars with >= 1 Cycle-1 sector
    n_sectors        len(pattern)
    consecutive      True if one unbroken run of sectors
    class            coverage class (claude_files/coverage.py: coverage_class)
    first_sector, last_sector, span_sectors, max_missing_run

Provenance and known biases (also in datasets/coverage/README.md):
- Cycle 1 only: later sectors are dropped, so a star seen in S1 and S28 counts as
  single-sector here. Stars observed only after Cycle 1 are dropped.
- Southern ecliptic only (Cycle 1 = southern hemisphere): all 3,438 stars have
  ecliptic latitude < -5.9 deg.
- How target_df_shortform.csv's stars were selected is unknown.

Usage
-----
    python tools/build_coverage_library.py
    python tools/build_coverage_library.py --targets target_df_shortform.csv --out datasets/coverage/cycle1_patterns.csv
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
    """CLI entry point: build the library and write it as CSV with a provenance header."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=Path, default=ROOT / "target_df_shortform.csv")
    p.add_argument("--out", type=Path, default=coverage.DEFAULT_LIBRARY)
    args = p.parse_args(argv)

    targets = pd.read_csv(args.targets, index_col=0)
    lib = coverage.build_library(targets)
    sha = hashlib.sha256(args.targets.read_bytes()).hexdigest()[:16]
    n_any = int(lib["n_stars"].sum())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(f"# Built by tools/build_coverage_library.py from {args.targets.name} "
                 f"(sha256 {sha}...): {len(targets)} stars, {n_any} with >= 1 Cycle-1 sector, "
                 f"{len(lib)} distinct patterns. Cycle 1 = sectors 1-13. See datasets/coverage/README.md.\n")
        lib.to_csv(fh, index=False)
    print(f"{len(targets)} stars in {args.targets.name}; {n_any} with Cycle-1 data; "
          f"{len(lib)} patterns -> {args.out}")
    print(lib.groupby("class", sort=False).agg(patterns=("pattern", "size"), stars=("n_stars", "sum"))
             .assign(frac=lambda d: d.stars / n_any).to_string())


if __name__ == "__main__":
    main()
