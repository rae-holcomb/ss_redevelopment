#!/usr/bin/env python3
"""compare_runs_exact.py -- prove two runs are bit-identical except for wall-clock fields (T12).

Stricter than tools/compare_runs.py: walks EVERY value, recursively, NaN-aware and exact.

Scorecard runs (directories with rows.jsonl + metrics.json):
    every value of every star's rows.jsonl record (incl. per-method candidate
    outcomes and the per-candidate gate records) and every metrics.json key,
    skipping only runtime keys: `seconds`, `fit_seconds` (any depth) and the
    metrics.json `config` block (commit, wall time, timestamps).
Feature-table runs (directories with shards/*.pkl, tools/build_feature_tables.py):
    every star dict value (except `seconds`) and every cell of every candidate
    DataFrame (same columns, same order, exact values).

Usage
-----
    python tools/compare_runs_exact.py RUN_A RUN_B [--json out.json]

Prints a summary (values compared, values differing, runtime keys skipped) and
exits 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RUNTIME_KEYS = {"seconds", "fit_seconds"}
# Input-file provenance: the absolute path differs between checkouts (worktrees resolve
# data/smarts_dev to different absolute roots), so `path` values are compared by file name.
PATH_KEYS = {"path"}


def _equal_leaf(a, b) -> bool:
    """Exact equality for JSON-ish leaves; NaN equals NaN; int 1 != float 1.0 is NOT flagged."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return type(a) is type(b) and a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                              and not isinstance(a, bool) and a == b)


def walk(a, b, path: str, stats: Counter, diffs: list, skip_top=()):
    """Recursively compare a vs b, counting leaves and recording differing paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in RUNTIME_KEYS or (not path and k in skip_top):
                stats["runtime_skipped"] += 1
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: only in {'A' if k in a else 'B'}")
                continue
            if k in PATH_KEYS and isinstance(a[k], str) and isinstance(b[k], str):
                stats["leaves"] += 1
                if Path(a[k]).name != Path(b[k]).name:
                    diffs.append(f"{path}.{k}: {a[k]!r} != {b[k]!r}")
                continue
            walk(a[k], b[k], f"{path}.{k}" if path else str(k), stats, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", stats, diffs)
    else:
        stats["leaves"] += 1
        if not _equal_leaf(a, b):
            diffs.append(f"{path}: {a!r} != {b!r}")


def _rows(run: Path) -> dict:
    """{star_id: record} from a scorecard rows.jsonl."""
    out = {}
    for line in (run / "rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["star_id"]] = r
    return out


def compare_scorecards(a: Path, b: Path) -> dict:
    """Compare two scorecard output directories; returns a summary dict."""
    stats, diffs = Counter(), []
    ra, rb = _rows(a), _rows(b)
    if set(ra) != set(rb):
        diffs.append(f"star sets differ: {len(set(ra) ^ set(rb))} stars in only one run")
    for sid in sorted(set(ra) & set(rb)):
        walk(ra[sid], rb[sid], f"rows[{sid}]", stats, diffs)
    row_leaves = stats["leaves"]
    ma, mb = (json.loads((p / "metrics.json").read_text()) for p in (a, b))
    walk(ma, mb, "", stats, diffs, skip_top=("config", "runtime"))
    return dict(kind="scorecard", n_stars=len(set(ra) & set(rb)), row_values=row_leaves,
                metric_values=stats["leaves"] - row_leaves, values_compared=stats["leaves"],
                runtime_keys_skipped=stats["runtime_skipped"], n_diffs=len(diffs), diffs=diffs[:500])


def _frame_equal(x: pd.DataFrame, y: pd.DataFrame, where: str, stats: Counter, diffs: list):
    """Exact cell-by-cell comparison of two candidate DataFrames."""
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        diffs.append(f"{where}: shape/columns differ {x.shape} vs {y.shape}")
        return
    for c in x.columns:
        for i, (u, v) in enumerate(zip(x[c].tolist(), y[c].tolist())):
            stats["leaves"] += 1
            if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
                same = np.array_equal(np.asarray(u), np.asarray(v), equal_nan=True)
            else:
                same = _equal_leaf(u, v)
            if not same:
                diffs.append(f"{where}[{i}].{c}: {u!r} != {v!r}")


def compare_feature_tables(a: Path, b: Path) -> dict:
    """Compare two build_feature_tables output directories shard by shard."""
    stats, diffs = Counter(), []
    sa = {p.stem: p for p in (a / "shards").glob("*.pkl")}
    sb = {p.stem: p for p in (b / "shards").glob("*.pkl")}
    if set(sa) != set(sb):
        diffs.append(f"shard sets differ: {len(set(sa) ^ set(sb))}")
    for sid in sorted(set(sa) & set(sb)):
        with open(sa[sid], "rb") as fa, open(sb[sid], "rb") as fb:
            da, db = pickle.load(fa), pickle.load(fb)
        walk(da["star"], db["star"], f"star[{sid}]", stats, diffs)
        _frame_equal(da["candidates"], db["candidates"], f"cands[{sid}]", stats, diffs)
    return dict(kind="feature_tables", n_stars=len(set(sa) & set(sb)), values_compared=stats["leaves"],
                runtime_keys_skipped=stats["runtime_skipped"], n_diffs=len(diffs), diffs=diffs[:500])


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write the summary here")
    args = ap.parse_args(argv)
    fn = compare_feature_tables if (args.run_a / "shards").is_dir() else compare_scorecards
    summary = fn(args.run_a, args.run_b)
    summary.update(run_a=str(args.run_a), run_b=str(args.run_b))
    print(f"{summary['kind']}: {summary['n_stars']} stars, {summary['values_compared']} values compared, "
          f"{summary['n_diffs']} differ ({summary['runtime_keys_skipped']} runtime fields skipped)")
    for d in summary["diffs"][:20]:
        print("  DIFF", d)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
    sys.exit(1 if summary["n_diffs"] else 0)


if __name__ == "__main__":
    main()
