#!/usr/bin/env python3
"""
check_goals.py -- grade a scorecard run against the project goals, and
(optionally) against a baseline run.

This is the "test" in test-driven development for science metrics: a goal
in benchmarks/goals.json is a threshold on one metric in a scorecard
metrics.json. An agent's task is typically "make goal X pass without
tripping any regression guard".

Goals file format (benchmarks/goals.json)
-----------------------------------------
    {
      "goals": [
        {"id": "G1", "description": "...", "dataset": "synthetic:n=200,seed=0",
         "metric": "final.by_bin.<1d.correct", "op": ">=", "target": 0.8,
         "status": "active" | "draft" | "met"}
      ],
      "regression_guard": {"max_drop": 0.02,
                           "metrics": ["final.overall.correct",
                                       {"pattern": "final.by_bin.*.correct", "max_drop": 0.06}]}
    }

`metric` is a dotted path into metrics.json; '*' (in a goal or a
regression-guard path) expands over every key at that level. A wildcard
goal passes only if *every* expanded metric meets the target; the table
shows the worst value. A guard entry may be a plain
pattern (uses the global max_drop) or {"pattern": ..., "max_drop": ...}
-- per-bin metrics need looser limits because each bin holds few stars
(one star in a 50-star bin = 0.02). Only Rae changes goals.json --
agents may propose changes in their report but must not edit it.

Usage
-----
    python benchmarks/check_goals.py runs/new/metrics.json
    python benchmarks/check_goals.py runs/new/metrics.json --baseline benchmarks/baselines/synth200_s0.json

Exit code is 1 if any *active* goal fails or any regression guard trips
(useful for automation), else 0. Draft goals are reported but never fail.
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def get_path(d: dict, path: str):
    """Look up a dotted path like 'final.by_bin.<1d.correct' in nested dict `d`.

    Bin labels contain dots in principle, so this walks greedily: at each
    level it tries the longest dotted prefix that is an existing key.
    Returns None if the path doesn't resolve.
    """
    parts = path.split(".")
    cur = d
    i = 0
    while i < len(parts):
        if not isinstance(cur, dict):
            return None
        for j in range(len(parts), i, -1):
            key = ".".join(parts[i:j])
            if key in cur:
                cur = cur[key]
                i = j
                break
        else:
            return None
    return cur


def expand(d: dict, pattern: str) -> list:
    """Expand a dotted pattern with '*' wildcards into concrete paths present in `d`."""
    parts = pattern.split(".")
    paths = [""]
    for part in parts:
        nxt = []
        for p in paths:
            node = get_path(d, p) if p else d
            if not isinstance(node, dict):
                continue
            keys = list(node) if part == "*" else [part]
            nxt += [f"{p}.{k}" if p else k for k in keys]
        paths = nxt
    return [p for p in paths if isinstance(get_path(d, p), (int, float))]


def _fmt(x) -> str:
    """Format a metric value for the table."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:6.3f}"


def main(argv=None) -> int:
    """CLI entry point; returns the process exit code."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="metrics.json from scorecard.py")
    p.add_argument("--baseline", type=Path, help="baseline metrics.json to compare against")
    p.add_argument("--goals", type=Path, default=ROOT / "benchmarks" / "goals.json")
    args = p.parse_args(argv)

    m = json.loads(args.metrics.read_text())
    base = json.loads(args.baseline.read_text()) if args.baseline else None
    goals = json.loads(args.goals.read_text())
    dataset = m.get("config", {}).get("dataset", "?")
    failed = False

    print(f"Run: {args.metrics}  (dataset {dataset}, commit {m.get('config', {}).get('git_commit')})")
    if base:
        print(f"Baseline: {args.baseline}  (commit {base.get('config', {}).get('git_commit')})")
    print(f"\n{'goal':5} {'status':7} {'result':6} {'value':>6} {'target':>9} {'base':>6}  metric")
    for g in goals.get("goals", []):
        if g.get("dataset") and g["dataset"] != dataset:
            continue
        if "*" in g["metric"]:
            # Wildcard goal: every expanded metric must satisfy op/target.
            # Report the worst value (and which path) so a FAIL is actionable.
            paths = expand(m, g["metric"])
            vals = {p_: get_path(m, p_) for p_ in paths}
            bad = [p_ for p_, x in vals.items() if not OPS[g["op"]](x, g["target"])]
            ok = bool(paths) and not bad
            worst_key = max if g["op"] in ("<=", "<") else min
            v = worst_key(vals.values()) if vals else None
            b = None
            if base:
                bvals = [get_path(base, p_) for p_ in expand(base, g["metric"])]
                b = worst_key(bvals) if bvals else None
            if bad:
                g = dict(g, description=(g.get("description", "") + f" FAILING: {', '.join(bad)}").strip())
        else:
            v = get_path(m, g["metric"])
            ok = v is not None and OPS[g["op"]](v, g["target"])
            b = get_path(base, g["metric"]) if base else None
        verdict = "PASS" if ok else "FAIL"
        if not ok and g.get("status") == "active":
            failed = True
        print(f"{g['id']:5} {g.get('status', 'draft'):7} {verdict:6} {_fmt(v)} {g['op']:>2} {g['target']:<6} "
              f"{_fmt(b)}  {g['metric']}")
        if g.get("description"):
            print(f"{'':21}{g['description']}")

    if base:
        guard = goals.get("regression_guard", {})
        default_drop = guard.get("max_drop", 0.02)
        print("\nRegression guard (fail if a metric drops by more than its limit):")
        print(f"{'base':>7} {'new':>7} {'delta':>7} {'limit':>6}  metric")
        for entry in guard.get("metrics", []):
            pat = entry["pattern"] if isinstance(entry, dict) else entry
            max_drop = entry.get("max_drop", default_drop) if isinstance(entry, dict) else default_drop
            for path in expand(m, pat):
                v, b = get_path(m, path), get_path(base, path)
                if b is None or v is None or (isinstance(b, float) and math.isnan(b)):
                    continue
                delta = v - b
                flag = ""
                if delta < -max_drop:
                    flag, failed = "  <-- REGRESSION", True
                print(f"{b:7.3f} {v:7.3f} {delta:+7.3f} {max_drop:6.3f}  {path}{flag}")

    print("\nOVERALL:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
