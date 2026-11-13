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

Metric groups added in scorecard v4 (T22/T23) are addressable the same way:
final.by_band_width.<class>.*, final.by_coverage_class.<class>.*,
final.searchable.{overall,by_bin,by_coverage_class}.* and final.unsearchable.*.
Baselines written before v4 lack these keys: a goal or guard on them reports
"n/a" for the baseline and cannot trip, so they are reported, not guarded,
until a v4 baseline exists.

Usage
-----
    python benchmarks/check_goals.py runs/new/metrics.json
    python benchmarks/check_goals.py runs/new/metrics.json --baseline benchmarks/baselines/synth200_s0.json

Exit code is 1 if any *active* goal fails or any regression guard trips
(useful for automation), else 0. Draft goals are reported but never fail.

Other inputs (2026-11-13, Rae's OK)
-----------------------------------
The positional file may also be:

* a headline card, ``headline.json`` from ``tools/headline_card.py``. Its dataset string is
  ``meta.dataset`` (``headline:<manifest stem>:<mode>,population=<pop>``). A goal on it names a
  ``"system"``: a card label, or a kind (``"ranker"`` / ``"hand"``) when exactly one card has that
  kind; pick one of several with ``--system ranker="ranker v1"``. The metric path is inside that
  card's ``values`` (e.g. ``R4_precision.0.1``);
* a runtime summary, ``--summary-json`` from ``benchmarks/profile_pipeline.py`` (dataset
  ``profile:<star list>:<mode>,features=<0|1>``, metrics under ``runtime``).

Extra goal keys:

* ``"target_system": "hand"``: the target is the same metric on another card (e.g. ranker >= hand);
* ``"relative": {"max_drop": 0.02}`` (higher is better) or ``{"max_rise_frac": 0.15}`` (lower is
  better, e.g. seconds per star): compare with the same metric in ``--baseline``. Without a baseline
  the goal reports n/a and never fails.
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


def card_values(data: dict, system: str, overrides: dict | None = None):
    """Return (label, values dict) of the headline card `system` names, or (None, reason).

    `system` is a card label or a kind ("ranker", "hand"); a kind must match exactly one card unless
    `overrides` maps it to a label (from --system KIND=LABEL).
    """
    cards = data.get("cards", {})
    want = (overrides or {}).get(system, system)
    if want in cards:
        return want, cards[want].get("values", {})
    # older cards (e.g. the P28 report's cards.json) have no "kind": use the label's first word
    hits = [lab for lab, c in cards.items() if (c.get("kind") or lab.split()[0]) == want]
    if len(hits) == 1:
        return hits[0], cards[hits[0]].get("values", {})
    if not hits:
        return None, f"no card for system {want!r}"
    return None, f"system {want!r} is ambiguous ({', '.join(hits)}); pass --system {want}=LABEL"


def run_dataset(data: dict) -> str:
    """Dataset string of a scorecard metrics.json, a profile summary or a headline card."""
    if "cards" in data and "meta" in data:
        meta = data["meta"]
        if meta.get("dataset"):
            return meta["dataset"]
        pops = {c.get("population") for c in data["cards"].values()}
        return f"headline:?:{meta.get('mode')},population={pops.pop() if len(pops) == 1 else '?'}"
    return data.get("config", {}).get("dataset", "?")


def goal_root(data: dict | None, g: dict, overrides: dict | None):
    """The dict a goal's metric path is looked up in (the card's values for headline goals)."""
    if data is None:
        return None, None
    if "system" in g and "cards" in data:
        return card_values(data, g["system"], overrides)
    return None, data


def _fmt(x) -> str:
    """Format a metric value for the table."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:6.3f}"


def main(argv=None) -> int:
    """CLI entry point; returns the process exit code."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path,
                   help="metrics.json from scorecard.py, headline.json, or a profile_pipeline summary")
    p.add_argument("--baseline", type=Path, help="baseline metrics.json to compare against")
    p.add_argument("--goals", type=Path, default=ROOT / "benchmarks" / "goals.json")
    p.add_argument("--system", action="append", metavar="KIND=LABEL",
                   help="headline cards: which card a goal's system kind means (repeatable)")
    args = p.parse_args(argv)

    m = json.loads(args.metrics.read_text())
    base = json.loads(args.baseline.read_text()) if args.baseline else None
    goals = json.loads(args.goals.read_text())
    overrides = dict(x.split("=", 1) for x in (args.system or []))
    dataset = run_dataset(m)
    failed = False

    print(f"Run: {args.metrics}  (dataset {dataset}, commit {m.get('config', {}).get('git_commit')})")
    if base:
        print(f"Baseline: {args.baseline}  (commit {base.get('config', {}).get('git_commit')})")
    print(f"\n{'goal':5} {'status':7} {'result':6} {'value':>6} {'target':>9} {'base':>6}  metric")
    for g in goals.get("goals", []):
        if g.get("dataset") and g["dataset"] != dataset:
            continue
        label, root = goal_root(m, g, overrides)
        _, broot = goal_root(base, g, overrides)
        note = ""
        target_txt = f"{g.get('op', ''):>2} {g.get('target', ''):<6}"
        if root is None or isinstance(root, str):
            # headline goal whose system can't be resolved: report, fail only if active
            v, b, ok, note = None, None, False, f" [{label or ''}{root or ''}]"
        elif "*" in g["metric"]:
            # Wildcard goal: every expanded metric must satisfy op/target.
            # Report the worst value (and which path) so a FAIL is actionable.
            paths = expand(root, g["metric"])
            vals = {p_: get_path(root, p_) for p_ in paths}
            bad = [p_ for p_, x in vals.items() if not OPS[g["op"]](x, g["target"])]
            ok = bool(paths) and not bad
            worst_key = max if g["op"] in ("<=", "<") else min
            v = worst_key(vals.values()) if vals else None
            b = None
            if broot is not None and not isinstance(broot, str):
                bvals = [get_path(broot, p_) for p_ in expand(broot, g["metric"])]
                b = worst_key(bvals) if bvals else None
            if bad:
                g = dict(g, description=(g.get("description", "") + f" FAILING: {', '.join(bad)}").strip())
        else:
            v = get_path(root, g["metric"])
            b = get_path(broot, g["metric"]) if isinstance(broot, dict) else None
            if "relative" in g:
                rel = g["relative"]
                if b is None or v is None:
                    ok, note = None, " [n/a: needs --baseline with this metric]"
                elif "max_rise_frac" in rel:
                    ok = v <= b * (1 + rel["max_rise_frac"])
                    target_txt = f"<= base*{1 + rel['max_rise_frac']:.2f}"
                else:
                    ok = v >= b - rel.get("max_drop", 0.0)
                    target_txt = f">= base-{rel.get('max_drop', 0.0):g}"
            elif "target_system" in g:
                tlab, troot = card_values(m, g["target_system"], overrides)
                t = get_path(troot, g["metric"]) if isinstance(troot, dict) else None
                ok = v is not None and t is not None and OPS[g["op"]](v, t)
                target_txt = f"{g['op']:>2} {_fmt(t).strip():<6}"
                note = f" [{label} vs {tlab or troot}]"
            else:
                ok = v is not None and OPS[g["op"]](v, g["target"])
                if label:
                    note = f" [{label}]"
        verdict = "n/a" if ok is None else ("PASS" if ok else "FAIL")
        if ok is False and g.get("status") == "active":
            failed = True
        print(f"{g['id']:5} {g.get('status', 'draft'):7} {verdict:6} {_fmt(v)} {target_txt} "
              f"{_fmt(b)}  {g['metric']}{note}")
        if g.get("description"):
            print(f"{'':21}{g['description']}")

    if base and "cards" not in m:
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
