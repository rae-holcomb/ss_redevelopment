#!/usr/bin/env python3
"""compare_scorecard_keys.py -- prove every key of an older scorecard metrics.json is bit-identical in a newer one.

Usage: python tools/compare_scorecard_keys.py OLD_metrics.json NEW_metrics.json
Walks OLD recursively (skipping `config`, `runtime`: wall-clock/commit fields) and requires the same
path to exist in NEW with exactly equal value. New keys in NEW are listed by top-level group only.
Exit code 1 on any difference.
"""
import json
import sys


def walk(old, new, path, bad, counter):
    """Compare old vs new recursively, appending mismatching paths to `bad`."""
    if isinstance(old, dict):
        for k, v in old.items():
            if not path and k in ("config", "runtime"):
                continue
            if not isinstance(new, dict) or k not in new:
                bad.append(f"{path}.{k}: missing")
            else:
                walk(v, new[k], f"{path}.{k}" if path else k, bad, counter)
    else:
        counter[0] += 1
        if old != new and not (old != old and new != new):  # NaN == NaN
            bad.append(f"{path}: {old!r} != {new!r}")


def main():
    """CLI entry point."""
    old, new = (json.load(open(p)) for p in sys.argv[1:3])
    bad, n = [], [0]
    walk(old, new, "", bad, n)
    added = sorted(set(new.get("final", {})) - set(old.get("final", {})))
    print(f"{n[0]} old leaf values compared; {len(bad)} differ; new final.* groups: {added}")
    for b in bad[:20]:
        print("  DIFF", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
