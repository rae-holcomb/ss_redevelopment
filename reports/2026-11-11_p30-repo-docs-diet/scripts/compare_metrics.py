"""Key-by-key, NaN-aware comparison of two scorecard metrics.json files, ignoring runtime.* and config.

Usage: python compare_metrics.py BEFORE.json AFTER.json LABEL  -> appends one row to data/metrics_equivalence.csv
"""
import json, math, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "metrics_equivalence.csv"


def flatten(d, prefix=""):
    """Flatten nested dicts/lists into {dotted.key: leaf}."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def same(a, b):
    """Exact equality, treating NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main(before, after, label):
    """Compare and record counts of compared / differing / one-sided keys."""
    skip = lambda k: k.startswith("runtime.") or k == "config" or k.startswith("config.")
    a = {k: v for k, v in flatten(json.load(open(before))).items() if not skip(k)}
    b = {k: v for k, v in flatten(json.load(open(after))).items() if not skip(k)}
    common = sorted(set(a) & set(b))
    diff = [k for k in common if not same(a[k], b[k])]
    row = dict(mode=label, keys_compared=len(common), keys_differing=len(diff),
               only_before=len(set(a) - set(b)), only_after=len(set(b) - set(a)))
    print(row, diff[:10])
    df = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    df = pd.concat([df[df.get("mode", pd.Series(dtype=str)) != label] if len(df) else df, pd.DataFrame([row])])
    df.to_csv(OUT, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
