#!/usr/bin/env python3
"""
rescore_subset.py -- recompute a finished scorecard run's metrics on a subset of its stars, without rerunning.

`scorecard.compute_metrics` is a pure function of per_star.csv + per_method.csv, so restricting both tables
to a manifest's star_ids and recomputing gives exactly what a fresh scorecard on that manifest would report
(same code, same per-star results; checked to 0 differing leaf values on the full `_v7` tables, 2026-11-16).
Used to move results onto the noise-clean manifests (tools/build_clean_manifests.py) at no compute cost.

The output metrics.json keeps the source config, with `dataset` rewritten for the subset manifest, and adds
`config.rescored_from` (source run) and `config.rescored_with` (this tool + date). Run-level wall time is
not recomputed (`config.wall_seconds` is dropped); per-star runtime metrics are recomputed.

Usage
-----
    python tools/rescore_subset.py RUN_DIR_OR_TABLES --manifest datasets/dev_medium_clean.csv \\
        --source-metrics benchmarks/baselines/dev_medium_fullyear_v7.json --out OUT_DIR [--tables-out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import scorecard as sc  # noqa: E402


def subset_dataset_string(dataset: str, old_manifest: str, new_manifest: str) -> str:
    """Swap the manifest path at the head of a config.dataset string."""
    head, sep, rest = dataset.partition(",")
    if Path(head).name != Path(old_manifest).name:
        raise SystemExit(f"source dataset {dataset!r} does not start with {old_manifest}")
    return new_manifest + sep + rest


def rescore(tables: Path, manifest: Path, source_metrics: Path) -> tuple:
    """(metrics dict, per_star subset, per_method subset)."""
    ps = pd.read_csv(tables / "per_star.csv")
    pm = pd.read_csv(tables / "per_method.csv")
    keep = set(pd.read_csv(manifest)["star_id"].astype(str))
    missing = keep - set(ps["star_id"].astype(str))
    if missing:
        raise SystemExit(f"{len(missing)} manifest stars are not in {tables}/per_star.csv")
    ps = ps[ps["star_id"].astype(str).isin(keep)].reset_index(drop=True)
    pm = pm[pm["star_id"].astype(str).isin(keep)].reset_index(drop=True)
    src = json.loads(source_metrics.read_text())
    cfg = dict(src["config"])
    metrics = sc.compute_metrics(ps, pm, cfg.get("rel_tol", sc.ev.REL_TOL))
    old_manifest = cfg["dataset"].partition(",")[0]
    cfg["dataset"] = subset_dataset_string(cfg["dataset"], old_manifest, str(manifest.relative_to(ROOT))
                                           if manifest.is_absolute() else str(manifest))
    cfg.pop("wall_seconds", None)
    cfg["rescored_from"] = str(source_metrics)
    cfg["rescored_with"] = f"tools/rescore_subset.py, {time.strftime('%Y-%m-%d')}"
    metrics["config"] = cfg
    return metrics, ps, pm


def main(argv=None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", type=Path, help="run dir or baseline _tables dir with per_star.csv + per_method.csv")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-metrics", type=Path, required=True, help="the run's metrics.json (for its config)")
    p.add_argument("--out", type=Path, required=True, help="metrics.json path to write")
    p.add_argument("--tables-out", type=Path, default=None, help="also write the subset per_star/per_method here")
    args = p.parse_args(argv)
    metrics, ps, pm = rescore(args.tables, args.manifest, args.source_metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics, indent=2, default=float))
    tmp.replace(args.out)
    if args.tables_out:
        args.tables_out.mkdir(parents=True, exist_ok=True)
        ps.to_csv(args.tables_out / "per_star.csv", index=False)
        pm.to_csv(args.tables_out / "per_method.csv", index=False)
    f = metrics["final"]["overall"]
    print(f"{args.out}: n={f['n']} success {f['success']:.3f} wrong {f['wrong']:.3f}  ({metrics['config']['dataset']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
