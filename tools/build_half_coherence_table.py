#!/usr/bin/env python3
"""Side table of half-coherence features (`half_snr_min`, ...) for an existing T10/P31 feature table.

Ranker v1 (T11) needs `ml_features.half_coherence_features` on every candidate row, but the
P31 feature tables were built without it. Rebuilding the tables would rerun the whole
pipeline; this script only reloads each star's light curve (same loader, same sectors,
via the table's own ``config.json``) and evaluates the feature at every candidate's
``P_eval``. Output joins on (star_id, cand_index).

Resumable (one shard per star, written atomically; ``--resume`` skips done stars) and
parallel (``--n-workers``; use 1 for tables whose light curves are on the Data_Vault drive,
which has dropped under parallel reads).

Usage::

    python tools/build_half_coherence_table.py --table runs/p31_l0x_dev_medium_fullyear \\
        --outdir runs/p31_l0x_dev_medium_fullyear/half_coherence --n-workers 3 --resume
    python tools/build_half_coherence_table.py --table .../p31_tables_l0x/train_3k --data-root \\
        /Volumes/Data_Vault/smarts_data --outdir runs/p31_l0x_train_tables/train_3k/half_coherence \\
        --n-workers 1 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ml_features  # noqa: E402

HALF_COLS = ("half_snr_min", "half_snr_ratio", "half_amp_ratio")


def one_star(task: dict, periods: list, shard_dir: str) -> dict:
    """Load one star and write its half-coherence rows (cand_index, P_eval, features) atomically."""
    import scorecard as sc
    import smarts_data as sd
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    rows, err = [], None
    try:
        _, t, f, *_ = sc._load(task)
        cache = {}
        for ci, p in periods:
            key = round(float(p), 12) if np.isfinite(p) else None
            if key not in cache:
                cache[key] = (ml_features.half_coherence_features(t, f, float(p)) if key is not None
                              else dict.fromkeys(HALF_COLS, np.nan))
            rows.append(dict(star_id=task["star_id"], cand_index=ci, **cache[key]))
    except Exception as exc:  # noqa: BLE001 -- recorded; the run continues
        err = f"{type(exc).__name__}: {exc}"
    df = pd.DataFrame(rows, columns=["star_id", "cand_index", *HALF_COLS])
    with sd.atomic_path(Path(shard_dir) / f"{task['star_id']}.csv") as tmp:
        df.to_csv(tmp, index=False)
    return dict(star_id=task["star_id"], n=len(df), error=err, seconds=time.perf_counter() - t0)


def tasks_for_table(table: Path, data_root: str | None) -> tuple:
    """Scorecard tasks for the stars of a feature table, rebuilt from its config.json."""
    import build_feature_tables as bft
    cfg = json.loads((table / "config.json").read_text())
    ns = argparse.Namespace(manifest=str(ROOT / cfg["manifest"]), data_root=data_root or cfg["data_root"],
                            ignore_manifest_sectors=cfg["ignore_manifest_sectors"], only=cfg.get("only"),
                            limit=None, nonrotator=cfg.get("nonrotator", False),
                            only_set=cfg.get("only_set"), flags=None, coherent_list=None)
    if ns.data_root and not os.path.isabs(ns.data_root):
        ns.data_root = str(ROOT / ns.data_root)
    tasks = bft.build_tasks(ns)
    cands = pd.read_csv(table / "candidates.csv.gz", usecols=["star_id", "cand_index", "P_eval"])
    per = {k: list(zip(g.cand_index, g.P_eval.astype(float))) for k, g in cands.groupby("star_id")}
    tasks = [t for t in tasks if t["star_id"] in per]
    return tasks, per


def combine(outdir: Path) -> int:
    """Concatenate shards into ``half_coherence.csv.gz`` (atomic)."""
    parts = [pd.read_csv(p) for p in sorted((outdir / "shards").glob("*.csv"))]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["star_id", "cand_index", *HALF_COLS])
    tmp = outdir / "half_coherence.csv.gz.tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, outdir / "half_coherence.csv.gz")
    return len(df)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True, type=Path, help="feature-table dir (config.json + candidates)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--data-root", default=None, help="override the config's data_root")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    shard_dir = args.outdir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks, per = tasks_for_table(args.table, args.data_root)
    missing = [t["path"] for t in tasks if "path" in t and not Path(t["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} light-curve files missing (first: {missing[0]}); "
                         "plug in the Data_Vault drive or pass --data-root")
    if args.resume:
        done = {p.stem for p in shard_dir.glob("*.csv")}
        tasks = [t for t in tasks if t["star_id"] not in done]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} stars to do", flush=True)
    t0 = time.perf_counter()
    errs = 0
    if args.n_workers <= 1:
        results = (one_star(t, per[t["star_id"]], str(shard_dir)) for t in tasks)
        for i, r in enumerate(results, 1):
            errs += r["error"] is not None
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {time.perf_counter() - t0:.0f}s errors={errs}", flush=True)
    else:
        with ProcessPoolExecutor(args.n_workers, initializer=determinism.init_worker) as ex:
            futs = [ex.submit(partial(one_star, t, per[t["star_id"]], str(shard_dir))) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                errs += fu.result()["error"] is not None
                if i % 100 == 0 or i == len(tasks):
                    print(f"{i}/{len(tasks)} {time.perf_counter() - t0:.0f}s errors={errs}", flush=True)
    print(f"combined {combine(args.outdir)} rows; errors={errs}", flush=True)


if __name__ == "__main__":
    main()
