#!/usr/bin/env python3
"""Build the ranker v0 training-sample manifest (task T10): datasets/train_v0.csv.

Same stratified design as dev_medium (tools/build_dev_sets.py, T2) but drawn from the
**training pool**:

- SIM_ID < 800,000 (the dev sets and holdout use SIM_ID >= 800k, so no galaxy-noise
  source is shared between training and evaluation; datasets/README.md);
- quiet noise sources only (T20 flags: ``classification == "quiet"`` and not
  ``exclude_from_training``);
- minus any noise TIC that also appears in the SIM_ID >= 800k pool (the pool split is
  by SIM_ID, and one TIC turned out to sit on both sides).

- **strict quiet** (Rae, 2026-11-10): minus the 2026-11-09 review's 34 "quiet" noise
  sources with coherent short-period signals (``--coherent-list``). Ranks are drawn on
  the pool *before* this cut and the coherent rows are then dropped, so the selection
  is the non-strict one with each dropped row replaced by the next-ranked row of its
  stratum. ``--keep-coherent`` restores the non-strict pool (rows flagged
  ``noise_coherent_quiet``). Once the flags file's ``exclude_from_training`` covers
  these sources (``periodic_low_amplitude`` class), the list becomes redundant.

Two nested subsets, filled in rank order within every stratum, each balanced on its
own (period bin x amplitude tercile x shear class, then coverage class per bin):

- ``nested_1k`` = the first 1,000 (``train_order`` 0..999): the learning-curve point;
- the remaining 2,000, for 3,000 in all.

Files are extracted to the drive only (``<root>/extracted/train_v0/``); manifest
paths are relative to ``SMARTS_DATA_ROOT``. Reuses build_dev_sets' candidate ->
check -> quota loop with its own check cache (``extracted/t10_train/``), so it is
resumable and every write is atomic.

Usage
-----
    python tools/build_train_sample.py --plan-only      # quotas only, no drive reads
    python tools/build_train_sample.py --n-workers 2    # ~30-60 min of drive time from scratch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))
sys.path.insert(0, str(ROOT / "claude_files"))

import build_dev_sets as bds  # noqa: E402  (pins BLAS on import)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import smarts_data as sd  # noqa: E402

SEED = 20261110
SUBSETS = {"train_1k": 1000, "train_rest": 2000}   # nested: train_1k first
OUT_SUBDIR = "extracted/train_v0"                  # files, relative to the SMARTS data root
BUILD_SUBDIR = "t10_train"                         # check cache + staging, under extracted/
FLAGS = "extracted/galaxy_ids/noise_source_flags.csv.gz"
EXTRA_COLUMNS = ["subset", "nested_1k", "train_order", "noise_tic", "noise_classification",
                 "noise_coherent_quiet"]


def configure_bds():
    """Point build_dev_sets' module-level set definitions at the training sample.

    build_dev_sets reads SET_ORDER / SET_SIZES / BUILD_SUBDIR / HOLDOUT_SUBDIR /
    LOCAL_SETS at call time, so setting them here reuses its allocation, checking and
    manifest code unchanged with a separate cache and a drive-only destination.
    """
    bds.SET_ORDER = tuple(SUBSETS)
    bds.SET_SIZES = dict(SUBSETS)
    bds.BUILD_SUBDIR = BUILD_SUBDIR
    bds.HOLDOUT_SUBDIR = OUT_SUBDIR
    bds.LOCAL_SETS = ()
    if bds.process_bundle is not process_bundle:
        _ORIG["process_bundle"] = bds.process_bundle
        bds.process_bundle = process_bundle


_ORIG = {}


def process_bundle(n: int, rows: list, root: str) -> tuple:
    """Per-bundle worker that applies configure_bds() inside the worker process first.

    Pool workers are spawned (macOS) and re-import build_dev_sets with its T2 defaults,
    so without this the checks would land in T2's cache (extracted/t2_build/). This
    wrapper is what bds.run_checks submits once configure_bds() has run in the parent.
    """
    configure_bds()
    return _ORIG["process_bundle"](n, rows, root)


def load_train_pool(root: Path, coherent_path: Path) -> tuple:
    """Catalog rows eligible for training, with strata and noise columns.

    Returns
    -------
    (pool, info): pool rows with SIM_ID < POOL_CUT, quiet and not excluded, minus
    noise TICs shared with the val/test pool; info has the amplitude cuts (whole
    corpus, as T2) and the filter counts.
    """
    from catalog_to_table import load_catalog
    df = load_catalog(root)
    df["row"] = np.arange(len(df))
    prox = bds.amplitude_proxy(df)
    cuts = np.quantile(prox, [1 / 3, 2 / 3])
    df["amp_proxy"] = prox
    df["amp_tier"] = np.where(prox < cuts[0], "low", np.where(prox < cuts[1], "mid", "high"))
    df["pbin"] = bds.audit_pbin(df.PERIOD.values)
    df["shear_class"] = bds.shear_class(df.DIFFROT.values)
    fl = pd.read_csv(Path(root, FLAGS), usecols=["row", "sim_id", "tic", "classification", "exclude_from_training"])
    df = df.merge(fl.rename(columns={"tic": "noise_tic", "classification": "noise_classification"}),
                  on="row", how="left", validate="1:1")
    if not np.array_equal(df.sim_id.to_numpy(), df.SIM_ID.to_numpy()):
        raise SystemExit("flags file sim_id disagrees with the catalog SIM_ID")
    shared = set(df.noise_tic[df.SIM_ID < bds.POOL_CUT]) & set(df.noise_tic[df.SIM_ID >= bds.POOL_CUT])
    lt = df[df.SIM_ID < bds.POOL_CUT]
    quiet = lt[(lt.noise_classification == "quiet") & ~lt.exclude_from_training.astype(bool)]
    pool = quiet[~quiet.noise_tic.isin(shared)].copy().drop(columns="sim_id")
    pool["stratum"] = pool.pbin + "|" + pool.amp_tier + "|" + pool.shear_class
    pool = pool.rename(columns={"SIM_ID": "sim_id"})
    idx = pd.read_csv(Path(root, sd.EXTRACT_DIR, "bundle_index", "all_bundles.csv.gz"))
    pool = pool.merge(idx[["row", "bundle", "member"]], on="row", how="left", validate="1:1")
    if pool.bundle.isna().any():
        raise SystemExit(f"{int(pool.bundle.isna().sum())} pool rows missing from the bundle index")
    pool["bundle"] = pool.bundle.astype(int)
    coherent = set(pd.read_csv(coherent_path)["tic"].astype("int64"))
    pool["noise_coherent_quiet"] = pool.noise_tic.isin(coherent)
    info = dict(amp_cuts=[float(c) for c in cuts], n_corpus=len(df), n_lt800k=len(lt), n_quiet=len(quiet),
                shared_tics=sorted(int(t) for t in shared),
                n_dropped_shared_tic=int(quiet.noise_tic.isin(shared).sum()),
                n_pool_coherent_quiet=int(pool.noise_coherent_quiet.sum()))
    return pool.reset_index(drop=True), info


def main(argv=None):
    """CLI entry point: select, check (extract to the drive), assign coverage, write the manifest."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="SMARTS data root (default $SMARTS_DATA_ROOT or the Data_Vault path)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-workers", type=int, default=2, help="parallel bundle passes (drive-bound)")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "train_v0.csv")
    ap.add_argument("--coherent-list", type=Path,
                    default=ROOT / "reports" / "2026-11-10_t10-ranker-feature-tables" / "data" / "quiet_coherent.csv")
    ap.add_argument("--keep-coherent", action="store_true",
                    help="non-strict quiet pool: keep the coherent quiet sources (flagged)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args(argv)
    configure_bds()
    root = sd.require_drive(sd.data_root(args.data_root))
    t_start = time.time()
    pool, info = load_train_pool(root, args.coherent_list)
    pool = bds.rank_pool(pool, args.seed)  # ranks over the non-strict pool (stable selection)
    if not args.keep_coherent:
        info["n_pool_dropped_coherent"] = int(pool.noise_coherent_quiet.sum())
        pool = pool[~pool.noise_coherent_quiet].reset_index(drop=True)
    print(f"training pool: {len(pool)} rows ({json.dumps({k: v for k, v in info.items() if k != 'amp_cuts'})})",
          flush=True)
    chosen, checks, quotas, drive_s = bds.select(pool, root, args.seed, args.n_workers, plan_only=args.plan_only)
    if args.plan_only:
        q = pd.DataFrame(quotas).fillna(0).astype(int)
        q["pbin"] = q.index.str.split("|").str[0]
        print(q.groupby("pbin").sum().loc[list(bds.AUDIT_PBIN_LABELS)])
        return
    lib = cov.load_library()
    parts, fallbacks = [], {}
    for si, name in enumerate(bds.SET_ORDER):
        df = bds.assign_coverage(chosen[name], lib, args.seed, 10 + si)
        fallbacks[name] = int(df.coverage_fallback.sum())
        df["subset"] = name
        parts.append(df)
    allrows = pd.concat(parts, ignore_index=True)
    assert not allrows.row.duplicated().any(), "subsets overlap"
    man = bds.finish_manifest(allrows, "train", pool, lib)  # weights over all 3,000 vs the training pool
    extra = allrows.set_index("row")[["subset", "noise_tic", "noise_classification", "noise_coherent_quiet"]]
    man = man.join(extra, on="row")
    man["nested_1k"] = man.subset.eq("train_1k")
    man = man.sort_values(["nested_1k", "row"], ascending=[False, True]).reset_index(drop=True)
    man["train_order"] = np.arange(len(man))
    man["noise_tic"] = man.noise_tic.astype("int64")
    sd.require_drive(root)
    n_copied = bds.install_files(man, "train", root)
    bds.write_csv(man, args.out)
    print(f"{len(man)} rows -> {args.out} ({n_copied} files copied to {OUT_SUBDIR})", flush=True)
    excl = checks[~checks.passed]
    summary = dict(seed=args.seed, subsets=SUBSETS, pool_filter="SIM_ID < 800000 & quiet & ~exclude_from_training "
                   "& noise TIC not in SIM_ID >= 800000" + ("" if args.keep_coherent else " & not coherent-quiet (strict)"), **info, n_pool=len(pool), n_checked=len(checks),
                   n_excluded=len(excl), exclusion_reasons=excl.reason.str.split(":").str[0].value_counts().to_dict(),
                   coverage_fallbacks=fallbacks, n_coherent_quiet_in_sample=int(man.noise_coherent_quiet.sum()),
                   n_noise_tics=int(man.noise_tic.nunique()), data_root="$SMARTS_DATA_ROOT", files=OUT_SUBDIR,
                   drive_seconds_this_run=round(drive_s, 1), wall_seconds_this_run=round(time.time() - t_start, 1))
    with sd.atomic_path(args.out.with_name(args.out.stem + "_build_summary.json")) as tmp:
        tmp.write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
