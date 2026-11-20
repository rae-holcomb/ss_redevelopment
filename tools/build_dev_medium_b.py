#!/usr/bin/env python3
"""Build the second noise-clean dev set (task P57): datasets/dev_medium_b_clean.csv.

dev_medium_clean (874 stars) has judged every method change since mid-October, so the
pipeline has adapted to it by an unknown amount. dev_medium_b is an independent ~1,000-star
set drawn the same way, so thresholds can be tuned on one set and judged on the other
(Rae, 2026-11-19).

Same design as dev_medium (tools/build_dev_sets.py, T2), reusing its code unchanged:

- pool: catalog rows with SIM_ID >= 800,000 (so disjoint from train_v0 and every other
  SIM_ID < 800k training sample by construction);
- strata: audit period bin x amplitude-proxy tercile (cuts over the whole corpus) x shear
  class; quotas: equal shares for the five bins < 60 d, ~1/7 for 60-180 d, equal cells
  within a bin (water-filled under capacity);
- checks: corrupt galaxy noise (raw max |flux - 1| > 0.1) and native / whole sectors;
- coverage: a real-star sector pattern per row, classes balanced per period bin;
- weights: sampling_weight against the *whole* SIM_ID >= 800k pool (same reference as
  dev_medium), coverage_weight against the real star-season class mix;
- columns: exactly dev_medium's (build_dev_sets.MANIFEST_COLUMNS).

What differs:

- seed 20261120 (dev_medium: 20261015);
- the pool first drops every row in dev_small, dev_medium and the holdout, every row the
  T2 build already found corrupt (datasets/exclusions_log.csv), every row whose noise
  source is ``exclude_from_training`` in the T20 flags (the ``noise_exclude`` stars).
  Noise TICs *are* shared with the other sets by default: the SIM_ID >= 800k pool has only
  719 noise TICs (620 clean), each reused ~280 times, and dev_small + dev_medium + holdout
  already use 658 of them, so dropping shared TICs leaves 10 stars < 1 d and 110 at 1-3 d
  (``--exclude-shared-noise-tic`` does it anyway; dev_medium itself puts 874 stars on 447 TICs);
- a bin whose remaining capacity is below its target gets its capacity, not an error
  (the < 1 d bin is nearly used up by T2), so the set can come out slightly under 1,000;
- one output, already clean, so no unclean twin exists.

Drive I/O runs with one worker by default (the drive dropped under parallel reads on
2026-11-20; the P57 build used --n-workers 2 with Rae's OK on an idle machine); checks are cached per bundle on the drive under
``extracted/p57_build/checks/`` (atomic writes), staged files under
``extracted/p57_build/candidates/``, so an interrupted build resumes where it stopped. The
drive mount is re-checked after every bundle.

Outputs (datasets/): ``dev_medium_b_clean.csv``, ``dev_medium_b_strata_counts.csv``,
``dev_medium_b_exclusions_log.csv``, ``dev_medium_b_build_summary.json``; FITS files copied to
``data/smarts_dev/`` (manifest paths relative to that data root, like dev_medium).

Usage::

    python tools/build_dev_medium_b.py --plan-only   # pool filters and quotas, no extraction
    python tools/build_dev_medium_b.py               # build (~30-45 min of drive time, 1 worker)
    python tools/build_dev_medium_b.py --check       # rebuild in memory from the drive's check
                                                     # cache; exit 1 if the committed files differ
"""
from __future__ import annotations

import argparse
import json
import math
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

SEED = 20261120
NAME = "dev_medium_b"
SIZE = 1000
BUILD_SUBDIR = "p57_build"                         # check cache + staging, under extracted/
FLAGS = "extracted/galaxy_ids/noise_source_flags.csv.gz"
DATASETS = ROOT / "datasets"
EXISTING_SETS = ("dev_small", "dev_medium", "holdout")
OUT_FILES = {
    "manifest": f"{NAME}_clean.csv",
    "strata": f"{NAME}_strata_counts.csv",
    "exclusions": f"{NAME}_exclusions_log.csv",
    "summary": f"{NAME}_build_summary.json",
}
COVERAGE_SET_INDEX = 20                            # seeds assign_coverage's per-bin generators

_ORIG = {}


def configure_bds():
    """Point build_dev_sets' module-level set definitions at dev_medium_b.

    build_dev_sets reads SET_ORDER / SET_SIZES / BUILD_SUBDIR / LOCAL_SETS and calls
    ``allocate`` / ``process_bundle`` through module globals at call time, so setting them
    here reuses its candidate -> check -> quota loop with a separate cache, a local (laptop)
    destination and the capacity-clipped allocation below.
    """
    bds.SET_ORDER = (NAME,)
    bds.SET_SIZES = {NAME: SIZE}
    bds.BUILD_SUBDIR = BUILD_SUBDIR
    bds.LOCAL_SETS = (NAME,)
    if bds.allocate is not allocate:
        _ORIG["allocate"] = bds.allocate
        bds.allocate = allocate
    if bds.process_bundle is not process_bundle:
        _ORIG["process_bundle"] = bds.process_bundle
        bds.process_bundle = process_bundle


def process_bundle(n: int, rows: list, root: str) -> tuple:
    """Per-bundle worker that applies configure_bds() first (spawned workers re-import bds)."""
    configure_bds()
    return _ORIG["process_bundle"](n, rows, root)


def allocate(caps: dict, seed: int) -> dict:
    """build_dev_sets.allocate, except a bin short of capacity takes all it has.

    Same generators and water-filling as T2, so a bin with enough capacity gets exactly the
    allocation T2's code would give; only a bin whose remaining cells hold fewer rows than
    its target is clipped to its capacity (instead of raising).
    """
    left = dict(caps)
    out = {}
    for si, name in enumerate(bds.SET_ORDER):
        q = {}
        for bi, (pb, n_bin) in enumerate(bds.bin_targets(bds.SET_SIZES[name]).items()):
            keys = [f"{pb}|{a}|{s}" for a, s in bds.CELLS]
            rng = np.random.default_rng([seed, 1000 + si, bi])
            capv = np.array([max(left.get(k, 0), 0) for k in keys])
            alloc = bds.waterfill(min(int(n_bin), int(capv.sum())), capv, rng)
            for k, a in zip(keys, alloc):
                q[k] = int(a)
                left[k] = left.get(k, 0) - int(a)
        out[name] = q
    return out


def load_existing(datasets: Path = DATASETS) -> dict:
    """{set name: manifest} for dev_small, dev_medium and the holdout (the rows B must avoid)."""
    return {n: pd.read_csv(datasets / f"{n}.csv", usecols=["row", "sim_id"]) for n in EXISTING_SETS}


def filter_pool(pool: pd.DataFrame, flags: pd.DataFrame, existing: dict, t2_excluded: set,
                allow_shared_tic: bool = False) -> tuple:
    """Drop rows dev_medium_b must not use; return (filtered pool, counts per filter).

    Parameters
    ----------
    pool : bds.load_pool output (SIM_ID >= 800k, with ``row``).
    flags : T20 noise flags with columns row, sim_id, tic, exclude_from_training.
    existing : {name: DataFrame with ``row``} of the sets to stay disjoint from.
    t2_excluded : catalog rows the T2 build found corrupt / unloadable.
    allow_shared_tic : keep rows whose noise TIC also serves a star of ``existing``.
    """
    p = pool.merge(flags[["row", "tic", "exclude_from_training"]].rename(columns={"tic": "noise_tic"}),
                   on="row", how="left", validate="1:1")
    if p.noise_tic.isna().any():
        raise SystemExit(f"{int(p.noise_tic.isna().sum())} pool rows missing from the noise flags file")
    used_rows = set().union(*(set(m.row) for m in existing.values()))
    used_tics = set(p.noise_tic[p.row.isin(used_rows)])
    counts = dict(n_pool=len(p))
    masks = [
        ("in_existing_set", p.row.isin(used_rows)),
        ("t2_excluded", p.row.isin(t2_excluded)),
        ("noise_exclude", p.exclude_from_training.astype(bool)),
    ]
    if not allow_shared_tic:
        masks.append(("noise_tic_shared", p.noise_tic.isin(used_tics)))
    drop = np.zeros(len(p), bool)
    for name, m in masks:
        m = m.to_numpy() & ~drop                     # counted in order: first filter that applies
        counts[f"dropped_{name}"] = int(m.sum())
        drop |= m
    out = p[~drop].drop(columns=["exclude_from_training"]).reset_index(drop=True)
    counts["n_eligible"] = len(out)
    counts["n_existing_noise_tics"] = len(used_tics)
    return out, counts


def check_only_run_checks(root: Path, cand: pd.DataFrame, n_workers: int) -> float:
    """run_checks replacement for --check: never touch the bundles; fail if a candidate is uncached."""
    checked = bds.load_checks(root, cand.bundle.unique())
    missing = set(cand.row) - set(checked.row.astype(int))
    if missing:
        raise SystemExit(f"--check: {len(missing)} candidate rows are not in the check cache "
                         f"({Path(root, sd.EXTRACT_DIR, BUILD_SUBDIR)}); run the build first")
    return 0.0


def build(root: Path, seed: int, n_workers: int, allow_shared_tic: bool, plan_only: bool = False,
          datasets: Path = DATASETS) -> dict | None:
    """Select, check, assign coverage; return the output frames (None with plan_only)."""
    configure_bds()
    full_pool, info = bds.load_pool(root)
    flags = pd.read_csv(Path(root, FLAGS), usecols=["row", "sim_id", "tic", "exclude_from_training"])
    t2_ex = set(pd.read_csv(datasets / "exclusions_log.csv", usecols=["row"]).row)
    pool, counts = filter_pool(full_pool, flags, load_existing(datasets), t2_ex, allow_shared_tic)
    pool = bds.rank_pool(pool, seed)
    print(f"pool: {json.dumps(counts)}", flush=True)
    chosen, checks, quotas, drive_s = bds.select(pool, root, seed, n_workers, plan_only=plan_only)
    if plan_only:
        q = pd.DataFrame(quotas).fillna(0).astype(int)
        q["pbin"] = q.index.str.split("|").str[0]
        caps = pool.groupby("pbin").size().rename("eligible")
        print(q.groupby("pbin").sum().join(caps).loc[list(bds.AUDIT_PBIN_LABELS)])
        return None
    lib = cov.load_library()
    df = bds.assign_coverage(chosen[NAME], lib, seed, COVERAGE_SET_INDEX)
    man = bds.finish_manifest(df, NAME, full_pool, lib)

    excl = checks[~checks.passed]
    excl = excl[["row", "sim_id", "bundle", "member", "stratum", "rank", "reason", "max_abs_dev",
                 "n_dev_gt_thresh", "orbits_hit", "native_sectors", "n_native_sectors"]].sort_values("row")
    full_n = full_pool.groupby("stratum").size()
    elig_n = pool.groupby("stratum").size()
    rows = []
    for k in sorted(full_n.index):
        pb, a, s = k.split("|")
        rows.append(dict(stratum=k, pbin=pb, amp_tier=a, shear_class=s, n_pool=int(full_n[k]),
                         n_eligible=int(elig_n.get(k, 0)), n_checked=int((checks.stratum == k).sum()),
                         n_excluded=int((excl.stratum == k).sum()), quota=quotas[NAME].get(k, 0),
                         n=int((man.stratum == k).sum())))
    strata = pd.DataFrame(rows)
    targets = bds.bin_targets(SIZE)
    got = man.groupby("pbin").size()
    summary = dict(
        seed=seed, size_target=SIZE, n_rows=len(man), pool_cut=bds.POOL_CUT, corrupt_thresh=bds.CORRUPT_THRESH,
        oversample=bds.OVERSAMPLE, long_fraction=bds.LONG_FRACTION, amp_tercile_cuts=info["amp_cuts"],
        allow_shared_noise_tic=allow_shared_tic, pool_filters=counts,
        bin_shortfall={pb: int(t - got.get(pb, 0)) for pb, t in targets.items() if got.get(pb, 0) < t},
        n_checked=len(checks), n_excluded=len(excl),
        exclusion_reasons=excl.reason.str.split(":").str[0].value_counts().to_dict(),
        coverage_fallbacks=int(df.coverage_fallback.sum()), n_noise_tics=int(pool.set_index("row")
                                                                               .noise_tic.reindex(man.row).nunique()),
        n_bundles_checked=int(checks.bundle.nunique()), data_root=bds.LOCAL_DATA_ROOT,
        library=str(cov.DEFAULT_LIBRARY.relative_to(ROOT)), drive_seconds_this_run=round(drive_s, 1))
    return dict(manifest=man, strata=strata, exclusions=excl, summary=summary)


def render(out: dict) -> dict:
    """{output key: file text}; the summary omits run-time fields so --check can compare it."""
    s = {k: v for k, v in out["summary"].items() if k != "drive_seconds_this_run"}
    return dict(manifest=out["manifest"].to_csv(index=False), strata=out["strata"].to_csv(index=False),
                exclusions=out["exclusions"].to_csv(index=False),
                summary=json.dumps(s, indent=2, default=int) + "\n")


def main(argv=None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="SMARTS data root (default $SMARTS_DATA_ROOT or the Data_Vault path)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-workers", type=int, default=1,
                    help="parallel bundle passes (default 1: the drive drops under parallel reads)")
    ap.add_argument("--out-dir", type=Path, default=DATASETS)
    ap.add_argument("--exclude-shared-noise-tic", action="store_true",
                    help="drop rows whose noise TIC also serves a dev_small/dev_medium/holdout star "
                         "(infeasible for the short-period bins; see the docstring)")
    ap.add_argument("--plan-only", action="store_true", help="pool filters and quotas only (reads the catalog)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory from the drive's check cache and compare with the committed files")
    args = ap.parse_args(argv)

    root = sd.require_drive(sd.data_root(args.data_root))
    t0 = time.time()
    if args.check:
        configure_bds()
        bds.run_checks = check_only_run_checks
    out = build(root, args.seed, args.n_workers, not args.exclude_shared_noise_tic, plan_only=args.plan_only)
    if out is None:
        return 0
    text = render(out)
    if args.check:
        stale = False
        for key, fname in OUT_FILES.items():
            p = args.out_dir / fname
            ok = p.exists() and p.read_text() == text[key]
            stale |= not ok
            print(f"{fname}: {'ok' if ok else 'STALE'}")
        return 1 if stale else 0
    sd.require_drive(root)
    n_copied = bds.install_files(out["manifest"], NAME, root)
    for key, fname in OUT_FILES.items():
        with sd.atomic_path(args.out_dir / fname) as tmp:
            tmp.write_text(text[key])
    print(f"{len(out['manifest'])} rows -> {args.out_dir / OUT_FILES['manifest']} "
          f"({n_copied} files copied to {bds.LOCAL_DIR}; {time.time() - t0:.0f} s)", flush=True)
    print(json.dumps(out["summary"], indent=2, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
