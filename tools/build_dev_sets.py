#!/usr/bin/env python3
"""Build the frozen SMARTS dev sets and holdout (task T2): dev_small, dev_medium, holdout.

Spec: docs/TASKS.md, T2 "Rae's design decisions (2026-10-15)". Every set is drawn
from the SIM_ID >= 800,000 galaxy-noise pool (training uses SIM_ID < 800k), and
rows are joined to files by catalog **row index** (file smarts-tess-v1.0-RRRRRR =
catalog row RRRRRR; see reports/2026-10-15_t1-smarts-catalog-audit/).

Procedure
---------
1. **Strata.** Each pool row gets a stratum = audit period bin (7 bins, <1 to
   120-180 d) x amplitude-proxy tercile (cuts from the whole corpus) x shear class
   (anti / none / solar). Every stratum's pool rows are put in one fixed random
   order (seeded), the *rank*.
2. **Quotas.** Per set: the five bins below 60 d get equal shares; 60-120 and
   120-180 d together get ~1/7 (the "should decline" stars). Within a bin, the
   nine amplitude x shear cells get equal shares, water-filled under each cell's
   capacity (some <1 d cells are smaller than an equal share). Sets are allocated
   in the order dev_small, dev_medium, holdout, each from what is left.
3. **Candidates and checks.** For each stratum, the first ceil(1.3 x quota) rows in
   rank order are candidates. They are extracted in one streaming pass per needed
   bundle (only the needed bundles) to a staging folder on the drive, and checked:
   corrupt galaxy noise (``flux_outliers.scan``, raw max |flux - 1| > 0.1) and a
   ``load_smarts_fits`` call that records the kept sectors (``native_sectors``).
   Failing rows go to ``datasets/exclusions_log.csv``. Passing rows, in rank order,
   fill dev_small first, then dev_medium, then holdout. A stratum that runs short
   gets more candidates (next ranks) in another round; a stratum whose whole pool
   is checked gets its capacity lowered to its passing count and the quotas are
   re-allocated. The result depends only on the seed and the data.
4. **Files.** dev_small and dev_medium files are copied to the laptop
   (``data/smarts_dev/``, the local working set); holdout files are copied to
   ``<root>/extracted/holdout/`` on the drive. Manifest ``path`` values are
   relative to the set's data root: ``data/smarts_dev`` for the dev sets and the
   SMARTS data root (``$SMARTS_DATA_ROOT``) for the holdout.
5. **Coverage.** Each row gets a sector pattern from the pooled all-cycles library
   (``datasets/coverage/all_cycles_patterns.csv``), drawn only from patterns
   that are subsets of its ``native_sectors``. Coverage classes are balanced
   within every period bin of every set (classes dealt in shuffled blocks of 8,
   then a feasibility-respecting assignment), and the pattern is uniform within the
   class among the feasible ones (T16a's default weighting).
6. **Weights.** ``sampling_weight`` = (stratum share of the SIM_ID >= 800k corpus) /
   (stratum share of the set), so weighted means estimate corpus-level rates.
   ``coverage_weight`` = (class share of real star-seasons in the library) /
   (class share of the set), for reweighting to the real-world coverage mix.
   ``combined_weight`` is their product.

Outputs: ``datasets/{dev_small,dev_medium,holdout}.csv``,
``datasets/exclusions_log.csv``, ``datasets/strata_counts.csv``,
``datasets/build_summary.json``; staging and check cache on the drive under
``<root>/extracted/t2_build/``.

Resumable and safe to interrupt: per-bundle check results are cached on the drive
(``t2_build/checks/NNN.csv``, atomic writes), extracted files are skipped when
already present, and every output is written atomically. If the drive vanishes,
the script stops with a message; rerun the same command to resume.

Usage::

    python tools/build_dev_sets.py                 # full build (~20-25 min of drive time)
    python tools/build_dev_sets.py --plan-only     # quotas and candidate counts, no extraction
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import smarts_data as sd  # noqa: E402

DEFAULT_SEED = 20261015
POOL_CUT = 800_000                     # SIM_ID >= POOL_CUT: val/test galaxy-noise pool
CORRUPT_THRESH = 0.1                   # raw max |flux - 1| above this = corrupt galaxy noise
OVERSAMPLE = 1.3                       # candidates per stratum = ceil(OVERSAMPLE * quota)
SET_SIZES = {"dev_small": 50, "dev_medium": 1000, "holdout": 3000}
SET_ORDER = ("dev_small", "dev_medium", "holdout")
LOCAL_SETS = ("dev_small", "dev_medium")
LONG_FRACTION = 1 / 7                  # share of each set in 60-120 + 120-180 d
LOCAL_DIR = ROOT / "data" / "smarts_dev"
LOCAL_DATA_ROOT = "data/smarts_dev"    # data root of the dev manifests (repo-relative)
HOLDOUT_SUBDIR = "extracted/holdout"   # holdout files, relative to the SMARTS data root
BUILD_SUBDIR = "t2_build"              # under <root>/extracted/: staging + check cache

# Same bins/strata as the T1 audit (tools/audit/catalog_audit.py).
from catalog_audit import AUDIT_PBIN_LABELS, amplitude_proxy, audit_pbin, shear_class  # noqa: E402

SHORT_BINS = AUDIT_PBIN_LABELS[:5]
LONG_BINS = AUDIT_PBIN_LABELS[5:]
AMP_TIERS = ("low", "mid", "high")
SHEAR_CLASSES = ("anti", "none", "solar")
CELLS = [(a, s) for a in AMP_TIERS for s in SHEAR_CLASSES]

CHECK_FIELDS = ["row", "member", "passed", "reason", "max_abs_dev", "n_dev_gt_thresh",
                "orbits_hit", "native_sectors", "n_native_sectors", "n_orbits_flagged"]

MANIFEST_COLUMNS = [
    # identifiers
    "path", "star_id", "row", "sim_id", "bundle", "member",
    # truth
    "true_period", "shear", "minlat", "maxlat", "activity", "incl",
    "cycle", "overlap", "tspot", "bfly",
    # differential rotation (evaluation.active_band_period_range)
    "P_lo", "P_hi", "band_disjoint", "band_extends",
    # native data and assigned coverage
    "native_sectors", "n_native_sectors", "native_class", "max_abs_dev",
    "sectors", "n_sectors", "coverage_class", "n_feasible_patterns",
    # strata and weights
    "stratum", "pbin", "eval_pbin", "amp_tier", "amp_proxy", "shear_class",
    "sampling_weight", "coverage_weight", "combined_weight",
]


# --------------------------------------------------------------------------- catalog

def load_pool(root: Path) -> tuple:
    """Load the catalog and label strata.

    Returns
    -------
    (pool, corpus_info): ``pool`` is a DataFrame of the SIM_ID >= POOL_CUT rows
    with columns row, sim_id, the SMARTS parameters, pbin, amp_tier, amp_proxy,
    shear_class, stratum and bundle; ``corpus_info`` holds the amplitude tercile
    cuts (computed over the whole 1M corpus) and the corpus size.
    """
    from catalog_to_table import load_catalog
    df = load_catalog(root)
    df["row"] = np.arange(len(df))
    prox = amplitude_proxy(df)
    cuts = np.quantile(prox, [1 / 3, 2 / 3])
    df["amp_proxy"] = prox
    df["amp_tier"] = np.where(prox < cuts[0], "low", np.where(prox < cuts[1], "mid", "high"))
    df["pbin"] = audit_pbin(df.PERIOD.values)
    df["shear_class"] = shear_class(df.DIFFROT.values)
    pool = df[df.SIM_ID >= POOL_CUT].copy()
    pool["stratum"] = pool.pbin + "|" + pool.amp_tier + "|" + pool.shear_class
    pool = pool.rename(columns={"SIM_ID": "sim_id"})
    idx = pd.read_csv(Path(root, sd.EXTRACT_DIR, "bundle_index", "all_bundles.csv.gz"))
    pool = pool.merge(idx[["row", "bundle", "member"]], on="row", how="left", validate="1:1")
    if pool.bundle.isna().any():
        raise SystemExit(f"{int(pool.bundle.isna().sum())} pool rows missing from the bundle index")
    pool["bundle"] = pool.bundle.astype(int)
    return pool.reset_index(drop=True), dict(amp_cuts=[float(c) for c in cuts], n_corpus=len(df))


def rank_pool(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add a per-stratum random rank (0 = first candidate), reproducible from ``seed``.

    Each stratum uses its own generator seeded by (seed, stratum index), so a
    stratum's order doesn't depend on the others.
    """
    pool = pool.sort_values("row").reset_index(drop=True)
    pool["rank"] = -1
    for i, (name, g) in enumerate(sorted(pool.groupby("stratum").groups.items())):
        rng = np.random.default_rng([seed, i])
        perm = rng.permutation(len(g))
        pool.loc[np.asarray(g)[perm], "rank"] = np.arange(len(g))
    return pool


# --------------------------------------------------------------------------- quotas

def split_even(n: int, k: int, order=None) -> np.ndarray:
    """Split ``n`` into ``k`` near-equal integers; the remainder goes to ``order``'s first slots."""
    out = np.full(k, n // k)
    order = np.arange(k) if order is None else np.asarray(order)
    out[order[: n - out.sum()]] += 1
    return out


def waterfill(n: int, caps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Allocate ``n`` as equally as possible across cells, never exceeding ``caps``.

    Cells that can't take an equal share are filled to capacity and the rest is
    spread over the others. The leftover (n not divisible) goes to randomly chosen
    cells with spare capacity.
    """
    caps = np.asarray(caps, int)
    if n > caps.sum():
        raise ValueError(f"need {n} but only {caps.sum()} available")
    alloc = np.zeros(len(caps), int)
    while alloc.sum() < n:
        open_ = np.flatnonzero(alloc < caps)
        share = (n - alloc.sum()) // len(open_)
        if share == 0:
            extra = rng.permutation(open_)[: n - alloc.sum()]
            alloc[extra] += 1
            break
        alloc[open_] = np.minimum(caps[open_], alloc[open_] + share)
    return alloc


def bin_targets(n: int) -> dict:
    """Stars per audit period bin for a set of size ``n`` (equal short bins; ~1/7 long)."""
    n_long = int(round(n * LONG_FRACTION))
    short = split_even(n - n_long, len(SHORT_BINS))  # remainder to the shortest bins
    long_ = split_even(n_long, len(LONG_BINS))
    return dict(zip(SHORT_BINS + LONG_BINS, list(short) + list(long_)))


def allocate(caps: dict, seed: int) -> dict:
    """Per-set, per-stratum quotas.

    Parameters
    ----------
    caps : {stratum: capacity} (pool size, or passing count once a stratum is
        fully checked).
    seed : base seed for the remainder choices.

    Returns
    -------
    {set_name: {stratum: quota}}
    """
    left = dict(caps)
    out = {}
    for si, name in enumerate(SET_ORDER):
        q = {}
        for bi, (pb, n_bin) in enumerate(bin_targets(SET_SIZES[name]).items()):
            keys = [f"{pb}|{a}|{s}" for a, s in CELLS]
            rng = np.random.default_rng([seed, 1000 + si, bi])
            alloc = waterfill(n_bin, np.array([left.get(k, 0) for k in keys]), rng)
            for k, a in zip(keys, alloc):
                q[k] = int(a)
                left[k] = left.get(k, 0) - int(a)
        out[name] = q
    return out


# --------------------------------------------------------------------------- extraction + checks

def check_path(root: Path, n: int) -> Path:
    """Cached check results of bundle ``n`` (drive)."""
    return Path(root, sd.EXTRACT_DIR, BUILD_SUBDIR, "checks", f"{n:03d}.csv")


def staging_dir(root: Path) -> Path:
    """Where candidate FITS are extracted on the drive."""
    return Path(root, sd.EXTRACT_DIR, BUILD_SUBDIR, "candidates")


def check_file(path: Path) -> dict:
    """Corrupt-noise check plus native sectors for one extracted FITS file.

    Returns a dict with the CHECK_FIELDS (except row/member). ``passed`` is False
    with ``reason`` 'corrupt_noise' (raw max |flux - 1| > CORRUPT_THRESH),
    'load_error' (load_smarts_fits raised) or 'no_native_sectors'.
    """
    from flux_outliers import scan
    from preprocessing import load_smarts_fits
    s = scan(path, CORRUPT_THRESH)
    rec = dict(max_abs_dev=s["max_abs_dev"], n_dev_gt_thresh=s["n_dev_gt_thresh"],
               orbits_hit=s["orbits_hit"], native_sectors="", n_native_sectors=0,
               n_orbits_flagged=-1, passed=True, reason="")
    try:
        lc = load_smarts_fits(path)
        secs = sorted(int(x) for x in lc.sectors_used)
        rec.update(native_sectors=" ".join(map(str, secs)), n_native_sectors=len(secs),
                   n_orbits_flagged=int(lc.n_orbits_flagged))
    except Exception as exc:  # noqa: BLE001 -- recorded as an exclusion
        rec.update(passed=False, reason=f"load_error: {type(exc).__name__}: {exc}"[:200])
        return rec
    if s["max_abs_dev"] > CORRUPT_THRESH:
        rec.update(passed=False, reason="corrupt_noise")
    elif not secs:
        rec.update(passed=False, reason="no_native_sectors")
    return rec


def process_bundle(n: int, rows: list, root: str) -> tuple:
    """One streaming pass over bundle ``n``: extract ``rows`` to staging and check them.

    Module-level (picklable) per-bundle worker. Merges with any cached check
    results for the bundle and rewrites the cache atomically.

    Returns
    -------
    (n, n_new_checked, seconds)
    """
    import bundle_index
    warnings_off()
    root = Path(root)
    t0 = time.time()
    cp = check_path(root, n)
    cached = pd.read_csv(cp, dtype={"native_sectors": str, "orbits_hit": str}).fillna(
        {"native_sectors": "", "orbits_hit": "", "reason": ""}) if cp.exists() else pd.DataFrame(columns=CHECK_FIELDS)
    todo = sorted(set(int(r) for r in rows) - set(cached.row.astype(int)))
    if not todo:
        return n, 0, 0.0
    stage = staging_dir(root)
    stage.mkdir(parents=True, exist_ok=True)
    have = {r for r in todo if (stage / f"smarts-tess-v1.0-{r:06d}.fits").exists()}
    if set(todo) - have:
        idx = bundle_index.scan_bundle(root, n, extract_rows=set(todo) - have, extract_to=stage)
    else:
        idx = pd.read_csv(bundle_index.index_path(root, n))
    member = dict(zip(idx.row.astype(int), idx.member))
    recs = []
    for r in todo:
        p = stage / f"smarts-tess-v1.0-{r:06d}.fits"
        if not p.exists():
            recs.append(dict(row=r, member=member.get(r, ""), passed=False, reason="missing_from_bundle"))
            continue
        recs.append(dict(row=r, member=member[r], **check_file(p)))
    out = pd.concat([cached, pd.DataFrame(recs)], ignore_index=True)[CHECK_FIELDS].sort_values("row")
    with sd.atomic_path(cp) as tmp:
        out.to_csv(tmp, index=False)
    return n, len(todo), time.time() - t0


def warnings_off():
    """Silence library warnings inside worker processes."""
    import warnings
    warnings.simplefilter("ignore")


def load_checks(root: Path, bundles) -> pd.DataFrame:
    """Concatenate cached check results for ``bundles``."""
    parts = [pd.read_csv(check_path(root, n), dtype={"native_sectors": str, "orbits_hit": str})
             for n in sorted(set(bundles)) if check_path(root, n).exists()]
    if not parts:
        return pd.DataFrame(columns=CHECK_FIELDS)
    out = pd.concat(parts, ignore_index=True)
    out[["native_sectors", "orbits_hit", "reason"]] = out[["native_sectors", "orbits_hit", "reason"]].fillna("")
    out["passed"] = out.passed.astype(str).str.lower().eq("true")
    return out


def run_checks(root: Path, cand: pd.DataFrame, n_workers: int) -> float:
    """Extract and check every candidate row not already in the check cache.

    Groups candidates by bundle and runs one streaming pass per bundle that still
    has unchecked rows. Returns the drive seconds spent (sum over bundles).
    """
    checked = load_checks(root, cand.bundle.unique())
    done = set(checked.row.astype(int))
    need = cand[~cand.row.isin(done)]
    jobs = {int(n): g.row.astype(int).tolist() for n, g in need.groupby("bundle")}
    if not jobs:
        return 0.0
    print(f"  checking {len(need)} new candidates in {len(jobs)} bundles ({n_workers} worker(s))", flush=True)
    worker = partial(process_bundle, root=str(root))
    t0 = time.time()
    total = 0.0
    if n_workers <= 1:
        results = (worker(n, rows) for n, rows in jobs.items())
        for i, (n, k, s) in enumerate(results):
            sd.require_drive(root)
            total += s
            print(f"    bundle {n:03d}: {k} checked ({s:.1f} s) [{i + 1}/{len(jobs)}, {time.time() - t0:.0f} s]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=determinism.init_worker) as ex:
            futs = {ex.submit(worker, n, rows): n for n, rows in jobs.items()}
            for i, fut in enumerate(as_completed(futs)):
                try:
                    n, k, s = fut.result()
                except Exception as exc:  # noqa: BLE001
                    sd.require_drive(root)  # exits cleanly if the drive vanished
                    raise RuntimeError(f"bundle {futs[fut]:03d} failed: {exc}") from exc
                total += s
                print(f"    bundle {n:03d}: {k} checked ({s:.1f} s) [{i + 1}/{len(jobs)}, {time.time() - t0:.0f} s]", flush=True)
    return time.time() - t0


# --------------------------------------------------------------------------- selection

def select(pool: pd.DataFrame, root: Path, seed: int, n_workers: int, plan_only: bool = False,
           max_rounds: int = 10) -> tuple:
    """Iterate candidates -> checks -> quotas until every stratum quota is filled.

    Returns
    -------
    (chosen, checks, quotas, drive_seconds): ``chosen`` maps set name to a
    DataFrame of pool rows (with check columns); ``checks`` holds all checked
    candidates; ``quotas`` the final per-set per-stratum quotas.
    """
    caps = pool.groupby("stratum").size().to_dict()
    depth = {}  # stratum -> number of ranks taken as candidates
    drive_s = 0.0
    for rnd in range(max_rounds):
        quotas = allocate(caps, seed)
        total_q = {k: sum(q[k] for q in quotas.values()) for k in caps}
        for k, tq in total_q.items():
            depth[k] = min(caps_pool(pool, k), max(depth.get(k, 0), math.ceil(OVERSAMPLE * tq)))
        cand = pool[pool["rank"] < pool.stratum.map(depth)]
        print(f"round {rnd}: {len(cand)} candidates for {sum(total_q.values())} slots "
              f"in {cand.bundle.nunique()} bundles", flush=True)
        if plan_only:
            return None, None, quotas, 0.0
        drive_s += run_checks(root, cand, n_workers)
        checks = load_checks(root, cand.bundle.unique())
        checks = checks[checks.row.isin(cand.row)]
        passed = cand.merge(checks[checks.passed], on=["row", "member"], how="inner")
        n_pass = passed.groupby("stratum").size()
        short = {k: tq for k, tq in total_q.items() if n_pass.get(k, 0) < tq}
        if not short:
            break
        changed = False
        for k, tq in short.items():
            if depth[k] >= caps_pool(pool, k):          # whole stratum checked: lower capacity
                caps[k] = int(n_pass.get(k, 0))
            else:                                       # take more ranks next round
                deficit = tq - n_pass.get(k, 0)
                depth[k] = min(caps_pool(pool, k), depth[k] + math.ceil(1.5 * deficit) + 2)
            changed = True
        print(f"  {len(short)} strata short; re-planning", flush=True)
        if not changed:
            break
    else:
        raise SystemExit("quotas not filled after max_rounds")

    # Fill sets in rank order: dev_small first, then dev_medium, then holdout.
    passed = passed.sort_values(["stratum", "rank"])
    chosen = {name: [] for name in SET_ORDER}
    for k, g in passed.groupby("stratum"):
        start = 0
        for name in SET_ORDER:
            q = quotas[name].get(k, 0)
            chosen[name].append(g.iloc[start:start + q])
            start += q
    chosen = {name: pd.concat(v, ignore_index=True) for name, v in chosen.items()}
    return chosen, checks.merge(cand[["row", "stratum", "rank", "sim_id", "bundle"]], on="row"), quotas, drive_s


def caps_pool(pool: pd.DataFrame, k: str) -> int:
    """Number of pool rows in stratum ``k``."""
    return int((pool.stratum == k).sum())


# --------------------------------------------------------------------------- coverage

def feasible_patterns(lib: pd.DataFrame, native: set) -> dict:
    """{class: [pattern strings]} of library patterns contained in ``native``."""
    out = {}
    for pat, cls in zip(lib.pattern, lib["class"]):
        if set(cov.parse_pattern(pat)) <= native:
            out.setdefault(cls, []).append(pat)
    return out


def assign_coverage(df: pd.DataFrame, lib: pd.DataFrame, seed: int, set_index: int) -> pd.DataFrame:
    """Assign each row a coverage class and pattern, balanced per period bin.

    Within each period bin, classes are dealt in shuffled blocks of all eight
    classes (so counts differ by at most one), then matched to rows with
    ``scipy.optimize.linear_sum_assignment`` so that each row only gets a class it
    can support (some pattern of the class is a subset of its native sectors).
    Ties are broken randomly. A row that can't get any slot it supports (only
    when a bin lacks enough 13-sector files for 'full') falls back to a random
    feasible class and is counted in the build summary. The pattern is then
    drawn uniformly among the class's feasible patterns.
    """
    from scipy.optimize import linear_sum_assignment
    classes = list(cov.COVERAGE_CLASSES)
    df = df.copy()
    df["sectors"], df["coverage_class"], df["n_feasible_patterns"] = "", "", 0
    df["coverage_fallback"] = False
    for bi, pb in enumerate(AUDIT_PBIN_LABELS):
        sub = df.index[df.pbin == pb]
        if len(sub) == 0:
            continue
        rng = np.random.default_rng([seed, 2000 + set_index, bi])
        n_blocks = math.ceil(len(sub) / len(classes))
        slots = np.concatenate([rng.permutation(classes) for _ in range(n_blocks)])[: len(sub)]
        feas = [feasible_patterns(lib, set(map(int, df.at[i, "native_sectors"].split()))) for i in sub]
        cost = np.array([[0.0 if c in f else 1e6 for c in slots] for f in feas])
        cost += rng.random(cost.shape) * 1e-3
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            i, f = sub[r], feas[r]
            cls = slots[c]
            if cls not in f:
                cls = rng.choice(sorted(f))
                df.at[i, "coverage_fallback"] = True
            pats = sorted(f[cls])
            df.at[i, "coverage_class"] = cls
            df.at[i, "sectors"] = pats[int(rng.integers(len(pats)))]
            df.at[i, "n_feasible_patterns"] = len(pats)
    df["n_sectors"] = df.sectors.map(lambda s: len(cov.parse_pattern(s)))
    return df


# --------------------------------------------------------------------------- manifest

def finish_manifest(df: pd.DataFrame, name: str, pool: pd.DataFrame, lib: pd.DataFrame) -> pd.DataFrame:
    """Add truth, diffrot, weight and path columns; return in MANIFEST_COLUMNS order."""
    df = df.copy()
    df["star_id"] = df.row.map(lambda r: f"smarts-tess-v1.0-{int(r):06d}")
    fname = df.star_id + ".fits"
    df["path"] = fname if name in LOCAL_SETS else HOLDOUT_SUBDIR + "/" + fname
    df = df.rename(columns={"PERIOD": "true_period", "DIFFROT": "shear", "MINLAT": "minlat",
                            "MAXLAT": "maxlat", "ACTIVITY": "activity", "INCL": "incl", "CYCLE": "cycle",
                            "OVERLAP": "overlap", "TSPOT": "tspot", "BFLY": "bfly"})
    band = np.array([ev.active_band_period_range(p, s, lo, hi) for p, s, lo, hi in
                     zip(df.true_period, df.shear, df.minlat, df.maxlat)])
    df["P_lo"], df["P_hi"] = band[:, 0], band[:, 1]
    lo_r, hi_r = df.P_lo / df.true_period, df.P_hi / df.true_period
    df["band_disjoint"] = (lo_r > 1 + ev.REL_TOL) | (hi_r < 1 - ev.REL_TOL)
    df["band_extends"] = (hi_r > 1 + ev.REL_TOL) | (lo_r < 1 - ev.REL_TOL)
    df["eval_pbin"] = df.true_period.map(ev.period_bin)
    df["native_class"] = df.native_sectors.map(lambda s: cov.coverage_class(cov.parse_pattern(s)))
    corpus_frac = pool.groupby("stratum").size() / len(pool)
    set_frac = df.groupby("stratum").size() / len(df)
    df["sampling_weight"] = df.stratum.map(corpus_frac / set_frac)
    real = lib.groupby("class").n_stars.sum() / lib.n_stars.sum()
    cls_frac = df.groupby("coverage_class").size() / len(df)
    df["coverage_weight"] = df.coverage_class.map(real / cls_frac)
    df["combined_weight"] = df.sampling_weight * df.coverage_weight
    df["n_native_sectors"] = df.n_native_sectors.astype(int)
    return df.sort_values("row")[MANIFEST_COLUMNS].reset_index(drop=True)


def install_files(df: pd.DataFrame, name: str, root: Path) -> int:
    """Copy a set's FITS from the drive staging folder to its home; return files copied.

    Dev sets go to LOCAL_DIR (laptop), the holdout to <root>/extracted/holdout.
    Each copy is atomic and skipped if the destination already has the same size.
    """
    dest = LOCAL_DIR if name in LOCAL_SETS else Path(root, HOLDOUT_SUBDIR)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in df.star_id:
        src = staging_dir(root) / f"{sid}.fits"
        dst = dest / f"{sid}.fits"
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        with sd.atomic_path(dst) as tmp:
            shutil.copyfile(src, tmp)
        n += 1
    return n


def write_csv(df: pd.DataFrame, path: Path, **kw):
    """Atomic CSV write."""
    with sd.atomic_path(path) as tmp:
        df.to_csv(tmp, index=False, **kw)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="SMARTS data root (default $SMARTS_DATA_ROOT or the Data_Vault path)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-workers", type=int, default=2, help="parallel bundle passes (drive-bound; 2 is plenty)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    ap.add_argument("--plan-only", action="store_true", help="print quotas and candidate counts, then stop")
    args = ap.parse_args(argv)

    root = sd.require_drive(sd.data_root(args.data_root))
    t_start = time.time()
    pool, info = load_pool(root)
    pool = rank_pool(pool, args.seed)
    print(f"pool: {len(pool)} rows with SIM_ID >= {POOL_CUT} (of {info['n_corpus']}); "
          f"amp tercile cuts {info['amp_cuts']}", flush=True)
    chosen, checks, quotas, drive_s = select(pool, root, args.seed, args.n_workers, plan_only=args.plan_only)
    if args.plan_only:
        q = pd.DataFrame(quotas).fillna(0).astype(int)
        q["pbin"] = q.index.str.split("|").str[0]
        print(q.groupby("pbin").sum().loc[list(AUDIT_PBIN_LABELS)])
        return

    lib = cov.load_library()
    out = {}
    for si, name in enumerate(SET_ORDER):
        df = assign_coverage(chosen[name], lib, args.seed, si)
        fallback = int(df.coverage_fallback.sum())
        out[name] = (finish_manifest(df, name, pool, lib), fallback)

    # disjointness guard before writing anything
    rows = [set(m.row) for m, _ in out.values()]
    assert not (rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2]), "sets overlap"

    copied = {}
    for name, (m, _) in out.items():
        sd.require_drive(root)
        copied[name] = install_files(m, name, root)
        write_csv(m, args.out_dir / f"{name}.csv")
        print(f"{name}: {len(m)} rows -> {args.out_dir / (name + '.csv')} ({copied[name]} files copied)", flush=True)

    used = set().union(*rows)
    excl = checks[~checks.passed].copy()
    excl = excl[["row", "sim_id", "bundle", "member", "stratum", "rank", "reason", "max_abs_dev",
                 "n_dev_gt_thresh", "orbits_hit", "native_sectors", "n_native_sectors"]].sort_values("row")
    write_csv(excl, args.out_dir / "exclusions_log.csv")

    counts = []
    pool_n = pool.groupby("stratum").size()
    for k in sorted(pool_n.index):
        pb, a, s = k.split("|")
        rec = dict(stratum=k, pbin=pb, amp_tier=a, shear_class=s, n_pool=int(pool_n[k]),
                   n_checked=int((checks.stratum == k).sum()), n_excluded=int((excl.stratum == k).sum()))
        for name in SET_ORDER:
            rec[f"quota_{name}"] = quotas[name].get(k, 0)
            rec[f"n_{name}"] = int((out[name][0].stratum == k).sum())
        counts.append(rec)
    write_csv(pd.DataFrame(counts), args.out_dir / "strata_counts.csv")

    summary = dict(seed=args.seed, pool_cut=POOL_CUT, corrupt_thresh=CORRUPT_THRESH, oversample=OVERSAMPLE,
                   set_sizes=SET_SIZES, long_fraction=LONG_FRACTION, amp_tercile_cuts=info["amp_cuts"],
                   n_pool=len(pool), n_checked=len(checks), n_excluded=len(excl),
                   exclusion_reasons=excl.reason.str.split(":").str[0].value_counts().to_dict(),
                   n_used=len(used), coverage_fallbacks={k: v for k, (_, v) in out.items()},
                   n_bundles_checked=int(checks.bundle.nunique()),
                   drive_seconds_this_run=round(drive_s, 1), wall_seconds_this_run=round(time.time() - t_start, 1),
                   data_roots={"dev_small": LOCAL_DATA_ROOT, "dev_medium": LOCAL_DATA_ROOT,
                               "holdout": "$SMARTS_DATA_ROOT"},
                   library=str(cov.DEFAULT_LIBRARY.relative_to(ROOT)))
    with sd.atomic_path(args.out_dir / "build_summary.json") as tmp:
        tmp.write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
