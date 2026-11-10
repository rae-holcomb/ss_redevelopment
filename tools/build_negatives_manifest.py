#!/usr/bin/env python3
"""Build the T27 non-rotator negatives manifest from T20's recovered noise stacks.

Every SMARTS file is a rotator injected into one of 1,719 TESS noise light curves. T20
recovered each noise light curve by a per-cadence median over its low-amplitude members
(``tools/audit/galaxy_stack.py``; ``extracted/galaxy_stack/fits/NNNN_all.fits`` is the
A+B stack). Quiet stacks are real-noise **non-rotators** with exactly the training data's
reduction, so they are the ranker's negatives.

Sources (Rae / T27 brief, 2026-11-10)
-------------------------------------
- classification ``quiet`` or ``systematic_common_mode`` with ``exclude_from_training``
  False (``misclassified_star`` and every contamination class are out);
- **train** negatives: sources whose files are *all* in the SIM_ID < 800k pool and that
  share no TIC with dev_medium or the holdout;
- **dev_medium** (evaluation) negatives: sources that appear in ``datasets/dev_medium.csv``;
- a source that appears only in the holdout is never used.

Sampling
--------
Per source, ``--n-draws`` seeded coverage draws plus one full-year row:

- draw k: within each set, coverage classes are dealt in shuffled blocks of all eight
  classes (the rotator manifests' scheme, ``build_dev_sets.assign_coverage``) and matched
  to sources that can support them (a library pattern contained in the stack's native
  whole sectors); the pattern within the class is drawn **frequency-weighted** by the
  library's ``n_stars`` among feasible patterns. RNG = ``[seed, set_index, draw]``.
- ``full``: empty ``sectors`` (every native sector, = ``--ignore-manifest-sectors``).

So the negatives' class mix matches the rotator manifests (train_v0 / dev_medium are
class-balanced), and the ranker can't learn "coverage class -> non-rotator".

Outputs
-------
    datasets/negatives_v0.csv         one row per (source, draw); paths relative to SMARTS_DATA_ROOT
    datasets/negatives_v0_sources.csv one row per source: native sectors, set, class
    (cache) <root>/extracted/negatives_v0/native_sectors.csv  per-stack check_file result

Usage
-----
    python tools/build_negatives_manifest.py                 # default seed, 3 draws
    python tools/build_negatives_manifest.py --n-draws 3 --seed 20261110
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))

import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import smarts_data as sd  # noqa: E402

FLAGS_REL = "extracted/galaxy_ids/noise_source_flags.csv.gz"
STACK_REL = "extracted/galaxy_stack"
CACHE_REL = "extracted/negatives_v0"
NEG_CLASSES = ("quiet", "systematic_common_mode")
TRAIN_POOL = "lt800k"
SETS = ("train", "dev_medium")
LABEL_SOURCE = "smarts_noise_stack"
DEFAULT_SEED = 20261110
OUT = ROOT / "datasets" / "negatives_v0.csv"
OUT_SOURCES = ROOT / "datasets" / "negatives_v0_sources.csv"


def stack_relpath(galaxy: int) -> str:
    """Path of a source's A+B stack FITS, relative to SMARTS_DATA_ROOT."""
    return f"{STACK_REL}/fits/{int(galaxy):04d}_all.fits"


def neg_star_id(tic: int, draw) -> str:
    """Unique star id of one negative row: ``neg-<tic>-d<k>`` or ``neg-<tic>-full``."""
    return f"neg-{int(tic)}-{'full' if draw == 'full' else f'd{int(draw)}'}"


def source_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Per-TIC classification, exclusion flag and the set of pools its files are in."""
    return flags.groupby("tic").agg(
        source_class=("classification", "first"),
        n_classes=("classification", "nunique"),
        exclude=("exclude_from_training", "first"),
        pools=("pool", lambda s: "|".join(sorted(set(s)))),
        galaxy=("galaxy", "first"),
        n_files=("row", "size"))


def manifest_tics(man: pd.DataFrame, flags: pd.DataFrame) -> set:
    """Noise TICs of a rotator manifest, joined on catalog ``row`` (every manifest has it)."""
    return set(man["row"].map(flags.set_index("row")["tic"]).dropna().astype("int64"))


def select_sources(flags: pd.DataFrame, dev_medium_tics: set, holdout_tics: set) -> pd.DataFrame:
    """Train and dev_medium negative sources under the strict pool rules (see module docstring).

    Returns one row per source with ``tic``, ``galaxy``, ``source_class``, ``pools`` and
    ``eval_set`` ('train' or 'dev_medium').
    """
    per = source_table(flags)
    if (per.n_classes > 1).any():
        raise ValueError("a noise TIC has more than one classification in the flags file")
    ok = per[per.source_class.isin(NEG_CLASSES) & ~per.exclude.astype(bool)].copy()
    train = ok[(ok.pools == TRAIN_POOL) & ~ok.index.isin(dev_medium_tics | holdout_tics)].copy()
    train["eval_set"] = "train"
    dev = ok[ok.index.isin(dev_medium_tics)].copy()
    dev["eval_set"] = "dev_medium"
    out = pd.concat([train, dev]).reset_index()
    if out.tic.duplicated().any():
        raise AssertionError("a source landed in both negative sets")
    return out[["tic", "galaxy", "source_class", "pools", "n_files", "eval_set"]]


def check_disjoint(src: pd.DataFrame, train_v0_tics: set, dev_medium_tics: set, holdout_tics: set) -> dict:
    """Leakage checks between negative sets and the rotator manifests; raises on any overlap.

    Wrong directions: train negatives vs dev_medium / holdout rotators, dev_medium
    negatives vs train_v0 rotators, and train vs dev_medium negatives.
    """
    tr = set(src.tic[src.eval_set == "train"])
    dv = set(src.tic[src.eval_set == "dev_medium"])
    res = dict(train_neg_vs_dev_neg=len(tr & dv), train_neg_vs_dev_medium_rot=len(tr & dev_medium_tics),
               train_neg_vs_holdout_rot=len(tr & holdout_tics), dev_neg_vs_train_v0_rot=len(dv & train_v0_tics),
               dev_neg_not_in_dev_medium=len(dv - dev_medium_tics),
               dev_neg_also_in_holdout=len(dv & holdout_tics),
               train_neg_also_in_train_v0=len(tr & train_v0_tics))
    bad = {k: v for k, v in res.items() if k not in ("dev_neg_also_in_holdout", "train_neg_also_in_train_v0") and v}
    if bad:
        raise AssertionError(f"pool leakage: {bad}")
    return res


def check_stack(path: Path) -> dict:
    """Corrupt-noise check plus native / native-whole sectors for one stack FITS.

    Same logic and threshold as ``build_dev_sets.check_file`` (whose flux scan needs a
    SMARTS file name, so the raw max |flux - 1| is computed here): ``native_sectors`` is
    what a full-year load keeps, ``native_whole_sectors`` the sectors with both orbits
    unflagged (what ``load_smarts_fits(sectors=[...])`` keeps for a requested sector).
    """
    from astropy.io import fits
    import build_dev_sets as bds
    from preprocessing import _orbit_idxs_for_sector, load_smarts_fits
    with fits.open(path) as h:
        raw = np.asarray(h["LIGHTCURVE"].data["flux"], float)
    rec = dict(max_abs_dev=float(np.nanmax(np.abs(raw - 1.0))), native_sectors="", n_native_sectors=0,
               native_whole_sectors="", n_whole_sectors=0, n_orbits_flagged=-1, passed=True, reason="")
    try:
        lc = load_smarts_fits(path)
    except Exception as exc:  # noqa: BLE001 -- recorded as an exclusion
        rec.update(passed=False, reason=f"load_error: {type(exc).__name__}: {exc}"[:200])
        return rec
    secs = sorted(int(x) for x in lc.sectors_used)
    flagged = lc.diagnostics.get("segment_flagged")
    n_orb = len(lc.segments)
    whole = [x for x in secs if _orbit_idxs_for_sector(x, n_orb) and
             (flagged is None or not any(flagged[i] for i in _orbit_idxs_for_sector(x, n_orb)))]
    rec.update(native_sectors=" ".join(map(str, secs)), n_native_sectors=len(secs),
               native_whole_sectors=" ".join(map(str, whole)), n_whole_sectors=len(whole),
               n_orbits_flagged=int(lc.n_orbits_flagged))
    if rec["max_abs_dev"] > bds.CORRUPT_THRESH:
        rec.update(passed=False, reason="corrupt_noise")
    elif not whole:
        rec.update(passed=False, reason="no_whole_sectors")
    return rec


def native_sectors(paths: list, cache: Path) -> pd.DataFrame:
    """``check_stack`` (corrupt check + native / whole sectors) per stack, cached.

    The cache is written atomically after the scan, so a drive unplug loses nothing.
    """
    import warnings
    warnings.simplefilter("ignore")
    rows = []
    if cache.exists():  # resume: keep what an earlier (possibly unplugged) run checked
        rows = pd.read_csv(cache, dtype={"native_sectors": str, "native_whole_sectors": str},
                           keep_default_na=False).to_dict("records")
    done = {r["path"] for r in rows}
    todo = [p for p in paths if p not in done]
    cache.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        with sd.atomic_path(cache) as tmp:
            pd.DataFrame(rows).to_csv(tmp, index=False)

    for i, p in enumerate(todo):
        if not Path(p).parent.exists():
            flush()
            raise SystemExit("Data_Vault drive unplugged mid-scan; plug it in and rerun (progress is cached)")
        rec = check_stack(Path(p))
        rec["path"] = p
        rows.append(rec)
        if (i + 1) % 100 == 0:
            flush()
            print(f"  native sectors {len(done) + i + 1}/{len(paths)}", flush=True)
    if todo:
        flush()
    df = pd.DataFrame(rows)
    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    return df[df.path.isin(set(paths))]


def feasible_weighted(lib: pd.DataFrame, native: set) -> dict:
    """{class: (patterns, n_stars weights)} of library patterns contained in ``native``."""
    out = {}
    for pat, cls, w in zip(lib.pattern, lib["class"], lib.n_stars):
        if set(cov.parse_pattern(pat)) <= native:
            p, ws = out.setdefault(cls, ([], []))
            p.append(pat)
            ws.append(float(w))
    return out


def assign_draw(src: pd.DataFrame, lib: pd.DataFrame, seed: int, set_index: int, draw: int) -> pd.DataFrame:
    """One coverage draw for every source of one set: class-balanced, frequency-weighted pattern.

    Classes are dealt in shuffled blocks of the eight coverage classes and matched to
    sources that support them (``scipy.optimize.linear_sum_assignment``); a source that
    can't take its slot falls back to a random feasible class (``coverage_fallback``).
    """
    from scipy.optimize import linear_sum_assignment
    classes = list(cov.COVERAGE_CLASSES)
    rng = np.random.default_rng([seed, set_index, draw])
    n = len(src)
    slots = np.concatenate([rng.permutation(classes) for _ in range(math.ceil(n / len(classes)))])[:n]
    feas = [feasible_weighted(lib, set(map(int, str(s).split()))) for s in src.native_whole_sectors]
    cost = np.array([[0.0 if c in f else 1e6 for c in slots] for f in feas]) + rng.random((n, n)) * 1e-3
    ri, ci = linear_sum_assignment(cost)
    out = []
    for r, c in zip(ri, ci):
        f = feas[r]
        cls, fallback = slots[c], False
        if cls not in f:
            cls, fallback = rng.choice(sorted(f)), True
        pats, w = f[cls]
        w = np.asarray(w)
        pat = pats[int(rng.choice(len(pats), p=w / w.sum()))]
        out.append(dict(tic=src.tic.iloc[r], draw_id=str(draw), sectors=pat, coverage_class=cls,
                        n_feasible_patterns=len(pats), coverage_fallback=fallback))
    return pd.DataFrame(out)


def build_manifest(src: pd.DataFrame, lib: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    """All negative rows: ``n_draws`` coverage draws + one full-year row per source."""
    parts = []
    for si, name in enumerate(SETS):
        s = src[src.eval_set == name].reset_index(drop=True)
        for k in range(n_draws):
            parts.append(assign_draw(s, lib, seed, si, k))
        parts.append(pd.DataFrame(dict(tic=s.tic, draw_id="full", sectors="", coverage_class="fullyear",
                                       n_feasible_patterns=0, coverage_fallback=False)))
    d = pd.concat(parts, ignore_index=True).merge(src, on="tic", how="left", validate="m:1")
    d["star_id"] = [neg_star_id(t, k) for t, k in zip(d.tic, d.draw_id)]
    d["path"] = d.galaxy.map(stack_relpath)
    d["noise_tic"] = d.tic.astype("int64")
    d["pool"] = d.pools
    d["n_sectors"] = [len(cov.parse_pattern(s)) if s else len(str(n).split())
                      for s, n in zip(d.sectors, d.native_sectors)]
    d["seed"] = seed
    d["is_rotator"] = False
    d["label_source"] = LABEL_SOURCE
    cols = ["star_id", "path", "noise_tic", "galaxy", "source_class", "pool", "eval_set", "draw_id",
            "sectors", "n_sectors", "coverage_class", "n_feasible_patterns", "coverage_fallback", "seed",
            "native_sectors", "native_whole_sectors", "is_rotator", "label_source"]
    d = d[cols].sort_values(["eval_set", "noise_tic", "draw_id"]).reset_index(drop=True)
    if d.star_id.duplicated().any():
        raise AssertionError("duplicate star_id in the negatives manifest")
    return d


def main(argv=None):
    """CLI entry point: select sources, check pools, read native sectors, draw coverage, write."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None, help="SMARTS_DATA_ROOT (default env var / Data_Vault)")
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--out-sources", type=Path, default=OUT_SOURCES)
    args = ap.parse_args(argv)
    root = sd.require_drive(sd.data_root(args.data_root))
    flags = pd.read_csv(root / FLAGS_REL, usecols=["row", "tic", "galaxy", "pool", "classification",
                                                    "exclude_from_training"])
    man = {n: pd.read_csv(ROOT / "datasets" / f"{n}.csv", usecols=["row"]) for n in
           ("dev_medium", "train_v0", "holdout")}
    tics = {n: manifest_tics(m, flags) for n, m in man.items()}
    src = select_sources(flags, tics["dev_medium"], tics["holdout"])
    leak = check_disjoint(src, tics["train_v0"], tics["dev_medium"], tics["holdout"])
    print("leakage checks:", leak)
    paths = [str(root / stack_relpath(g)) for g in src.galaxy]
    ns = native_sectors(paths, root / CACHE_REL / "native_sectors.csv").set_index("path")
    src["native_sectors"] = [ns.at[p, "native_sectors"] for p in paths]
    src["native_whole_sectors"] = [ns.at[p, "native_whole_sectors"] for p in paths]
    src["stack_check_passed"] = [bool(ns.at[p, "passed"]) for p in paths]
    bad = src[~src.stack_check_passed]
    if len(bad):
        print(f"dropping {len(bad)} sources whose stack fails check_file:", bad.tic.tolist())
    src = src[src.stack_check_passed].reset_index(drop=True)
    lib = cov.load_library()
    d = build_manifest(src, lib, args.n_draws, args.seed)
    with sd.atomic_path(args.out) as tmp:
        d.to_csv(tmp, index=False)
    with sd.atomic_path(args.out_sources) as tmp:
        src.assign(path=src.galaxy.map(stack_relpath)).to_csv(tmp, index=False)
    print(d.groupby(["eval_set", "coverage_class"]).size().unstack(0))
    print(f"wrote {args.out} ({len(d)} rows, {d.noise_tic.nunique()} sources)")


if __name__ == "__main__":
    main()
