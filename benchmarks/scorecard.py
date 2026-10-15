#!/usr/bin/env python3
"""
scorecard.py -- the project's standard benchmark.

Runs the full rotation pipeline on a dataset with known periods and writes
a standard set of metrics, tables and figures. Every agent/human change to
candidate generation or fitting is judged by comparing two scorecard runs
(see check_goals.py), so this script's definitions are the contract:
DON'T change metric definitions casually -- old reports become
incomparable. If a definition must change, bump SCORECARD_VERSION.

What it measures, per star
--------------------------
1. Stage 1 (candidate generation): every discovered guess_* method is run
   on its own; its top-N candidate list is scored with
   evaluation.best_outcome -> correct / diffrot / half / double / wrong /
   none, plus the rank of the match. "Recall" = fraction of stars where a
   *successful* period (P_eq, or within the active-latitude-band range for
   differentially rotating stars) is anywhere in the method's list.
2. Stage 2 (final answer): the candidates from --pipeline-methods are
   pooled and passed to fit_rotation_period; the selected period is
   scored with evaluation.classify_period. A declined answer is "none".
   "success" = correct + diffrot (see evaluation.py).
3. Runtime, exceptions, and pipeline success flag/message.

Datasets
--------
    --synthetic N [--seed S]     synthetic.make_suite(N, seed=S)
    --manifest dev_small.csv     real SMARTS FITS; CSV needs a `path` column
                                 (optional `star_id`, `true_period`, `shear`,
                                 `minlat`, `maxlat`; anything absent is read
                                 from the FITS header: PERIOD, DIFFROT,
                                 MINLAT, MAXLAT). An optional `sectors`
                                 column ("1-2-13") fixes each row's coverage.
    --data-root DIR              resolve relative manifest `path`s against DIR
                                 (default $SMARTS_DATA_ROOT if set, else the
                                 current directory, as before). A relative DIR is
                                 taken from the repo root. The T2 dev sets need
                                 `--data-root data/smarts_dev`; the holdout needs
                                 the SMARTS data root (the Data_Vault drive).
                                 Recorded in config.dataset as ",data_root=DIR".
    --ignore-manifest-sectors    load every native sector (full-year scoring)
                                 even if the manifest has a `sectors` column, for
                                 paired assigned-vs-full-year comparisons on the
                                 same stars. A manifest with a `sectors` column
                                 adds ",sectors=manifest" or ",sectors=ignored"
                                 to config.dataset.
    --coverage SPEC [--coverage-seed S] [--coverage-weighting uniform|frequency]
                                 patchy sector coverage (T16a), applied to either
                                 dataset: full (default, legacy), real-weighted
                                 (patterns drawn by real-star frequency), a class
                                 (single, consecutive-2, gapped-2, consecutive-3-5,
                                 gapped-3-5, consecutive-6-12, gapped-6-12) or group
                                 (3-5, 6-12), with patterns uniform within the class
                                 by default, or pattern:1-2-13. Patterns come from
                                 datasets/coverage/all_cycles_patterns.csv (TESS
                                 years 1-7 pooled; --coverage-library picks another,
                                 e.g. cycle2_patterns.csv; see claude_files/coverage.py).
                                 Non-full coverage is appended to config.dataset, e.g.
                                 "synthetic:n=100,seed=0,coverage=single,coverage_seed=0,coverage_library=all_cycles".

Outputs (in --outdir)
---------------------
    rows.jsonl                   one JSON record per star (resume log)
    per_star.csv                 flattened per-star table, incl. coverage
                                 descriptors (v3): n_sectors, sectors,
                                 coverage_class, total_baseline_days,
                                 longest_contiguous_days, largest_gap_days,
                                 duty_cycle
    per_method.csv               one row per (star, method)
    metrics.json                 summary metrics (what check_goals reads)
    fig_final_outcomes_by_bin.png
    fig_method_recall_by_bin.png
    fig_measured_vs_true.png

Usage
-----
    python benchmarks/scorecard.py --synthetic 200 --seed 0 --outdir runs/synth200_baseline
    python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev --outdir runs/dev_small
    python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev \
        --ignore-manifest-sectors --outdir runs/dev_small_fullyear
    python benchmarks/scorecard.py ... --resume       # skip stars already in rows.jsonl
    python benchmarks/scorecard.py ... --n-workers 1  # serial, for debugging
    python benchmarks/scorecard.py --synthetic 100 --coverage single --outdir runs/synth100_single
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

# Worker setup (T14): pin BLAS to 1 thread BEFORE numpy is imported, here and
# (via the inherited environment) in every spawned pool worker. See
# claude_files/determinism.py for why this is needed for bit-reproducibility.
import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import guesses  # noqa: E402
from guesses import two_view_guesses  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402

SCORECARD_VERSION = 3  # v2 (2026-10-13): diffrot outcome; recall/union_recall count success
# v3 (2026-10-14, T16a): --coverage dataset option + per-star coverage descriptor
# columns. Metric definitions and outcome logic are unchanged from v2, and
# --coverage full (default) reproduces v2 per-star results exactly.
# (2026-10-15, T2: dataset options --data-root and --ignore-manifest-sectors;
# metric definitions unchanged, so no version bump.)
# One worker per physical performance core (4 on the i5 workstation, 10 on a
# 10P+4E Apple Silicon chip). T14 measured 4/8/10 workers on the 4-core/8-thread
# i5: identical throughput within run-to-run noise, so extra hyperthread workers
# only add memory and contention. See determinism.default_n_workers.
DEFAULT_N_WORKERS = determinism.default_n_workers()
DEFAULT_PIPELINE_METHODS = ("pairwise_histogram", "lombscargle", "acf_fft")
# Project convention: n_guesses=10 for most methods, 5 for acf_fft_highpass
# (which returns n_guesses candidates per internal smoothing window).
N_GUESSES = {"acf_fft_highpass": 5}
N_GUESSES_DEFAULT = 10


def discover_guess_functions() -> dict:
    """Return {method_name: fn} for every guess_* in guesses.py taking the
    standard (time, flux, acf_lags, acf) arguments (introspection, so new
    methods are scored automatically)."""
    import inspect
    req = {"time", "flux", "acf_lags", "acf"}
    return dict(sorted(
        (n[len("guess_"):], f) for n, f in inspect.getmembers(guesses, inspect.isfunction)
        if n.startswith("guess_") and req.issubset(inspect.signature(f).parameters)
    ))


def _gap_filled(flux: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN gaps (guess_wavelet needs gap-free flux)."""
    idx = np.arange(len(flux))
    ok = np.isfinite(flux)
    return np.interp(idx, idx[ok], flux[ok])


def _missing(x) -> bool:
    """True for None or non-finite values."""
    return x is None or not np.isfinite(float(x))


def _load(task: dict):
    """Materialize (star_id, time, flux, true_period, period_range, meta, sectors) for one task.

    period_range is the active-latitude-band (P_lo, P_hi) for the DIFFROT
    outcome, or None if the differential-rotation truth isn't available.
    sectors is the list of sectors actually present (for coverage descriptors).
    """
    if task["kind"] == "synthetic":
        lc = task["lc"]
        meta = {k: v for k, v in lc.params.items() if k not in ("spots", "period_range", "sectors")}
        sectors = lc.params.get("sectors") or list(range(1, int(lc.params.get("n_sectors", 1)) + 1))
        return lc.star_id, lc.time, lc.flux, lc.true_period, lc.params.get("period_range"), meta, sectors
    from astropy.io import fits
    from preprocessing import load_smarts_fits
    pre = load_smarts_fits(task["path"], **task.get("load_kwargs", {}))
    with fits.open(task["path"]) as hdul:
        hdr = hdul[0].header
    true_p = task.get("true_period")
    if _missing(true_p):
        true_p = pre.true_period
    shear = task.get("shear") if not _missing(task.get("shear")) else hdr.get("DIFFROT")
    minlat = task.get("minlat") if not _missing(task.get("minlat")) else hdr.get("MINLAT")
    maxlat = task.get("maxlat") if not _missing(task.get("maxlat")) else hdr.get("MAXLAT")
    prange = None
    if not any(_missing(v) for v in (shear, minlat, maxlat)):
        prange = ev.active_band_period_range(float(true_p), float(shear), float(minlat), float(maxlat))
    meta = {"path": str(task["path"]), "shear": shear, "minlat_deg": minlat, "maxlat_deg": maxlat}
    if task.get("load_kwargs", {}).get("sectors") is not None:
        # requested pattern; `sectors` (descriptor column) is what survived flagged-sector removal
        meta["sectors_requested"] = cov.format_pattern(task["load_kwargs"]["sectors"])
    return task["star_id"], pre.time, pre.flux, float(true_p), prange, meta, pre.sectors_used


def _coverage_columns(t: np.ndarray, f: np.ndarray, sectors) -> dict:
    """Per-star coverage descriptor columns (scorecard v3); empty dict if no sectors."""
    if not sectors:
        return {}
    return cov.coverage_descriptors(t, f, sectors)


def score_one_star(task: dict, pipeline_methods: tuple, rel_tol: float) -> dict:
    """Run Stage 1 (every method) and Stage 2 on one star and score both.

    Module-level so ProcessPoolExecutor can pickle it; extra args are bound
    with functools.partial.

    Parameters
    ----------
    task : dict with 'kind' ('synthetic' or 'fits') and either 'lc' or
        'path'/'star_id'/'true_period'/'load_kwargs'.
    pipeline_methods : methods whose candidates are pooled for the fit.
    rel_tol : tolerance passed to evaluation functions.

    Returns
    -------
    dict : JSON-serializable record (see module docstring).
    """
    warnings.simplefilter("ignore")
    rec = {"star_id": task.get("star_id") or getattr(task.get("lc"), "star_id", None)}
    t_start = time.perf_counter()
    try:
        star_id, t, f, true_p, prange, meta, sectors = _load(task)
        rec.update(star_id=star_id, true_period=true_p, period_bin=ev.period_bin(true_p), meta=meta,
                   period_lo=prange[0] if prange else None, period_hi=prange[1] if prange else None)
        rec.update(_coverage_columns(t, f, sectors))
        lags, acf, counts = compute_acf(t, f, return_counts=True)  # Stage 2 (comb fit): ~0.7 B
        lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)  # Stage 1 view: B/3

        methods = {}
        pooled = []
        for name, fn in discover_guess_functions().items():
            t0 = time.perf_counter()
            try:
                flux_in = _gap_filled(f) if name == "wavelet" else f
                out = two_view_guesses(fn, t, flux_in, (lags1, acf1), (lags, acf),
                                       n_guesses=N_GUESSES.get(name, N_GUESSES_DEFAULT))
                outcome, rank = ev.best_outcome([g.P0 for g in out], true_p, rel_tol, prange)
                methods[name] = dict(outcome=outcome, rank=rank, n_candidates=len(out),
                                     top1=(out[0].P0 if out else None), error=None,
                                     seconds=time.perf_counter() - t0)
                if name in pipeline_methods:
                    pooled.extend(out)
            except Exception as exc:  # noqa: BLE001 -- record and continue
                methods[name] = dict(outcome=ev.NONE, rank=None, n_candidates=0, top1=None,
                                     error=f"{type(exc).__name__}: {exc}",
                                     seconds=time.perf_counter() - t0)
        rec["methods"] = methods

        t0 = time.perf_counter()
        res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
        P = float(res.best_fit.P) if res.success else None
        rec.update(final_period=P, final_outcome=ev.classify_period(P, true_p, rel_tol, prange),
                   pipeline_success=bool(res.success), pipeline_message=res.message[:300],
                   n_pooled=len(pooled), fit_seconds=time.perf_counter() - t0, error=None)
    except Exception as exc:  # noqa: BLE001
        rec.update(final_period=None, final_outcome=ev.NONE, pipeline_success=False,
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
    rec["seconds"] = time.perf_counter() - t_start
    return rec


def _coverage_args(args) -> tuple:
    """(spec, seed, weighting, library_path) from CLI args, with defaults for callers lacking them."""
    return (getattr(args, "coverage", cov.FULL), getattr(args, "coverage_seed", 0),
            getattr(args, "coverage_weighting", "uniform"),
            getattr(args, "coverage_library", None) or cov.DEFAULT_LIBRARY)


def sample_coverage(args, n: int) -> list:
    """One sector pattern (tuple) or None (= all sectors) per star, per --coverage."""
    spec, seed, weighting, lib_path = _coverage_args(args)
    if spec == cov.FULL:
        return [None] * n
    lib = None if spec.startswith("pattern:") else cov.load_library(lib_path)
    return cov.sample_patterns(spec, n, seed=seed, library=lib, weighting=weighting)


def manifest_data_root(args):
    """Data root for relative manifest paths: --data-root, else $SMARTS_DATA_ROOT, else None.

    Returns (root_as_given, resolved_path) or (None, None). A relative root is
    resolved against the repo root, so `--data-root data/smarts_dev` works from
    any working directory.
    """
    import os
    given = getattr(args, "data_root", None) or os.environ.get("SMARTS_DATA_ROOT") or None
    if not given:
        return None, None
    given = str(given)
    path = Path(given).expanduser()
    return given, (path if path.is_absolute() else ROOT / path)


def resolve_manifest_path(path_value, root) -> str:
    """Manifest `path` -> path to open: absolute paths unchanged, relative ones joined to `root`."""
    p = Path(str(path_value)).expanduser()
    return str(p if p.is_absolute() or root is None else Path(root) / p)


def _manifest_has_sectors(manifest) -> bool:
    """True if the manifest CSV has a `sectors` column (reads the header only)."""
    try:
        return "sectors" in pd.read_csv(manifest, nrows=0).columns
    except (OSError, ValueError):
        return False


def dataset_string(args) -> str:
    """config.dataset for metrics.json. --coverage full keeps the v2 string, so
    existing baselines and goals.json entries still apply. Manifest runs append
    ",data_root=..." when a data root is used and ",sectors=manifest|ignored" when
    the manifest has a `sectors` column."""
    base = f"synthetic:n={args.synthetic},seed={args.seed}" if args.synthetic else str(args.manifest)
    if not args.synthetic:
        given, _ = manifest_data_root(args)
        if given:
            base += f",data_root={given}"
        if _manifest_has_sectors(args.manifest):
            base += ",sectors=" + ("ignored" if getattr(args, "ignore_manifest_sectors", False) else "manifest")
    spec, seed, weighting, lib_path = _coverage_args(args)
    if spec == cov.FULL:
        return base
    extra = f",coverage={spec}"
    if (spec in cov.COVERAGE_CLASSES or spec in cov.CLASS_GROUPS) and weighting != "uniform":
        extra += f",coverage_weighting={weighting}"
    if not spec.startswith("pattern:"):
        lib_name = Path(lib_path).stem.removesuffix("_patterns")
        extra += f",coverage_seed={seed},coverage_library={lib_name}"
    return base + extra


def build_tasks(args) -> list:
    """Translate CLI dataset arguments into a list of task dicts."""
    if args.synthetic:
        from synthetic import make_suite
        pats = sample_coverage(args, args.synthetic)
        sectors = None if all(p is None for p in pats) else pats
        return [dict(kind="synthetic", lc=lc, star_id=lc.star_id)
                for lc in make_suite(args.synthetic, seed=args.seed, sectors=sectors)]
    man = pd.read_csv(args.manifest)
    if "path" not in man.columns:
        raise SystemExit(f"{args.manifest} needs a 'path' column")
    pats = sample_coverage(args, len(man))
    if "sectors" in man.columns and not getattr(args, "ignore_manifest_sectors", False):
        if _coverage_args(args)[0] != cov.FULL:
            raise SystemExit(f"{args.manifest} has a 'sectors' column; don't also pass --coverage")
        pats = [None if pd.isna(s) or str(s).strip() == "" else cov.parse_pattern(s) for s in man["sectors"]]
    _, root = manifest_data_root(args)
    tasks = []
    for (_, r), pat in zip(man.iterrows(), pats):
        tasks.append(dict(kind="fits", path=resolve_manifest_path(r["path"], root),
                          star_id=str(r["star_id"]) if "star_id" in man.columns else Path(r["path"]).stem,
                          true_period=float(r["true_period"]) if "true_period" in man.columns else None,
                          load_kwargs={} if pat is None else {"sectors": list(pat)}))
    return tasks


def _git_commit() -> str:
    """Current git commit hash (plus '-dirty' if uncommitted changes), or 'unknown'."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "claude_files"], cwd=ROOT)
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def flatten(records: list) -> tuple:
    """Turn JSON records into (per_star, per_method) DataFrames."""
    star_rows, meth_rows = [], []
    for r in records:
        star_rows.append({k: v for k, v in r.items() if k not in ("methods", "meta", "traceback")}
                         | {f"meta_{k}": v for k, v in (r.get("meta") or {}).items()})
        for m, d in (r.get("methods") or {}).items():
            meth_rows.append(dict(star_id=r["star_id"], true_period=r.get("true_period"),
                                  period_lo=r.get("period_lo"), period_hi=r.get("period_hi"),
                                  period_bin=r.get("period_bin"), method=m, **d))
    return pd.DataFrame(star_rows), pd.DataFrame(meth_rows)


def compute_metrics(per_star: pd.DataFrame, per_method: pd.DataFrame, rel_tol: float = ev.REL_TOL) -> dict:
    """Summary metrics. Keys here are what goals.json refers to -- keep stable.

    Layout:
        final.overall.{n,success,correct,diffrot,half,double,alias,wrong,none}
        final.by_bin.<bin>.{...same...}
        methods.<method>.overall.{...outcome fractions..., recall, top1_success, error_rate}
        methods.<method>.by_bin.<bin>.{...}
        union_recall.overall / union_recall.by_bin.<bin>   (any method listed a success period)
    recall/top1/union all count success (correct + diffrot).
        runtime.{median_seconds, p90_seconds}
        errors.n_star_errors
    """
    m: dict = {"final": {}, "methods": {}, "union_recall": {}, "runtime": {}, "errors": {}}
    m["final"]["overall"] = ev.outcome_fractions(per_star["final_outcome"])
    m["final"]["by_bin"] = {b: ev.outcome_fractions(g["final_outcome"])
                            for b, g in per_star.groupby("period_bin")}
    if len(per_method):
        pm = per_method.copy()
        def _rng(lo, hi):
            """Rebuild a period_range tuple from flattened columns (None if absent)."""
            return None if _missing(lo) else (lo, hi)
        pm["top1_outcome"] = [ev.classify_period(top1, p, rel_tol, _rng(lo, hi))
                              for top1, p, lo, hi in zip(pm["top1"], pm["true_period"],
                                                         pm["period_lo"], pm["period_hi"])]

        def block(g):
            """Outcome fractions + recall/top1/error rate for one group."""
            d = ev.outcome_fractions(g["outcome"])
            succ = g["outcome"].isin(ev.SUCCESS_OUTCOMES)
            d["recall"] = d["success"]
            d["top1_success"] = float(np.mean(g["top1_outcome"].isin(ev.SUCCESS_OUTCOMES)))
            d["error_rate"] = float(g["error"].notna().mean())
            d["median_rank"] = float(g.loc[succ, "rank"].median()) if succ.any() else None
            return d

        for meth, g in pm.groupby("method"):
            m["methods"][meth] = {"overall": block(g),
                                  "by_bin": {b: block(gb) for b, gb in g.groupby("period_bin")}}
        hit = pm.assign(hit=pm["outcome"].isin(ev.SUCCESS_OUTCOMES)).groupby(["star_id", "period_bin"])["hit"].any()
        hit = hit.reset_index()
        m["union_recall"]["overall"] = float(hit["hit"].mean())
        m["union_recall"]["by_bin"] = {b: float(g["hit"].mean()) for b, g in hit.groupby("period_bin")}
    m["runtime"] = {"median_seconds": float(per_star["seconds"].median()),
                    "p90_seconds": float(per_star["seconds"].quantile(0.9))}
    m["errors"]["n_star_errors"] = int(per_star["error"].notna().sum())
    return m


OUTCOME_COLORS = {ev.CORRECT: "#2e7d32", ev.DIFFROT: "#81c784", ev.HALF: "#f9a825", ev.DOUBLE: "#ef6c00",
                  ev.WRONG: "#c62828", ev.NONE: "#9e9e9e"}


def make_figures(per_star: pd.DataFrame, per_method: pd.DataFrame, outdir: Path, title: str):
    """Write the three standard scorecard figures to `outdir`."""
    bins = [b for b in ev.PERIOD_BIN_LABELS if b in set(per_star["period_bin"])]

    # 1. stacked final outcomes by period bin
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(bins))
    for o in ev.OUTCOMES:
        vals = np.array([np.mean(per_star.loc[per_star.period_bin == b, "final_outcome"] == o) for b in bins])
        ax.bar(bins, vals, bottom=bottom, color=OUTCOME_COLORS[o], label=o)
        bottom += vals
    counts = per_star["period_bin"].value_counts()
    ax.set_xticks(range(len(bins)), [f"{b}\n(n={counts.get(b, 0)})" for b in bins])
    ax.set_ylabel("fraction of stars")
    ax.set_title(f"Final answer by true period -- {title}")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_final_outcomes_by_bin.png", dpi=130)
    plt.close(fig)

    # 2. per-method recall heatmap
    if len(per_method):
        tab = (per_method.assign(hit=per_method.outcome.isin(ev.SUCCESS_OUTCOMES))
               .pivot_table(index="method", columns="period_bin", values="hit", aggfunc="mean")
               .reindex(columns=bins))
        fig, ax = plt.subplots(figsize=(1.3 * len(bins) + 3, 0.5 * len(tab) + 1.5))
        im = ax.imshow(tab.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(bins)), bins)
        ax.set_yticks(range(len(tab)), tab.index)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="recall (success period in top-N)")
        ax.set_title(f"Stage 1 recall by method -- {title}")
        fig.tight_layout()
        fig.savefig(outdir / "fig_method_recall_by_bin.png", dpi=130)
        plt.close(fig)

    # 3. measured vs true with alias lines
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lo = max(min(per_star["true_period"].min(), 0.1), 0.05)
    hi = per_star["true_period"].max() * 2.5
    xx = np.array([lo, hi])
    for fac, ls in [(1, "-"), (0.5, "--"), (2, "--")]:
        ax.plot(xx, fac * xx, color="0.6", ls=ls, lw=0.8)
    for o in ev.OUTCOMES[:-1]:
        s = per_star[per_star.final_outcome == o]
        ax.scatter(s.true_period, s.final_period, s=12, color=OUTCOME_COLORS[o], label=f"{o} ({len(s)})")
    n_none = int((per_star.final_outcome == ev.NONE).sum())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true period [d]")
    ax.set_ylabel("measured period [d]")
    ax.set_title(f"{title}\n(declined/no answer: {n_none})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_measured_vs_true.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, help="number of synthetic stars")
    src.add_argument("--manifest", type=Path, help="CSV with a 'path' column of SMARTS FITS files")
    p.add_argument("--seed", type=int, default=0, help="synthetic suite seed")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--pipeline-methods", default=",".join(DEFAULT_PIPELINE_METHODS),
                   help="comma-separated methods pooled for the final fit")
    p.add_argument("--rel-tol", type=float, default=ev.REL_TOL)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--resume", action="store_true", help="skip stars already in rows.jsonl")
    p.add_argument("--label", default=None, help="short name for figure titles (default: outdir name)")
    p.add_argument("--coverage", default=cov.FULL,
                   help=f"sector coverage: one of {cov.valid_coverage_specs()} or pattern:1-2-13 (default full)")
    p.add_argument("--coverage-seed", type=int, default=0, help="seed for drawing coverage patterns")
    p.add_argument("--coverage-weighting", choices=("uniform", "frequency"), default="uniform",
                   help="for a class --coverage: patterns equally likely (uniform) or by real-star count")
    p.add_argument("--coverage-library", type=Path, default=None,
                   help="coverage library CSV (default datasets/coverage/all_cycles_patterns.csv)")
    p.add_argument("--data-root", default=None,
                   help="root for relative manifest paths (default $SMARTS_DATA_ROOT, else cwd); "
                        "the T2 dev sets need data/smarts_dev")
    p.add_argument("--ignore-manifest-sectors", action="store_true",
                   help="ignore the manifest's `sectors` column (score every native sector)")
    args = p.parse_args(argv)
    if args.coverage not in cov.valid_coverage_specs() and not args.coverage.startswith("pattern:"):
        p.error(f"--coverage must be one of {cov.valid_coverage_specs()} or pattern:<sectors>")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "rows.jsonl"
    done = {}
    if args.resume and log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["star_id"]] = r
    elif log.exists():
        log.unlink()

    tasks = [t for t in build_tasks(args) if t["star_id"] not in done]
    missing = [t["path"] for t in tasks if t["kind"] == "fits" and not Path(t["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(tasks)} manifest files not found (first: {missing[0]}). "
                         "Pass --data-root (dev sets: data/smarts_dev) or set $SMARTS_DATA_ROOT; "
                         "for drive data, check the Data_Vault drive is plugged in.")
    methods = tuple(m.strip() for m in args.pipeline_methods.split(",") if m.strip())
    worker = partial(score_one_star, pipeline_methods=methods, rel_tol=args.rel_tol)
    determinism.warn_if_nondeterministic_blas()
    print(f"{len(done)} already done, {len(tasks)} to run, {args.n_workers} worker(s)")

    t0 = time.time()
    with open(log, "a") as fh:
        def _write(rec):
            """Append one record to the resume log and report progress."""
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            done[rec["star_id"]] = rec
            print(f"  [{len(done)}] {rec['star_id']}: P_true={rec.get('true_period', float('nan')):.3f} "
                  f"-> {rec['final_outcome']} ({rec['seconds']:.1f}s)", flush=True)

        if args.n_workers <= 1:
            for t in tasks:
                _write(worker(t))
        else:
            with ProcessPoolExecutor(max_workers=args.n_workers, initializer=determinism.init_worker) as ex:
                futs = [ex.submit(worker, t) for t in tasks]
                for fut in as_completed(futs):
                    _write(fut.result())

    per_star, per_method = flatten(list(done.values()))
    per_star.to_csv(outdir / "per_star.csv", index=False)
    per_method.to_csv(outdir / "per_method.csv", index=False)
    metrics = compute_metrics(per_star, per_method, args.rel_tol)
    metrics["config"] = dict(
        scorecard_version=SCORECARD_VERSION, git_commit=_git_commit(), rel_tol=args.rel_tol,
        dataset=dataset_string(args),
        coverage=dict(spec=args.coverage, seed=args.coverage_seed, weighting=args.coverage_weighting,
                      library=str(args.coverage_library or cov.DEFAULT_LIBRARY.relative_to(ROOT))),
        data_root=manifest_data_root(args)[0] if args.manifest else None,
        ignore_manifest_sectors=bool(args.ignore_manifest_sectors),
        pipeline_methods=list(methods), n_guesses_default=N_GUESSES_DEFAULT, n_guesses_overrides=N_GUESSES,
        wall_seconds=time.time() - t0, created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    make_figures(per_star, per_method, outdir, args.label or outdir.name)

    f = metrics["final"]["overall"]
    print(f"\nFinal answer (n={f['n']}): success {f['success']:.1%} (correct {f['correct']:.1%}, "
          f"diffrot {f['diffrot']:.1%})  alias {f['alias']:.1%} "
          f"(half {f['half']:.1%}, double {f['double']:.1%})  wrong {f['wrong']:.1%}  none {f['none']:.1%}")
    print(f"Wrote {outdir}/metrics.json and figures.")


if __name__ == "__main__":
    main()
