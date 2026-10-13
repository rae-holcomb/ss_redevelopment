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
   evaluation.best_outcome -> correct / half / double / wrong / none, plus
   the rank of the match. "Recall" = fraction of stars where the true
   period is anywhere in the method's list.
2. Stage 2 (final answer): the candidates from --pipeline-methods are
   pooled and passed to fit_rotation_period; the selected period is
   scored with evaluation.classify_period. A declined answer is "none".
3. Runtime, exceptions, and pipeline success flag/message.

Datasets
--------
    --synthetic N [--seed S]     synthetic.make_suite(N, seed=S)
    --manifest dev_small.csv     real SMARTS FITS; CSV needs a `path` column
                                 (optional `star_id`, `true_period`; if
                                 true_period is absent it's read from the
                                 FITS header via load_smarts_fits)

Outputs (in --outdir)
---------------------
    rows.jsonl                   one JSON record per star (resume log)
    per_star.csv                 flattened per-star table
    per_method.csv               one row per (star, method)
    metrics.json                 summary metrics (what check_goals reads)
    fig_final_outcomes_by_bin.png
    fig_method_recall_by_bin.png
    fig_measured_vs_true.png

Usage
-----
    python benchmarks/scorecard.py --synthetic 200 --seed 0 --outdir runs/synth200_baseline
    python benchmarks/scorecard.py --manifest datasets/dev_small.csv --outdir runs/dev_small_baseline
    python benchmarks/scorecard.py ... --resume       # skip stars already in rows.jsonl
    python benchmarks/scorecard.py ... --n-workers 1  # serial, for debugging
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import evaluation as ev  # noqa: E402
import guesses  # noqa: E402
from acf_utils import compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402

SCORECARD_VERSION = 1
DEFAULT_N_WORKERS = 10
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


def _load(task: dict):
    """Materialize (star_id, time, flux, true_period, meta) for one task."""
    if task["kind"] == "synthetic":
        lc = task["lc"]
        meta = {k: v for k, v in lc.params.items() if k != "spots"}
        return lc.star_id, lc.time, lc.flux, lc.true_period, meta
    from preprocessing import load_smarts_fits
    pre = load_smarts_fits(task["path"], **task.get("load_kwargs", {}))
    true_p = task.get("true_period")
    if true_p is None or not np.isfinite(true_p):
        true_p = pre.true_period
    return task["star_id"], pre.time, pre.flux, float(true_p), {"path": str(task["path"])}


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
        star_id, t, f, true_p, meta = _load(task)
        rec.update(star_id=star_id, true_period=true_p, period_bin=ev.period_bin(true_p), meta=meta)
        lags, acf = compute_acf(t, f)

        methods = {}
        pooled = []
        for name, fn in discover_guess_functions().items():
            t0 = time.perf_counter()
            try:
                flux_in = _gap_filled(f) if name == "wavelet" else f
                out = fn(t, flux_in, lags, acf, n_guesses=N_GUESSES.get(name, N_GUESSES_DEFAULT))
                outcome, rank = ev.best_outcome([g.P0 for g in out], true_p, rel_tol)
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
        res = fit_rotation_period(lags, acf, pooled)
        P = float(res.best_fit.P) if res.success else None
        rec.update(final_period=P, final_outcome=ev.classify_period(P, true_p, rel_tol),
                   pipeline_success=bool(res.success), pipeline_message=res.message[:300],
                   n_pooled=len(pooled), fit_seconds=time.perf_counter() - t0, error=None)
    except Exception as exc:  # noqa: BLE001
        rec.update(final_period=None, final_outcome=ev.NONE, pipeline_success=False,
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
    rec["seconds"] = time.perf_counter() - t_start
    return rec


def build_tasks(args) -> list:
    """Translate CLI dataset arguments into a list of task dicts."""
    if args.synthetic:
        from synthetic import make_suite
        return [dict(kind="synthetic", lc=lc, star_id=lc.star_id)
                for lc in make_suite(args.synthetic, seed=args.seed)]
    man = pd.read_csv(args.manifest)
    if "path" not in man.columns:
        raise SystemExit(f"{args.manifest} needs a 'path' column")
    tasks = []
    for _, r in man.iterrows():
        tasks.append(dict(kind="fits", path=r["path"],
                          star_id=str(r["star_id"]) if "star_id" in man.columns else Path(r["path"]).stem,
                          true_period=float(r["true_period"]) if "true_period" in man.columns else None))
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
                                  period_bin=r.get("period_bin"), method=m, **d))
    return pd.DataFrame(star_rows), pd.DataFrame(meth_rows)


def compute_metrics(per_star: pd.DataFrame, per_method: pd.DataFrame, rel_tol: float = ev.REL_TOL) -> dict:
    """Summary metrics. Keys here are what goals.json refers to -- keep stable.

    Layout:
        final.overall.{n,correct,half,double,alias,wrong,none}
        final.by_bin.<bin>.{...same...}
        methods.<method>.overall.{...outcome fractions..., recall, top1_correct, error_rate}
        methods.<method>.by_bin.<bin>.{...}
        union_recall.overall / union_recall.by_bin.<bin>   (any method found P)
        runtime.{median_seconds, p90_seconds}
        errors.n_star_errors
    """
    m: dict = {"final": {}, "methods": {}, "union_recall": {}, "runtime": {}, "errors": {}}
    m["final"]["overall"] = ev.outcome_fractions(per_star["final_outcome"])
    m["final"]["by_bin"] = {b: ev.outcome_fractions(g["final_outcome"])
                            for b, g in per_star.groupby("period_bin")}
    if len(per_method):
        pm = per_method.copy()
        pm["top1_outcome"] = [ev.classify_period(top1, p, rel_tol)
                              for top1, p in zip(pm["top1"], pm["true_period"])]

        def block(g):
            """Outcome fractions + recall/top1/error rate for one group."""
            d = ev.outcome_fractions(g["outcome"])
            d["recall"] = d["correct"]
            d["top1_correct"] = float(np.mean(g["top1_outcome"] == ev.CORRECT))
            d["error_rate"] = float(g["error"].notna().mean())
            d["median_rank"] = float(g.loc[g["outcome"] == ev.CORRECT, "rank"].median()) \
                if (g["outcome"] == ev.CORRECT).any() else None
            return d

        for meth, g in pm.groupby("method"):
            m["methods"][meth] = {"overall": block(g),
                                  "by_bin": {b: block(gb) for b, gb in g.groupby("period_bin")}}
        hit = pm.assign(hit=pm["outcome"] == ev.CORRECT).groupby(["star_id", "period_bin"])["hit"].any()
        hit = hit.reset_index()
        m["union_recall"]["overall"] = float(hit["hit"].mean())
        m["union_recall"]["by_bin"] = {b: float(g["hit"].mean()) for b, g in hit.groupby("period_bin")}
    m["runtime"] = {"median_seconds": float(per_star["seconds"].median()),
                    "p90_seconds": float(per_star["seconds"].quantile(0.9))}
    m["errors"]["n_star_errors"] = int(per_star["error"].notna().sum())
    return m


OUTCOME_COLORS = {ev.CORRECT: "#2e7d32", ev.HALF: "#f9a825", ev.DOUBLE: "#ef6c00",
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
        tab = (per_method.assign(hit=per_method.outcome == ev.CORRECT)
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
        fig.colorbar(im, ax=ax, label="recall (true P in top-N)")
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
    args = p.parse_args(argv)

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
    methods = tuple(m.strip() for m in args.pipeline_methods.split(",") if m.strip())
    worker = partial(score_one_star, pipeline_methods=methods, rel_tol=args.rel_tol)
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
            with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                futs = [ex.submit(worker, t) for t in tasks]
                for fut in as_completed(futs):
                    _write(fut.result())

    per_star, per_method = flatten(list(done.values()))
    per_star.to_csv(outdir / "per_star.csv", index=False)
    per_method.to_csv(outdir / "per_method.csv", index=False)
    metrics = compute_metrics(per_star, per_method, args.rel_tol)
    metrics["config"] = dict(
        scorecard_version=SCORECARD_VERSION, git_commit=_git_commit(), rel_tol=args.rel_tol,
        dataset=(f"synthetic:n={args.synthetic},seed={args.seed}" if args.synthetic else str(args.manifest)),
        pipeline_methods=list(methods), n_guesses_default=N_GUESSES_DEFAULT, n_guesses_overrides=N_GUESSES,
        wall_seconds=time.time() - t0, created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    make_figures(per_star, per_method, outdir, args.label or outdir.name)

    f = metrics["final"]["overall"]
    print(f"\nFinal answer (n={f['n']}): correct {f['correct']:.1%}  alias {f['alias']:.1%} "
          f"(half {f['half']:.1%}, double {f['double']:.1%})  wrong {f['wrong']:.1%}  none {f['none']:.1%}")
    print(f"Wrote {outdir}/metrics.json and figures.")


if __name__ == "__main__":
    main()
