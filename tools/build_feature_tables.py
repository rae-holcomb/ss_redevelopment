#!/usr/bin/env python3
"""Build ranker feature tables (task T10): one row per candidate, every candidate, graded labels.

For each star in a manifest, run exactly the scorecard's pipeline (same loader, same
Stage 1 methods and n_guesses, same pooled methods, same ``fit_rotation_period`` call),
then write one row per candidate the fit saw:

- fitted candidates, including gate-failed ones (``ml_features.extract_candidate_features``
  plus every gate's value / pass / threshold / enabled from the scorecard's per-candidate
  records);
- candidates whose fit raised (``fit_error``) and candidates beyond the searchable range
  (``beyond_reach``, T16c), with the features that don't need a fit, evaluated at P0
  (``ml_features.extract_unfit_candidate_features``).

Labels (roadmap decision, Rae 2026-11-09): ``relevance`` = 2 for ``correct`` (within
REL_TOL of P_eq), 1 for ``diffrot`` (inside ``evaluation.active_band_period_range``),
0 otherwise, from ``evaluation.classify_period`` on the candidate's fitted period
(P0 when there's no fit). The alias class (half / double) is a column, not a label.
Every row carries ``star_id`` and ``noise_tic`` (the TESS noise source) for grouped
splits, plus the T20 noise flags and ``noise_coherent_quiet`` (the 2026-11-09 review's
list of quiet sources with coherent short-period signals).

Non-rotator negatives (T27) append with the same schema: ``is_rotator`` = False,
``label_source`` = their origin, ``true_period`` NaN, ``relevance`` 0. See
``docs/FEATURE_TABLE_SCHEMA.md`` for every column.

Outputs (in --outdir)
---------------------
    shards/<star_id>.pkl   per-star {"star": dict, "candidates": DataFrame}, written
                           atomically (resume log: --resume skips stars with a shard)
    candidates.csv.gz      all candidate rows (combiner)
    stars.csv.gz           one row per star: truth, final answer, counts, Stage 1 outcomes
    config.json            arguments, dataset string, git commit

Usage
-----
    python tools/build_feature_tables.py --manifest datasets/dev_medium.csv \
        --data-root data/smarts_dev --outdir runs/t10_dev_medium_assigned
    python tools/build_feature_tables.py --manifest datasets/dev_medium.csv \
        --data-root data/smarts_dev --ignore-manifest-sectors --outdir runs/t10_dev_medium_fullyear
    python tools/build_feature_tables.py --manifest datasets/train_v0.csv \
        --only nested_1k --outdir /Volumes/Data_Vault/smarts_data/extracted/t10_tables/train_1k
    ... --resume          # skip stars that already have a shard
    ... --combine-only    # just rebuild candidates.csv.gz / stars.csv.gz from the shards
    ... --combine-only --refresh-noise   # same, re-joining noise_* from the current flags file
    ... --n-workers 1     # serial, for debugging
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))

import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import ml_features  # noqa: E402
import scorecard as sc  # noqa: E402  (read-only reuse: same loader / methods / fit call)
import smarts_data as sd  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import GATE_NAMES, fit_rotation_period  # noqa: E402
from guesses import two_view_guesses  # noqa: E402

FLAGS_PATH = "extracted/galaxy_ids/noise_source_flags.csv.gz"  # relative to SMARTS_DATA_ROOT (T20)
COHERENT_LIST = ROOT / "reports" / "2026-11-10_t10-ranker-feature-tables" / "data" / "quiet_coherent.csv"
FLAG_COLUMNS = {"tic": "noise_tic", "classification": "noise_classification",
                "periodic": "noise_periodic", "exclude_from_training": "noise_exclude_from_training"}
# Manifest columns copied into the star table (when present).
STAR_CONTEXT = ["row", "sim_id", "pbin", "eval_pbin", "amp_tier", "shear_class", "stratum",
                "coverage_class", "sampling_weight", "coverage_weight", "combined_weight",
                "train_order", "nested_1k"]
# Per-row star context (every candidate row carries these).
ROW_CONTEXT = ["noise_tic", "noise_classification", "noise_periodic",
               "noise_exclude_from_training", "noise_coherent_quiet"]
RELEVANCE = {ev.CORRECT: 2, ev.DIFFROT: 1}
ALIAS_OUTCOMES = (ev.HALF, ev.DOUBLE)
LABEL_SOURCE = "smarts_injected"

# Columns that are identifiers, truth or bookkeeping: never model inputs (T11 uses
# feature_columns()). Everything else in candidates.csv.gz is a candidate feature.
NON_FEATURE_COLUMNS = [
    "star_id", "noise_tic", "noise_classification", "noise_periodic", "noise_exclude_from_training",
    "noise_coherent_quiet", "is_rotator", "label_source", "true_period", "period_lo", "period_hi",
    "period_bin", "P_eval", "outcome", "relevance", "alias_class", "selected", "cand_index",
    "contributing_methods", "failed_gates", "fit_error_message", "t0", "fitted_t0",
]


def feature_columns(df: pd.DataFrame) -> list:
    """Model-input columns of a candidate table: everything not in NON_FEATURE_COLUMNS.

    String columns (``method``) are kept; T11 decides how to encode them.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def relevance_label(outcome: str) -> int:
    """Graded relevance for a candidate outcome: 2 correct, 1 diffrot, 0 anything else."""
    return RELEVANCE.get(outcome, 0)


def _gate_columns(rec: dict) -> dict:
    """Rename one scorecard per-candidate record into ``gate_*`` and bookkeeping columns."""
    out = dict(passed_base_gates=rec["passed_base_gates"], selected=rec["selected"],
               failed_gates=rec["failed_gates"],
               n_failed_gates=len([g for g in rec["failed_gates"].split(";") if g]),
               contributing_methods=rec["contributing_methods"],
               acf_pair_frac_at_P=rec.get("acf_pair_frac_at_P"))
    for g in GATE_NAMES:
        for k in ("value", "pass", "threshold", "enabled"):
            if f"{g}_{k}" in rec:
                out[f"gate_{g}_{k}"] = rec[f"{g}_{k}"]
    return out


def candidate_table(res, pooled, all_guesses, t, f, lags, acf, counts, star_id, true_p, prange,
                    rel_tol) -> pd.DataFrame:
    """All candidate rows for one star: fitted (with gates) + fit-error + beyond-reach, labelled.

    Parameters
    ----------
    res : EnsembleResult from fit_rotation_period(lags, acf, pooled, acf_counts=counts).
    pooled : the guesses passed to the fit (kept for the record; features use all_guesses).
    all_guesses : every Stage 1 method's guesses (so method-spectrum features cover all methods).
    t, f : light curve; lags, acf, counts : the Stage 2 ACF and its pair counts.
    star_id, true_p, prange : identity and truth (prange = active-band (P_lo, P_hi) or None).
    rel_tol : evaluation tolerance (REL_TOL).

    Returns
    -------
    DataFrame (possibly empty) with features, gate columns and label columns.
    """
    parts = []
    recs = sc._candidate_rows(res, lags, counts, true_p, rel_tol, prange)
    n_fit = len(res.candidates)
    if n_fit:
        fit_df = ml_features.extract_candidate_features(res, all_guesses, t, f, lags, acf, star_id=star_id)
        gates = pd.DataFrame([_gate_columns(r) for r in recs[:n_fit]])
        if not np.allclose(fit_df["period"].to_numpy(float), [r["P0"] for r in recs[:n_fit]]):
            raise RuntimeError("candidate order mismatch between ml_features and scorecard records")
        fit_df = pd.concat([fit_df.reset_index(drop=True), gates], axis=1)
        fit_df["fit_error"] = False
        fit_df["beyond_reach"] = False
        fit_df["P_eval"] = fit_df["fitted_P"]
        parts.append(fit_df)
    unfit = ml_features.extract_unfit_candidate_features(res, all_guesses, t, f, lags, acf, star_id=star_id)
    if len(unfit):
        n_err = len(res.failed_candidates)
        # scorecard records: fitted, then fit-error, then (v5, P22b) beyond-reach rows
        failed_recs = recs[n_fit:n_fit + n_err]
        p0 = [float(c.period) for c in list(res.failed_candidates) + list(res.beyond_reach_candidates)]
        if len(recs) != n_fit + len(p0) or not np.allclose([r["P0"] for r in recs[n_fit:]], p0):
            raise RuntimeError("unfit-candidate order mismatch with scorecard records (expects scorecard v5)")
        unfit["selected"] = False
        unfit["passed_base_gates"] = False
        unfit["failed_gates"] = ["fit_error"] * n_err + ["beyond_reach"] * (len(unfit) - n_err)
        unfit["n_failed_gates"] = np.nan
        unfit["contributing_methods"] = ["+".join(c.contributing_methods) for c in
                                         list(res.failed_candidates) + list(res.beyond_reach_candidates)]
        unfit["fit_error_message"] = [r.get("error") for r in failed_recs] + [None] * (len(unfit) - n_err)
        unfit["P_eval"] = unfit["period"]
        parts.append(unfit)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["cand_index"] = np.arange(len(df))
    df["outcome"] = [ev.classify_period(p, true_p, rel_tol, prange) for p in df["P_eval"]]
    df["relevance"] = df["outcome"].map(relevance_label).astype(int)
    df["alias_class"] = df["outcome"].where(df["outcome"].isin(ALIAS_OUTCOMES), "")
    return df


def build_one_star(task: dict, shard_dir: str, pipeline_methods: tuple, rel_tol: float) -> dict:
    """Run the scorecard pipeline on one star and write its feature shard atomically.

    Module-level so ProcessPoolExecutor can pickle it (extra args via functools.partial).
    Mirrors ``scorecard.score_one_star`` step for step, so the selected row's period
    equals the scorecard's final answer for the same star and code.

    Parameters
    ----------
    task : scorecard task dict plus ``context`` (manifest + noise-flag columns).
    shard_dir : directory for ``<star_id>.pkl``.
    pipeline_methods : methods pooled into the fit (scorecard default).
    rel_tol : evaluation tolerance.

    Returns
    -------
    dict summary: star_id, n_rows, error, seconds.
    """
    warnings.simplefilter("ignore")
    ctx = dict(task.get("context", {}))
    star = dict(star_id=task["star_id"], is_rotator=True, label_source=LABEL_SOURCE)
    star.update(ctx)  # T27 can override is_rotator / label_source via the context
    cands = pd.DataFrame()
    t_start = time.perf_counter()
    try:
        star_id, t, f, true_p, prange, meta, sectors = sc._load(task)
        star.update(true_period=true_p, period_bin=ev.period_bin(true_p),
                    period_lo=prange[0] if prange else None, period_hi=prange[1] if prange else None,
                    sectors_requested=meta.get("sectors_requested"))
        star.update(sc._coverage_columns(t, f, sectors))
        star.update(sc._searchability_columns(t, f, true_p, prange))
        lags, acf, counts = compute_acf(t, f, return_counts=True)
        lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
        pooled, all_guesses = [], []
        for name, fn in sc.discover_guess_functions().items():
            try:
                flux_in = sc._gap_filled(f) if name == "wavelet" else f
                out = two_view_guesses(fn, t, flux_in, (lags1, acf1), (lags, acf),
                                       n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT))
                star[f"stage1_{name}_outcome"] = ev.best_outcome([g.P0 for g in out], true_p, rel_tol, prange)[0]
                all_guesses.extend(out)
                if name in pipeline_methods:
                    pooled.extend(out)
            except Exception as exc:  # noqa: BLE001 -- same policy as the scorecard
                star[f"stage1_{name}_outcome"] = ev.NONE
                star[f"stage1_{name}_error"] = f"{type(exc).__name__}: {exc}"
        res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
        P = float(res.best_fit.P) if res.success else None
        reason = res.decline_reason or {}
        star.update(final_period=P, final_outcome=ev.classify_period(P, true_p, rel_tol, prange),
                    pipeline_success=bool(res.success), decline_kind=reason.get("kind", ""),
                    decline_gates=";".join(reason.get("gates", [])), n_pooled=len(pooled),
                    n_fit_candidates=len(res.candidates), n_fit_error=len(res.failed_candidates),
                    n_beyond_reach=len(res.beyond_reach_candidates), error=None)
        cands = candidate_table(res, pooled, all_guesses, t, f, lags, acf, counts, star_id, true_p,
                                prange, rel_tol)
        if len(cands):
            for c in ROW_CONTEXT:
                cands.insert(1, c, ctx.get(c))
            cands["is_rotator"] = star["is_rotator"]
            cands["label_source"] = star["label_source"]
            cands["true_period"] = true_p
            cands["period_lo"] = star["period_lo"]
            cands["period_hi"] = star["period_hi"]
            cands["period_bin"] = star["period_bin"]
        star.update(n_rows=len(cands),
                    max_relevance=int(cands["relevance"].max()) if len(cands) else 0,
                    n_relevant=int((cands["relevance"] >= 1).sum()) if len(cands) else 0,
                    n_relevant_fitted=int(((cands["relevance"] >= 1) & ~cands["fit_error"]
                                           & ~cands["beyond_reach"]).sum()) if len(cands) else 0)
    except Exception as exc:  # noqa: BLE001 -- recorded, the run continues
        star.update(final_period=None, final_outcome=ev.NONE, pipeline_success=False, n_rows=0,
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        cands = pd.DataFrame()
    star["seconds"] = time.perf_counter() - t_start
    with sd.atomic_path(Path(shard_dir) / f"{task['star_id']}.pkl") as tmp:
        with open(tmp, "wb") as fh:
            pickle.dump({"star": star, "candidates": cands}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(star_id=task["star_id"], n_rows=len(cands), error=star.get("error"), seconds=star["seconds"])


def load_noise_context(man: pd.DataFrame, flags_path, coherent_path) -> pd.DataFrame:
    """Per-manifest-row noise context: T20 flags joined on catalog row, plus noise_coherent_quiet.

    Returns a DataFrame aligned with ``man`` with the ROW_CONTEXT columns (NaN where a
    row is missing from the flags file).
    """
    fl = pd.read_csv(flags_path, usecols=["row"] + list(FLAG_COLUMNS))
    fl = fl.rename(columns=FLAG_COLUMNS)
    out = man[["row"]].merge(fl, on="row", how="left", validate="1:1")
    coherent = set(pd.read_csv(coherent_path)["tic"].astype("int64"))
    out["noise_coherent_quiet"] = out["noise_tic"].isin(coherent)
    return out.drop(columns="row")


def build_tasks(args) -> list:
    """Scorecard task dicts for the manifest, each with a ``context`` of star metadata."""
    ns = argparse.Namespace(synthetic=None, manifest=args.manifest, data_root=args.data_root,
                            ignore_manifest_sectors=args.ignore_manifest_sectors)
    tasks = sc.build_tasks(ns)
    man = pd.read_csv(args.manifest)
    if "row" not in man.columns:
        man["row"] = [int(Path(p).stem.split("-")[-1]) for p in man["path"]]
    noise = load_noise_context(man, args.flags, args.coherent_list) if args.flags else None
    keep = np.ones(len(man), bool)
    if args.only:
        keep = man[args.only].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    out = []
    for i, task in enumerate(tasks):
        if not keep[i]:
            continue
        ctx = {c: man.at[i, c] for c in STAR_CONTEXT if c in man.columns}
        if noise is not None:
            ctx.update(noise.iloc[i].to_dict())
        ctx = {k: (v.item() if hasattr(v, "item") else v) for k, v in ctx.items()}
        task["context"] = ctx
        out.append(task)
    if args.limit:
        out = out[: args.limit]
    return out


def combine(outdir: Path, noise: pd.DataFrame | None = None) -> tuple:
    """Concatenate every shard into candidates.csv.gz and stars.csv.gz (atomic writes).

    noise : optional DataFrame indexed by star_id with the ROW_CONTEXT columns; when given
        (``--refresh-noise``), those columns are overwritten in both tables from it, so a
        flags-file update doesn't need a rebuild (the shards themselves are left as built).

    Returns (n_stars, n_candidate_rows).
    """
    stars, cands = [], []
    for p in sorted((outdir / "shards").glob("*.pkl")):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        stars.append(d["star"])
        if len(d["candidates"]):
            cands.append(d["candidates"])
    st = pd.DataFrame(stars)
    ca = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()
    if noise is not None:
        for df in (st, ca):
            if len(df):
                for c in ROW_CONTEXT:
                    df[c] = df["star_id"].map(noise[c])
    with sd.atomic_path(outdir / "stars.csv.gz") as tmp:
        st.to_csv(tmp, index=False)
    with sd.atomic_path(outdir / "candidates.csv.gz") as tmp:
        ca.to_csv(tmp, index=False)
    return len(st), len(ca)


def main(argv=None):
    """CLI entry point: build tasks, run the per-star workers (resumable), combine shards."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None,
                    help="root for relative manifest paths (default $SMARTS_DATA_ROOT; dev sets: data/smarts_dev)")
    ap.add_argument("--ignore-manifest-sectors", action="store_true", help="full-year: load every native sector")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--n-workers", type=int, default=sc.DEFAULT_N_WORKERS)
    ap.add_argument("--resume", action="store_true", help="skip stars that already have a shard")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--refresh-noise", action="store_true",
                    help="with --combine-only: re-join the noise_* columns from the current flags file")
    ap.add_argument("--only", default=None, help="manifest boolean column selecting rows (e.g. nested_1k)")
    ap.add_argument("--limit", type=int, default=None, help="first N selected rows (quick checks)")
    ap.add_argument("--smarts-root", default=None, help="SMARTS data root for the noise flags (default $SMARTS_DATA_ROOT or the Data_Vault path)")
    ap.add_argument("--flags", default=None, help="noise_source_flags.csv.gz (default <smarts-root>/" + FLAGS_PATH + ")")
    ap.add_argument("--no-flags", action="store_true", help="skip the noise-flag join (noise_* columns empty)")
    ap.add_argument("--coherent-list", default=str(COHERENT_LIST))
    args = ap.parse_args(argv)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    (outdir / "shards").mkdir(parents=True, exist_ok=True)
    if args.no_flags:
        args.flags = None
    elif args.flags is None and (args.refresh_noise or not args.combine_only):
        root = sd.require_drive(sd.data_root(args.smarts_root))  # stops cleanly if unplugged
        args.flags = str(Path(root) / FLAGS_PATH)
    if args.combine_only:
        noise = None
        if args.refresh_noise:
            man = pd.read_csv(args.manifest)
            if "row" not in man.columns:
                man["row"] = [int(Path(p).stem.split("-")[-1]) for p in man["path"]]
            noise = load_noise_context(man, args.flags, args.coherent_list)
            noise.index = man["star_id"] if "star_id" in man.columns else [Path(p).stem for p in man["path"]]
        print("combined: %d stars, %d candidate rows" % combine(outdir, noise))
        return
    tasks = build_tasks(args)
    missing = [t["path"] for t in tasks if not Path(t["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest files missing (first: {missing[0]}); wrong --data-root, "
                         "or the drive is unplugged?")
    if args.resume:
        done = {p.stem for p in (outdir / "shards").glob("*.pkl")}
        tasks = [t for t in tasks if t["star_id"] not in done]
    cfg = dict(manifest=str(args.manifest), data_root=args.data_root,
               ignore_manifest_sectors=args.ignore_manifest_sectors, only=args.only, limit=args.limit,
               flags=args.flags, coherent_list=args.coherent_list,
               dataset=sc.dataset_string(argparse.Namespace(
                   synthetic=None, manifest=args.manifest, data_root=args.data_root,
                   ignore_manifest_sectors=args.ignore_manifest_sectors)),
               pipeline_methods=list(sc.DEFAULT_PIPELINE_METHODS), rel_tol=ev.REL_TOL,
               scorecard_version=sc.SCORECARD_VERSION, git_commit=sc._git_commit())
    with sd.atomic_path(outdir / "config.json") as tmp:
        tmp.write_text(json.dumps(cfg, indent=2, default=str))
    worker = partial(build_one_star, shard_dir=str(outdir / "shards"),
                     pipeline_methods=sc.DEFAULT_PIPELINE_METHODS, rel_tol=ev.REL_TOL)
    print(f"{len(tasks)} stars to build -> {outdir} ({args.n_workers} workers)", flush=True)
    t0 = time.time()
    n_err = 0
    if args.n_workers <= 1:
        it = (worker(t) for t in tasks)
    else:
        ex = ProcessPoolExecutor(max_workers=args.n_workers, initializer=determinism.init_worker)
        it = (fut.result() for fut in as_completed([ex.submit(worker, t) for t in tasks]))
    for i, r in enumerate(it):
        n_err += r["error"] is not None
        if (i + 1) % 25 == 0 or i + 1 == len(tasks):
            el = time.time() - t0
            print(f"  {i + 1}/{len(tasks)} stars, {el:.0f} s elapsed, ~{el / (i + 1) * (len(tasks) - i - 1):.0f} s left, "
                  f"{n_err} errors", flush=True)
    if args.n_workers > 1:
        ex.shutdown()
    print("combined: %d stars, %d candidate rows" % combine(outdir))


if __name__ == "__main__":
    main()
