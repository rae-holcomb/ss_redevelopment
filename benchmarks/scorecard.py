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
    --coverage SPEC [--coverage-seed S] [--coverage-weighting frequency|uniform]
                                 patchy sector coverage (T16a), applied to either
                                 dataset: full (default, legacy), real-weighted
                                 (patterns drawn by real-star frequency), a class
                                 (single, consecutive-2, gapped-2, consecutive-3-5,
                                 gapped-3-5, consecutive-6-12, gapped-6-12) or group
                                 (3-5, 6-12), with patterns drawn by real-star frequency
                                 within the class by default, or pattern:1-2-13. Patterns come from
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
                                 duty_cycle; and (v4) acf_reach (frozen
                                 evaluation.acf_reach), P_over_reach (P_eq / R),
                                 searchable (P_lo <= R/2), band_width
                                 ((P_hi - P_lo) / P_eq), band_width_class,
                                 flux_amp (95th - 5th percentile of the flux)
                                 (v5, manifests with sim_id; join in
                                 benchmarks/noise_sources_dev.csv, built by
                                 tools/build_noise_table.py): noise_tic, noise_class,
                                 noise_periodic, noise_exclude, noise_quiet_coherent,
                                 noise_P_ref, noise_period_match (a `wrong` answer within
                                 REL_TOL of the noise source's P_ref; never a success),
                                 noise_alias ('half'/'double' of it; diagnostic)
                                 (v6, T26/P26; join of benchmarks/detectability_dev.csv,
                                 built by tools/build_detectability_table.py, on
                                 (star_id, loaded sectors)): detectability (snr_rot =
                                 D_band*sqrt(N_cyc), injected-signal S/N on rotation
                                 timescales) and detectable (detectability >=
                                 --detectable-threshold, default 5); empty where the star /
                                 coverage pattern has no table row (synthetic, other draws)
    per_method.csv               one row per (star, method)
    per_candidate.csv            one row per (star, candidate tested by the comb
                                 fit): fitted period and its outcome, every
                                 gate's value/threshold/pass, failed_gates (T16b);
                                 (v5) beyond_reach=True rows for candidates skipped as
                                 beyond the searchable range (not fit: P_fit/outcome empty),
                                 and P0_outcome (outcome of the raw P0) on every row
    metrics.json                 summary metrics (what check_goals reads)
    fig_final_outcomes_by_bin.png
    fig_method_recall_by_bin.png
    fig_measured_vs_true.png
    fig_completeness_map.png     (v4) success vs P/R x flux amplitude x coverage class

Usage
-----
    python benchmarks/scorecard.py --synthetic 200 --seed 0 --outdir runs/synth200_baseline
    python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev --outdir runs/dev_small
    python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev \
        --ignore-manifest-sectors --outdir runs/dev_small_fullyear
    python benchmarks/scorecard.py --manifest ... --noise-subset quiet_strict   # only stars on quiet noise sources (v5)
    python benchmarks/scorecard.py ... --resume       # skip stars already in rows.jsonl
    python benchmarks/scorecard.py ... --n-workers 1  # serial, for debugging
    python benchmarks/scorecard.py --synthetic 100 --coverage single --outdir runs/synth100_single
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
import comb_fit  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402

SCORECARD_VERSION = 6  # v6 (2026-11-10, T26/P26): per-star detectability/detectable columns + final.detectable.*, final.undetectable.*, final.detectable_searchable.*; earlier keys unchanged. v5 v5 (2026-11-10): noise-source columns/breakdowns (T24, T25), beyond_reach candidate rows (P22b); v4: v2 (2026-10-13): diffrot outcome; recall/union_recall count success
# v3 (2026-10-14, T16a): --coverage dataset option + per-star coverage descriptor
# columns. Metric definitions and outcome logic are unchanged from v2, and
# --coverage full (default) reproduces v2 per-star results exactly.
# (2026-10-15, T2: dataset options --data-root and --ignore-manifest-sectors;
# metric definitions unchanged, so no version bump.)
# (2026-10-21, T16b: descriptor columns decline_kind, decline_gates,
# acf_lag_max, n_fit_candidates, n_success_candidates, best_success_P,
# best_success_failed_gates, and per_candidate.csv; metric definitions and
# outcome logic unchanged, so no version bump.)
# v4 (2026-10-22, T22 + T23): new per-star columns (acf_reach, P_over_reach,
# searchable, band_width, band_width_class, flux_amp) and new metric groups
# final.by_band_width, final.by_coverage_class, final.searchable.*,
# final.unsearchable.*, plus the completeness-map figure. Every v3 metric key
# and the outcome logic are unchanged (bit-identical on the same data); old
# baselines simply lack the new keys, so check_goals reports but cannot guard them.
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
    if task["kind"] == "wg6r":
        from wg6 import load_wg6_roman
        lc = load_wg6_roman(task["path"], task["member"], band="F146", season=task["season"],
                            true_period=task["true_period"])
        return task["star_id"], lc.time, lc.flux, float(task["true_period"]), None, dict(task["meta"]), lc.sectors_used
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


# Band-width classes (T22): w = (P_hi - P_lo) / P_eq, the fractional spread of the
# success range from differential rotation. Cuts: 0 = solid body (or shear 0);
# 0.2 ~ twice REL_TOL (the band is then comparable to the tolerance window
# already, so success vs correct barely differ); 0.5 = band wide enough that
# diffrot answers span a factor ~1.5, where `correct` (P_eq only) and `success`
# should clearly diverge. Keys avoid '.' so check_goals dotted paths work.
BAND_WIDTH_CLASSES = ("zero", "lt_0p2", "0p2_to_0p5", "ge_0p5", "unknown")


def band_width_class(true_p, prange) -> tuple:
    """(w, class) with w = (P_hi - P_lo) / P_eq; class in BAND_WIDTH_CLASSES.

    'unknown' when the star has no active-band range; w is inf for an
    unbounded band (extreme shear), which falls in 'ge_0p5'.
    """
    if prange is None or _missing(prange[0]):
        return None, "unknown"
    lo, hi = float(prange[0]), float(prange[1])
    w = (hi - lo) / float(true_p) if np.isfinite(hi) else float("inf")
    if w <= 1e-9:
        return w, "zero"
    return w, "lt_0p2" if w < 0.2 else ("0p2_to_0p5" if w < 0.5 else "ge_0p5")


def _searchability_columns(t, f, true_p, prange) -> dict:
    """Per-star v4 descriptor columns: frozen ACF reach, searchability, band width, flux amplitude.

    P_lo is the low end of the success range (P_eq when there is no band).
    These feed the new metric groups only; no v3 metric or outcome uses them.
    """
    reach = ev.acf_reach(t, f)
    p_lo = prange[0] if (prange is not None and not _missing(prange[0])) else true_p
    w, wclass = band_width_class(true_p, prange)
    ok = np.isfinite(f)
    amp = float(np.percentile(f[ok], 95) - np.percentile(f[ok], 5)) if ok.sum() > 10 else None
    return dict(acf_reach=reach, P_over_reach=float(true_p) / reach if np.isfinite(reach) else None,
                searchable=ev.is_searchable(p_lo, reach), band_width=w, band_width_class=wclass, flux_amp=amp)


def _candidate_rows(res, acf_lags, acf_counts, true_p, rel_tol, prange) -> list:
    """Per-candidate gate records (T16b): one JSON-able dict per tested candidate.

    Descriptors only (no metric depends on them). `outcome` scores the
    candidate's *fitted* period with evaluation.classify_period, i.e. what
    the pipeline would report if it selected that candidate.
    `acf_pair_frac_at_P` is the ACF's valid-pair count at lag P_fit
    relative to lag 0 (how much of the light curve overlaps itself at one
    period), a coverage-dependent quantity some gate statistics scale with.
    """
    rows = []
    selected = res.candidates[0] if (res.success and res.candidates) else None
    c0 = max(float(acf_counts[0]), 1.0)
    for r in list(res.candidates) + list(res.failed_candidates):
        row = dict(P0=float(r.period), method=r.source_guess.method,
                   contributing_methods="+".join(r.contributing_methods),
                   n_duplicate_guesses=int(r.n_duplicate_guesses),
                   n_teeth_available=int(r.n_teeth_available), n_peaks_required=int(r.n_peaks_required),
                   few_peaks=bool(r.few_peaks), passed_base_gates=bool(r.passed_base_gates),
                   passed_gates=bool(r.passed_gates), selected=r is selected, error=r.error,
                   failed_gates=";".join(r.failed_gates), beyond_reach=False,
                   P0_outcome=ev.classify_period(float(r.period), true_p, rel_tol, prange))
        if r.fit is not None:
            Pf = float(r.fit.P)
            row.update(P_fit=Pf, redchi=float(r.fit.redchi),
                       outcome=ev.classify_period(Pf, true_p, rel_tol, prange),
                       acf_pair_frac_at_P=float(np.interp(Pf, acf_lags, acf_counts) / c0))
            for g, d in (r.gate_results or {}).items():
                row[f"{g}_value"] = d["value"]
                row[f"{g}_pass"] = bool(d["passed"])
                row[f"{g}_threshold"] = d["threshold"]
                row[f"{g}_enabled"] = bool(d["enabled"])
            if r.gate_results:
                row["half_lag_contrast"] = r.gate_results["half_lag_consistency"]["contrast"]
        else:
            row.update(P_fit=None, redchi=None, outcome=None, acf_pair_frac_at_P=None)
        rows.append(row)
    # P22(b), v5: candidates skipped as beyond the searchable range (never fit). Appended after
    # every v4 row, so existing rows and their cand_rank are unchanged.
    for r in getattr(res, "beyond_reach_candidates", None) or []:
        rows.append(dict(P0=float(r.period), method=r.source_guess.method,
                         contributing_methods="+".join(r.contributing_methods),
                         n_duplicate_guesses=int(r.n_duplicate_guesses),
                         n_teeth_available=None, n_peaks_required=None, few_peaks=None,
                         passed_base_gates=False, passed_gates=False, selected=False, error=None,
                         failed_gates="", beyond_reach=True,
                         P0_outcome=ev.classify_period(float(r.period), true_p, rel_tol, prange),
                         P_fit=None, redchi=None, outcome=None, acf_pair_frac_at_P=None))
    return rows


def _gate_descriptors(res, acf_lags, acf_counts, true_p, rel_tol, prange) -> dict:
    """Per-star gate descriptor columns (T16b) plus the per-candidate records.

    decline_kind / decline_gates: EnsembleResult.decline_reason (empty on
    success). n_success_candidates: fitted candidates whose fitted period is
    a success (correct/diffrot). best_success_P / best_success_failed_gates:
    the lowest-redchi such candidate and the gates it failed ('' if it
    passed every gate). Descriptor columns only: no metric or outcome uses
    them.
    """
    cands = _candidate_rows(res, acf_lags, acf_counts, true_p, rel_tol, prange)
    reason = res.decline_reason or {}
    succ = [c for c in cands if c["outcome"] in ev.SUCCESS_OUTCOMES]
    best_succ = min(succ, key=lambda c: c["redchi"] if np.isfinite(c["redchi"]) else np.inf) if succ else None
    return dict(
        decline_kind=reason.get("kind", ""), decline_gates=";".join(reason.get("gates", [])),
        acf_lag_max=float(acf_lags[-1]),
        n_fit_candidates=len(res.candidates), n_success_candidates=len(succ),
        best_success_P=best_succ["P_fit"] if best_succ else None,
        best_success_failed_gates=best_succ["failed_gates"] if best_succ else None,
        candidates=cands,
    )


NOISE_TABLE = ROOT / "benchmarks" / "noise_sources_dev.csv"
NOISE_SUBSETS = ("all", "quiet", "quiet_strict")


def load_noise_table(path=None) -> dict:
    """{sim_id: noise row dict} from the committed table (tools/build_noise_table.py); {} if absent."""
    path = Path(path or NOISE_TABLE)
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    return {int(r["sim_id"]): r for r in t.to_dict("records")}


def in_noise_subset(noise: dict | None, subset: str) -> bool:
    """Is a star (its `noise` task dict) in the named subset: 'quiet' = noise_class quiet, not periodic,
    not excluded; 'quiet_strict' = quiet and not a coherent-quiet source (T25 / review of T20)."""
    if subset == "all":
        return True
    if not noise:
        return False
    quiet = noise["noise_class"] == "quiet" and not noise["noise_periodic"] and not noise["noise_exclude"]
    return quiet and (subset == "quiet" or not noise["noise_quiet_coherent"])


def _noise_columns(noise: dict | None, P, outcome: str, rel_tol: float) -> dict:
    """Per-star v5 noise descriptor columns (T24, T25). {} if the star has no noise row.

    noise_period_match: the star's final answer is `wrong` AND within rel_tol of the noise source's P_ref.
    noise_alias: `wrong`, not a match, but within rel_tol of P_ref/2 or 2*P_ref ('half'/'double'; diagnostic).
    Neither changes final_outcome, and neither is ever a success.
    """
    if not noise:
        return {}
    pref = noise["noise_P_ref"]
    wrong = outcome == ev.WRONG
    return dict(noise_tic=int(noise["tic"]), noise_class=noise["noise_class"],
                noise_periodic=bool(noise["noise_periodic"]), noise_exclude=bool(noise["noise_exclude"]),
                noise_quiet_coherent=bool(noise["noise_quiet_coherent"]),
                noise_P_ref=None if _missing(pref) else float(pref),
                noise_period_match=bool(wrong and ev.noise_period_match(P, pref, rel_tol)),
                noise_alias=(ev.noise_alias_kind(P, pref, rel_tol) or "") if wrong else "")


DETECTABILITY_TABLE = ROOT / "benchmarks" / "detectability_dev.csv"
DETECTABLE_THRESHOLD = 5.0  # T26 proposal: keeps 97% of dev_medium successes; below it success < 2%


def load_detectability_table(path=None) -> pd.DataFrame | None:
    """The committed per-star detectability table (tools/build_detectability_table.py); None if absent."""
    path = Path(path or DETECTABILITY_TABLE)
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["star_id", "sectors", "snr_rot"], dtype={"sectors": str})


def add_detectability_columns(per_star: pd.DataFrame, table: pd.DataFrame | None,
                              threshold: float = DETECTABLE_THRESHOLD) -> pd.DataFrame:
    """Append v6 `detectability` and `detectable` columns, joined on (star_id, loaded `sectors`).

    Descriptor only: no outcome or earlier metric uses them. A star whose loaded coverage pattern has
    no table row (synthetic stars, coverage draws the table wasn't built for) gets NaN / None, and is
    left out of the detectable/undetectable subsets. Returns per_star unchanged if there is no table
    or no `sectors` column.
    """
    if table is None or "sectors" not in per_star.columns or not len(per_star):
        return per_star
    key = per_star[["star_id", "sectors"]].astype({"sectors": str})
    snr = key.merge(table.drop_duplicates(["star_id", "sectors"]), on=["star_id", "sectors"], how="left")["snr_rot"]
    out = per_star.copy()
    out["detectability"] = snr.to_numpy()
    out["detectable"] = [None if pd.isna(v) else bool(v >= threshold) for v in snr]
    return out


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
        rec.update(_searchability_columns(t, f, true_p, prange))
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
        fit_seconds = time.perf_counter() - t0
        P = float(res.best_fit.P) if res.success else None
        rec.update(final_period=P, final_outcome=ev.classify_period(P, true_p, rel_tol, prange),
                   pipeline_success=bool(res.success), pipeline_message=res.message[:300],
                   n_pooled=len(pooled), fit_seconds=fit_seconds, error=None)
        rec.update(_gate_descriptors(res, lags, counts, true_p, rel_tol, prange))
        rec.update(_noise_columns(task.get("noise"), P, rec["final_outcome"], rel_tol))
    except Exception as exc:  # noqa: BLE001
        rec.update(final_period=None, final_outcome=ev.NONE, pipeline_success=False,
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
    rec["seconds"] = time.perf_counter() - t_start
    return rec


def _coverage_args(args) -> tuple:
    """(spec, seed, weighting, library_path) from CLI args, with defaults for callers lacking them."""
    return (getattr(args, "coverage", cov.FULL), getattr(args, "coverage_seed", 0),
            getattr(args, "coverage_weighting", "frequency"),
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
    if not args.synthetic and parse_wg6_dataset(args.manifest) is not None:
        return base  # WG6 dataset strings are self-describing (set, noise variant, season); data root is $WG6_RESAMPLED_ROOT
    if not args.synthetic:
        given, _ = manifest_data_root(args)
        if given:
            base += f",data_root={given}"
        if _manifest_has_sectors(args.manifest):
            base += ",sectors=" + ("ignored" if getattr(args, "ignore_manifest_sectors", False) else "manifest")
    if getattr(args, "noise_subset", "all") != "all":
        base += f",noise_subset={args.noise_subset}"
    spec, seed, weighting, lib_path = _coverage_args(args)
    if spec == cov.FULL:
        return base
    extra = f",coverage={spec}"
    if spec in cov.COVERAGE_CLASSES or spec in cov.CLASS_GROUPS:
        extra += f",coverage_weighting={weighting}"  # always explicit: the default changed on 2026-10-15
    if not spec.startswith("pattern:"):
        lib_name = Path(lib_path).stem.removesuffix("_patterns")
        extra += f",coverage_seed={seed},coverage_library={lib_name}"
    return base + extra


WG6_DATASET_RE = re.compile(r"^wg6r_(fl|contam|beyond)_v0_improved_s([1-3])$")
WG6_MANIFEST = ROOT / "datasets" / "wg6_roman_v0.csv"


def parse_wg6_dataset(name):
    """Parse a WG6 Roman dataset string -> (set, season), or None if `name` isn't one.

    ``wg6r_fl_v0_improved_s<k>``: dev FL rotators (reachable: P_true <= R/2), scored against the template rotation period.
    ``wg6r_contam_v0_improved_s<k>``: dev non-rotators with a catalogue period and ``beyond_reach`` False (ECL, ELL, T2CEP,
    RRLYR, DSCT, ...), scored against ``P_phot_expected`` (P_orb/2 for ECL/ELL). **On this set "correct" means the pipeline
    locked onto the variable's own period: a false positive for rotation, to be reported as a per-class FP rate, not a
    success.** For ECL/ELL the full orbital period is also an acceptable, EB-flagged lock: against P_orb/2 it shows up as
    the ``double`` alias, so ``double`` on ECL/ELL = P_orb lock. (evaluation.py is unchanged.) CV and FL-flat have no
    period and are left out.
    ``wg6r_beyond_v0_improved_s<k>``: separate stress set of every dev star with ``beyond_reach`` (P_true > R/2 = 35.5 d: all
    HB and LPV, ~1/3 of ELL, the 89 d FL template); cannot be recovered from one season, so read it as a decline/FP stress
    test. Same true-period rule as above (FL rotators against P_true).
    Season k is 1-3 (the early GBTDS seasons, assumption in claude_files/wg6.py). The holdout is never selected.
    """
    m = WG6_DATASET_RE.match(str(name))
    return (m.group(1), int(m.group(2))) if m else None


def build_wg6_tasks(name) -> list:
    """Task dicts for a WG6 dataset string (see `parse_wg6_dataset`), read from datasets/wg6_roman_v0.csv.

    Zips are read in place under $WG6_RESAMPLED_ROOT (default: the Data_Vault drive path); no data root flag needed.
    """
    from wg6 import resampled_root
    which, season = parse_wg6_dataset(name)
    man = pd.read_csv(WG6_MANIFEST, low_memory=False)
    man = man[(man.eval_set == "dev") & (man.season == season)]
    br = man.beyond_reach.astype(bool)
    rot = man.is_rotator.astype(bool)
    if which == "fl":
        man = man[(man.cls == "FL") & rot & ~br]
    elif which == "contam":
        man = man[~rot & ~br & man.P_phot_expected.notna()]
    else:  # beyond
        man = man[br & man.P_phot_expected.notna()]
    man = man.assign(p_score=np.where(man.is_rotator.astype(bool), man.P_true, man.P_phot_expected))
    root = resampled_root()
    return [dict(kind="wg6r", path=str(root / r.zip), member=r.member, season=int(r.season), star_id=str(r.star_id),
                 true_period=float(r.p_score), load_kwargs={},
                 meta=dict(cls=r.cls, is_eb=bool(r.is_eb), is_rotator=bool(r.is_rotator), wfmag=r.wfmag,
                           fl_template=r.fl_template, P_true=r.P_true, season=int(r.season), path=str(root / r.zip)))
            for r in man.itertuples()]


def build_tasks(args) -> list:
    """Translate CLI dataset arguments into a list of task dicts."""
    if args.synthetic:
        from synthetic import make_suite
        pats = sample_coverage(args, args.synthetic)
        sectors = None if all(p is None for p in pats) else pats
        return [dict(kind="synthetic", lc=lc, star_id=lc.star_id)
                for lc in make_suite(args.synthetic, seed=args.seed, sectors=sectors)]
    if parse_wg6_dataset(args.manifest) is not None:
        return build_wg6_tasks(args.manifest)
    man = pd.read_csv(args.manifest)
    if "path" not in man.columns:
        raise SystemExit(f"{args.manifest} needs a 'path' column")
    pats = sample_coverage(args, len(man))
    if "sectors" in man.columns and not getattr(args, "ignore_manifest_sectors", False):
        if _coverage_args(args)[0] != cov.FULL:
            raise SystemExit(f"{args.manifest} has a 'sectors' column; don't also pass --coverage")
        pats = [None if pd.isna(s) or str(s).strip() == "" else cov.parse_pattern(s) for s in man["sectors"]]
    _, root = manifest_data_root(args)
    noise_tab = load_noise_table(getattr(args, "noise_table", None)) if "sim_id" in man.columns else {}
    subset = getattr(args, "noise_subset", "all")
    if subset != "all" and not noise_tab:
        raise SystemExit("--noise-subset needs the noise table (benchmarks/noise_sources_dev.csv) and a manifest "
                         "with a sim_id column; the holdout has none by design.")
    tasks = []
    for (_, r), pat in zip(man.iterrows(), pats):
        noise = noise_tab.get(int(r["sim_id"])) if noise_tab else None
        if not in_noise_subset(noise, subset):
            continue
        tasks.append(dict(noise=noise, kind="fits", path=resolve_manifest_path(r["path"], root),
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


def flatten_candidates(records: list) -> pd.DataFrame:
    """One row per (star, tested candidate) from the records' `candidates` lists (T16b)."""
    rows = []
    for r in records:
        for i, c in enumerate(r.get("candidates") or []):
            rows.append(dict(star_id=r["star_id"], true_period=r.get("true_period"),
                             period_bin=r.get("period_bin"), coverage_class=r.get("coverage_class"),
                             cand_rank=i, **c))
    return pd.DataFrame(rows)


def flatten(records: list) -> tuple:
    """Turn JSON records into (per_star, per_method) DataFrames."""
    star_rows, meth_rows = [], []
    for r in records:
        star_rows.append({k: v for k, v in r.items() if k not in ("methods", "meta", "traceback", "candidates")}
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
    m["final"].update(_v4_final_breakdowns(per_star))
    m["final"].update(_v5_noise_breakdowns(per_star))
    m["final"].update(_v6_detectability_breakdowns(per_star))
    m["runtime"] = {"median_seconds": float(per_star["seconds"].median()),
                    "p90_seconds": float(per_star["seconds"].quantile(0.9))}
    m["errors"]["n_star_errors"] = int(per_star["error"].notna().sum())
    return m


def _breakdown(df: pd.DataFrame, col: str) -> dict:
    """{value of `col`: outcome_fractions of final_outcome} over the non-null rows of `col`.

    The one place a new subset breakdown (e.g. T25's quiet-subset metrics) needs
    to hook in: add a column in score_one_star and call this on it.
    """
    if col not in df.columns:
        return {}
    d = df[df[col].notna()]
    return {str(k): ev.outcome_fractions(g["final_outcome"]) for k, g in d.groupby(col)}


def _subset_block(df: pd.DataFrame) -> dict:
    """overall + by_bin + by_coverage_class outcome fractions for a subset of stars."""
    return {"overall": ev.outcome_fractions(df["final_outcome"]),
            "by_bin": _breakdown(df, "period_bin"),
            "by_coverage_class": _breakdown(df, "coverage_class")}


def _v4_final_breakdowns(per_star: pd.DataFrame) -> dict:
    """The scorecard-v4 additions to `final` (T22, T23); {} keys are omitted if the columns are absent.

    final.by_band_width.<class>, final.by_coverage_class.<class>,
    final.searchable.{overall,by_bin,by_coverage_class},
    final.unsearchable.{overall,by_bin,by_coverage_class}
    (overall.none = correct-decline rate, .wrong = wrong answers, .success = bonus successes).
    """
    out: dict = {}
    if "band_width_class" in per_star.columns:
        out["by_band_width"] = _breakdown(per_star, "band_width_class")
    if "coverage_class" in per_star.columns:
        out["by_coverage_class"] = _breakdown(per_star, "coverage_class")
    if "searchable" in per_star.columns:
        known = per_star[per_star["searchable"].notna()]
        flag = known["searchable"].astype(bool)
        out["searchable"] = _subset_block(known[flag])
        out["unsearchable"] = _subset_block(known[~flag])
    return out


def _noise_block(df: pd.DataFrame) -> dict:
    """n, wrong split into wrong_noise_period / wrong_other (fractions of n) and noise_alias (count, a
    diagnostic subset of wrong_other: half/double of the noise period) for a set of stars."""
    n = len(df)
    wrong = df["final_outcome"] == ev.WRONG
    npm = df["noise_period_match"].fillna(False).astype(bool)
    alias = df["noise_alias"].fillna("").astype(str) != ""
    return {"n": n, "wrong": float(wrong.mean()) if n else 0.0,
            "wrong_noise_period": float((wrong & npm).mean()) if n else 0.0,
            "wrong_other": float((wrong & ~npm).mean()) if n else 0.0,
            "n_wrong": int(wrong.sum()), "n_wrong_noise_period": int((wrong & npm).sum()),
            "n_wrong_other": int((wrong & ~npm).sum()), "noise_alias": int((wrong & alias).sum())}


def _v5_noise_breakdowns(per_star: pd.DataFrame) -> dict:
    """The scorecard-v5 additions to `final` (T24, T25); {} if the per-star table has no noise columns.

    final.noise.{overall,by_bin,by_noise_class}: _noise_block (the wrong split; `wrong` itself is
        unchanged in final.overall).
    final.quiet_noise / final.quiet_noise_strict / final.periodic_noise: _subset_block plus a `noise`
        block, over stars on quiet noise sources / quiet minus coherent-quiet / non-quiet (noise_class
        != quiet or periodic or excluded).
    """
    if "noise_class" not in per_star.columns:
        return {}
    known = per_star[per_star["noise_class"].notna()]
    quiet = known["noise_class"].eq("quiet") & ~known["noise_periodic"].astype(bool) & ~known["noise_exclude"].astype(bool)
    strict = quiet & ~known["noise_quiet_coherent"].astype(bool)
    out = {"noise": {"overall": _noise_block(known),
                     "by_bin": {str(b): _noise_block(g) for b, g in known.groupby("period_bin")},
                     "by_noise_class": {str(c): _noise_block(g) for c, g in known.groupby("noise_class")}}}
    for name, mask in (("quiet_noise", quiet), ("quiet_noise_strict", strict), ("periodic_noise", ~quiet)):
        sub = known[mask]
        if len(sub):
            out[name] = _subset_block(sub) | {"noise": _noise_block(sub)}
    return out


def _v6_detectability_breakdowns(per_star: pd.DataFrame) -> dict:
    """The scorecard-v6 additions to `final` (T26/P26); {} if no star has a detectability value.

    final.detectable / final.undetectable / final.detectable_searchable: _subset_block over stars with
    a known `detectable` flag (detectable_searchable also needs searchable == True).
    """
    if "detectable" not in per_star.columns:
        return {}
    known = per_star[per_star["detectable"].notna()]
    if not len(known):
        return {}
    flag = known["detectable"].astype(bool)
    out = {"detectable": _subset_block(known[flag]), "undetectable": _subset_block(known[~flag])}
    if "searchable" in known.columns:
        both = flag & known["searchable"].fillna(False).astype(bool)
        out["detectable_searchable"] = _subset_block(known[both])
    return out


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


def make_completeness_map(per_star: pd.DataFrame, outdir: Path, title: str):
    """Completeness map (T23): final success rate vs P_eq/R, by flux amplitude and coverage class.

    Top: success rate vs P/R per coverage class (the searchable edge is the
    dashed line at P/R = 0.5 for non-diffrot stars). Bottom: success-rate
    heatmap over P/R x flux amplitude (amplitude terciles), all coverage
    classes pooled; cell labels give n. Skipped if the v4 columns are absent.
    """
    need = {"P_over_reach", "flux_amp", "final_outcome"}
    if not need.issubset(per_star.columns):
        return
    d = per_star.dropna(subset=["P_over_reach", "flux_amp"]).copy()
    if len(d) < 10:
        return
    d["success"] = d["final_outcome"].isin(ev.SUCCESS_OUTCOMES)
    edges = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5, np.inf])
    labels = [f"{a:g}-{b:g}" if np.isfinite(b) else f">{a:g}" for a, b in zip(edges[:-1], edges[1:])]
    d["x"] = pd.cut(d["P_over_reach"], edges, labels=False, right=False)
    classes = [c for c in cov.COVERAGE_CLASSES if "coverage_class" in d and (d["coverage_class"] == c).any()] or [None]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8.5))
    xs = np.arange(len(labels))
    for c in classes:
        g = d if c is None else d[d.coverage_class == c]
        rate = g.groupby("x")["success"].agg(["mean", "size"]).reindex(range(len(labels)))
        rate.loc[rate["size"] < 3, "mean"] = np.nan
        ax0.plot(xs, rate["mean"], marker="o", ms=4, label=f"{c or 'all'} (n={len(g)})")
    ax0.axvline(4.5, color="k", ls="--", lw=0.8)
    ax0.text(4.55, 0.02, "P/R = 0.5", fontsize=8)
    ax0.set_xticks(xs, labels, rotation=45, ha="right")
    ax0.set_xlabel("P_eq / R  (frozen ACF reach)")
    ax0.set_ylabel("final success rate (bins with n<3 hidden)")
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax0.set_title(f"Completeness vs P/R -- {title}")
    q = d["flux_amp"].quantile([1 / 3, 2 / 3]).values
    d["a"] = np.digitize(d["flux_amp"], q)
    grid = np.full((3, len(labels)), np.nan)
    cnt = np.zeros((3, len(labels)), dtype=int)
    for (a, x), g in d.groupby(["a", "x"]):
        cnt[int(a), int(x)] = len(g)
        if len(g) >= 3:
            grid[int(a), int(x)] = g["success"].mean()
    im = ax1.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
    for a in range(3):
        for x in range(len(labels)):
            if cnt[a, x]:
                v = grid[a, x]
                ax1.text(x, a, f"n={cnt[a, x]}", ha="center", va="center", fontsize=7,
                         color="0.5" if not np.isfinite(v) else ("white" if v < 0.6 else "black"))
    ax1.set_xticks(xs, labels, rotation=45, ha="right")
    ax1.set_yticks(range(3), [f"amp tercile {i + 1}\n(<{q[0]:.3g})" if i == 0 else
                              (f"amp tercile 3\n(>{q[1]:.3g})" if i == 2 else "amp tercile 2") for i in range(3)])
    ax1.set_xlabel("P_eq / R")
    fig.colorbar(im, ax=ax1, label="final success rate")
    fig.tight_layout()
    fig.savefig(outdir / "fig_completeness_map.png", dpi=130)
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
    p.add_argument("--coverage-weighting", choices=("frequency", "uniform"), default="frequency",
                   help="for a class --coverage: patterns drawn by real-star count (frequency, default) "
                        "or equally likely (uniform)")
    p.add_argument("--coverage-library", type=Path, default=None,
                   help="coverage library CSV (default datasets/coverage/all_cycles_patterns.csv)")
    p.add_argument("--data-root", default=None,
                   help="root for relative manifest paths (default $SMARTS_DATA_ROOT, else cwd); "
                        "the T2 dev sets need data/smarts_dev")
    p.add_argument("--ignore-manifest-sectors", action="store_true",
                   help="ignore the manifest's `sectors` column (score every native sector)")
    p.add_argument("--noise-subset", choices=NOISE_SUBSETS, default="all",
                   help="score only stars on quiet noise sources: 'quiet' (T20 class quiet) or 'quiet_strict' "
                        "(also drops the ~34 coherent-quiet sources); default all. Metrics for both subsets "
                        "are always reported (final.quiet_noise*, v5) whatever this is. Changes config.dataset.")
    p.add_argument("--noise-table", type=Path, default=None,
                   help="per-star noise table (default benchmarks/noise_sources_dev.csv)")
    p.add_argument("--detectability-table", type=Path, default=None,
                   help="per-star detectability table (default benchmarks/detectability_dev.csv; v6)")
    p.add_argument("--detectable-threshold", type=float, default=DETECTABLE_THRESHOLD,
                   help=f"detectable = detectability >= this (default {DETECTABLE_THRESHOLD:g}; T26)")
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
    missing = [t["path"] for t in tasks if t["kind"] in ("fits", "wg6r") and not Path(t["path"]).exists()]
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
    per_star = add_detectability_columns(per_star, load_detectability_table(args.detectability_table),
                                         args.detectable_threshold)
    per_star.to_csv(outdir / "per_star.csv", index=False)
    per_method.to_csv(outdir / "per_method.csv", index=False)
    flatten_candidates(list(done.values())).to_csv(outdir / "per_candidate.csv", index=False)
    metrics = compute_metrics(per_star, per_method, args.rel_tol)
    metrics["config"] = dict(
        scorecard_version=SCORECARD_VERSION, git_commit=_git_commit(), rel_tol=args.rel_tol,
        dataset=dataset_string(args),
        coverage=dict(spec=args.coverage, seed=args.coverage_seed, weighting=args.coverage_weighting,
                      library=str(args.coverage_library or cov.DEFAULT_LIBRARY.relative_to(ROOT))),
        data_root=manifest_data_root(args)[0] if args.manifest else None,
        ignore_manifest_sectors=bool(args.ignore_manifest_sectors),
        detectable_threshold=args.detectable_threshold,
        detectability_table=str(args.detectability_table or DETECTABILITY_TABLE.relative_to(ROOT)),
        pipeline_methods=list(methods), n_guesses_default=N_GUESSES_DEFAULT, n_guesses_overrides=N_GUESSES,
        # T31: speed variants in effect (env-selected; defaults = the unchanged methods)
        comb_variant=comb_fit.resolve_comb_variant()[0],
        cwt_variant=guesses.resolve_cwt_variant()[0],
        wall_seconds=time.time() - t0, created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    make_figures(per_star, per_method, outdir, args.label or outdir.name)
    make_completeness_map(per_star, outdir, args.label or outdir.name)

    f = metrics["final"]["overall"]
    print(f"\nFinal answer (n={f['n']}): success {f['success']:.1%} (correct {f['correct']:.1%}, "
          f"diffrot {f['diffrot']:.1%})  alias {f['alias']:.1%} "
          f"(half {f['half']:.1%}, double {f['double']:.1%})  wrong {f['wrong']:.1%}  none {f['none']:.1%}")
    print(f"Wrote {outdir}/metrics.json and figures.")


if __name__ == "__main__":
    main()
