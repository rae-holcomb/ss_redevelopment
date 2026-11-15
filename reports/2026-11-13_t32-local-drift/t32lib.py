"""T32 helpers: a fast production-path harness and per-peak drift diagnostics.

The harness runs exactly the scorecard's Stage 1 (the three pooled methods,
two ACF views) and Stage 2 (`fit_rotation_period`) on every star of a
manifest, but skips the non-pooled methods (wavelet etc.), which never touch
the final answer. Pooled Stage-1 guesses are cached per mode, so scoring a
comb variant costs only load + ACF + fit. Final outcomes are reproduced
exactly (checked against the scorecard's per_star.csv; see README).

Used for screening the T32 variants; official numbers come from
`benchmarks/scorecard.py` runs.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for p in ("claude_files", "benchmarks"):
    sys.path.insert(0, str(ROOT / p))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import scorecard as sc  # noqa: E402
import comb_fit as C  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from guesses import two_view_guesses  # noqa: E402


def make_tasks(mode: str, manifest: str = "datasets/dev_medium.csv", data_root: str = "data/smarts_dev") -> list:
    """Scorecard task dicts for `manifest` in mode 'fullyear' (ignore manifest sectors) or 'assigned'."""
    args = SimpleNamespace(synthetic=None, manifest=ROOT / manifest, ignore_manifest_sectors=(mode == "fullyear"),
                           coverage=cov.FULL, coverage_seed=0, coverage_weighting="frequency",
                           coverage_library=None, data_root=str(ROOT / data_root), noise_table=None,
                           noise_subset="all")
    return sc.build_tasks(args)


def load_star(task: dict) -> dict:
    """Load one star and compute both ACF views (the scorecard's steps)."""
    star_id, t, f, true_p, prange, meta, sectors = sc._load(task)
    lags, acf, counts = compute_acf(t, f, return_counts=True)
    lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    return dict(star_id=star_id, t=t, f=f, true_p=true_p, prange=prange, lags=lags, acf=acf, counts=counts,
                lags1=lags1, acf1=acf1)


def pooled_guesses(st: dict) -> list:
    """The pooled Stage-1 candidates (scorecard's DEFAULT_PIPELINE_METHODS, same order)."""
    fns = sc.discover_guess_functions()
    pooled = []
    for name, fn in fns.items():
        if name not in sc.DEFAULT_PIPELINE_METHODS:
            continue
        try:
            pooled.extend(two_view_guesses(fn, st["t"], st["f"], (st["lags1"], st["acf1"]), (st["lags"], st["acf"]),
                                           n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT)))
        except Exception:  # noqa: BLE001 -- the scorecard records and continues too
            pass
    return pooled


def candidate_rows(res, true_p, prange) -> list:
    """Per-candidate summary rows (gate values, pass flags, outcome, drift coefficient)."""
    rows = []
    for rank, r in enumerate(res.candidates):
        g = r.gate_results or {}
        fit = r.fit
        rows.append(dict(
            rank=rank, P0=r.period, P=fit.P, passed=bool(r.passed_gates), selected=bool(res.success and rank == 0),
            outcome=ev.classify_period(fit.P, true_p, ev.REL_TOL, prange),
            failed=";".join(r.failed_gates), redchi=fit.redchi, n_peaks_used=fit.n_peaks_used,
            n_required=r.n_peaks_required,
            **{f"v_{k}": v.get("value") for k, v in g.items()},
            drift_a=(r.diagnostics or {}).get("drift_a", np.nan),
            drift_a_frac=(r.diagnostics or {}).get("drift_a_frac", np.nan),
            vertex_scatter=(r.diagnostics or {}).get("vertex_scatter", np.nan),
        ))
    return rows


def run_star(task: dict, variants: tuple, cache_dir: str, keep_candidates: bool = True) -> dict:
    """Score one star under every comb variant in `variants` (module-level, picklable).

    Pooled guesses are read from / written to `cache_dir/<star_id>.pkl`.
    Returns {variant: dict(final_period, final_outcome, decline_gates, fit_seconds, candidates)}.
    """
    warnings.simplefilter("ignore")
    st = load_star(task)
    cp = Path(cache_dir) / f"{st['star_id']}.pkl"
    if cp.exists():
        pooled = pickle.loads(cp.read_bytes())
    else:
        pooled = pooled_guesses(st)
        tmp = cp.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(pooled))
        os.replace(tmp, cp)
    out = dict(star_id=st["star_id"], true_period=st["true_p"], period_bin=ev.period_bin(st["true_p"]),
               period_lo=st["prange"][0] if st["prange"] else None,
               period_hi=st["prange"][1] if st["prange"] else None, variants={})
    for v in variants:
        t0 = time.perf_counter()
        try:
            res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
            secs = time.perf_counter() - t0
            P = float(res.best_fit.P) if res.success else None
            rec = dict(final_period=P, final_outcome=ev.classify_period(P, st["true_p"], ev.REL_TOL, st["prange"]),
                       decline_gates=";".join((res.decline_reason or {}).get("gates", [])), fit_seconds=secs)
            if keep_candidates:
                rec["candidates"] = candidate_rows(res, st["true_p"], st["prange"])
        except Exception as exc:  # noqa: BLE001
            rec = dict(final_period=None, final_outcome=ev.NONE, decline_gates="", fit_seconds=np.nan,
                       error=f"{type(exc).__name__}: {exc}")
        out["variants"][v] = rec
    return out


def run_harness(mode: str, variants: tuple, out_pkl: str, n_workers: int = 3, manifest="datasets/dev_medium.csv",
                star_ids=None, keep_candidates=True) -> list:
    """Run run_star over a manifest (resumable per output pickle: re-runs only if absent)."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = make_tasks(mode, manifest)
    if star_ids is not None:
        keep = set(star_ids)
        tasks = [t for t in tasks if t["star_id"] in keep]
    cache = ROOT / "runs" / "t32" / "stage1_cache" / (Path(manifest).stem + "_" + mode)
    cache.mkdir(parents=True, exist_ok=True)
    worker = partial(run_star, variants=tuple(variants), cache_dir=str(cache), keep_candidates=keep_candidates)
    if n_workers <= 1:
        recs = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=determinism.init_worker) as ex:
            recs = list(ex.map(worker, tasks, chunksize=4))
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_pkl) + ".tmp")
    tmp.write_bytes(pickle.dumps(recs))
    os.replace(tmp, out_pkl)
    return recs


def summarize(recs: list, variant: str) -> dict:
    """Overall success / wrong / none / alias rates for one variant from harness records."""
    oc = pd.Series([r["variants"][variant]["final_outcome"] for r in recs])
    n = len(oc)
    return dict(variant=variant, n=n, success=float(oc.isin(["correct", "diffrot"]).sum() / n),
                wrong=float((oc == "wrong").sum() / n), none=float((oc == "none").sum() / n),
                alias=float(oc.isin(["half", "double"]).sum() / n),
                fit_s=float(np.nanmedian([r["variants"][variant]["fit_seconds"] for r in recs])))


def per_star_frame(recs: list, variant: str) -> pd.DataFrame:
    """One row per star for one variant."""
    rows = []
    for r in recs:
        v = r["variants"][variant]
        rows.append(dict(star_id=r["star_id"], true_period=r["true_period"], period_bin=r["period_bin"],
                         final_period=v["final_period"], final_outcome=v["final_outcome"],
                         decline_gates=v.get("decline_gates", ""), fit_seconds=v["fit_seconds"]))
    return pd.DataFrame(rows)


def candidate_frame(recs: list, variant: str) -> pd.DataFrame:
    """All fitted candidates of one variant, one row each, with star_id and true period."""
    rows = []
    for r in recs:
        for c in r["variants"][variant].get("candidates", []):
            rows.append(dict(star_id=r["star_id"], true_period=r["true_period"], period_bin=r["period_bin"], **c))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-peak drift diagnostics
def tracked_peaks(lags: np.ndarray, acf: np.ndarray, P: float, t0: float, n_max: int = 8,
                  search_frac: float = 0.25, refine_frac: float = 0.1) -> list:
    """Sequentially tracked raw-ACF peak positions for comb (P, t0) (solver-independent drift probe).

    Tooth n is searched within +/- search_frac*P of (tracked tooth n-1) + P, its
    maximum refined by a parabola over +/- refine_frac*P. Returns a list of
    (n, position) for teeth inside the lag range (n = 0 is t0 itself).
    """
    out = [(0, float(t0))]
    prev = float(t0)
    for n in range(1, n_max):
        guess = prev + P
        lo, hi = guess - search_frac * P, guess + search_frac * P
        if hi > lags[-1]:
            break
        m = (lags >= lo) & (lags <= hi)
        if m.sum() < 5:
            break
        i = np.flatnonzero(m)[np.argmax(acf[m])]
        mm = np.abs(lags - lags[i]) <= refine_frac * P
        x, y = lags[mm] - lags[i], acf[mm]
        pos = lags[i]
        if len(x) >= 5:
            c = np.polyfit(x, y, 2)
            if c[0] < 0:
                xv = -c[1] / (2 * c[0])
                if abs(xv) <= refine_frac * P:
                    pos = lags[i] + xv
        out.append((n, float(pos)))
        prev = float(pos)
    return out


def peak_records(st: dict, cand, label: str) -> list:
    """Per-window rows for one fitted CandidateResult: window, tied center, free local vertex, tracked peak."""
    fit = cand.fit
    P0, t0_seed = cand.period, cand.t0
    tp = dict(tracked_peaks(st["lags"], st["acf"], fit.P, fit.t0))
    rows = []
    for w in fit.windows:
        lag, y = st["lags"][w.mask], st["acf"][w.mask]
        mid, hw = 0.5 * (w.lag_hi + w.lag_lo), max(0.5 * (w.lag_hi - w.lag_lo), 1e-12)
        z = (lag - mid) / hw
        c = np.polyfit(z, y, 2)
        free = mid + (-c[1] / (2 * c[0])) * hw if c[0] < 0 else np.nan
        pp = fit.per_peak[w.n]
        rows.append(dict(label=label, n=w.n, P0=P0, t0_seed=t0_seed, P_fit=fit.P, t0_fit=fit.t0,
                         win_center=t0_seed + w.n * P0, lag_lo=w.lag_lo, lag_hi=w.lag_hi,
                         tied_center=pp["center"], vertex_in_window=pp["vertex_in_window"], height=pp["height"],
                         free_vertex=free, tracked=tp.get(w.n, np.nan)))
    return rows
