"""P43 helpers: T32's fast harness (t32lib) with a P43-owned Stage-1 cache, plus per-tooth diagnostics.

The Stage-1 cache lives in runs/p43/stage1_cache/<manifest>_<mode> (rebuilt here, not reused from
T32: P39 changed guess_acf_fft's |c|^2 rounding, so T32's cached pooled guesses may differ slightly).
Screening only; official numbers come from benchmarks/scorecard.py.
"""
from __future__ import annotations

import os
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports" / "2026-11-13_t32-local-drift"))
import t32lib as L  # noqa: E402  (pins BLAS threads, sets sys.path)

import numpy as np  # noqa: E402
import comb_fit as C  # noqa: E402
import evaluation as ev  # noqa: E402

CACHE = ROOT / "runs" / "p43" / "stage1_cache"


def cache_dir(manifest: str, mode: str) -> Path:
    """Stage-1 cache folder for one manifest + mode."""
    d = CACHE / (Path(manifest).stem + "_" + mode)
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_harness(mode: str, variants: tuple, out_pkl: str, n_workers: int = 3, manifest="datasets/dev_medium.csv",
                star_ids=None, keep_candidates=True) -> list:
    """t32lib.run_harness with the P43 cache (resumable per output pickle)."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = L.make_tasks(mode, manifest)
    if star_ids is not None:
        keep = set(star_ids)
        tasks = [t for t in tasks if t["star_id"] in keep]
    worker = partial(L.run_star, variants=tuple(variants), cache_dir=str(cache_dir(manifest, mode)),
                     keep_candidates=keep_candidates)
    if n_workers <= 1:
        recs = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=L.determinism.init_worker) as ex:
            recs = list(ex.map(worker, tasks, chunksize=4))
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_pkl) + ".tmp")
    tmp.write_bytes(pickle.dumps(recs))
    os.replace(tmp, out_pkl)
    return recs


def star_and_guesses(star_id: str, mode: str, manifest="datasets/dev_medium.csv"):
    """(star dict from t32lib.load_star, pooled Stage-1 guesses) for one star, via the P43 cache."""
    task = [t for t in L.make_tasks(mode, manifest) if t["star_id"] == star_id][0]
    st = L.load_star(task)
    cp = cache_dir(manifest, mode) / f"{star_id}.pkl"
    if cp.exists():
        pooled = pickle.loads(cp.read_bytes())
    else:
        pooled = L.pooled_guesses(st)
        tmp = cp.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(pooled))
        os.replace(tmp, cp)
    return st, pooled


def tooth_table(cand, spec: dict) -> list:
    """Per-tooth rows for one fitted local-family CandidateResult: comb centre m_n, free vertex v_n,
    fitted height h_n (with the comb centre fixed), free height, tolerance tol_n, within-tol flag."""
    fit = cand.fit
    li = fit.local_info or {}
    rows = []
    P = float(fit.P)
    base = float(spec.get("tol_base", 0.05))
    for w in sorted(fit.windows, key=lambda w: w.n):
        n = w.n
        pk = fit.per_peak[n]
        m = li.get("model_t0", fit.t0) + li.get("model_P", P) * n + li.get("drift_a", 0.0) * n * n
        v = li.get("free_vertex", {}).get(n, np.nan)
        tol = base * P * np.sqrt(max(n, 1))
        rows.append(dict(n=n, lag_lo=w.lag_lo, lag_hi=w.lag_hi, m=m, v=v, h=pk["height"],
                         h_free=li.get("free_height", {}).get(n, np.nan), A=pk["curvature"], tol=tol,
                         lag0=bool(m < 0.5 * P), within=bool(np.isfinite(v) and abs(v - m) <= tol)))
    return rows


def classify(P, st):
    """Outcome label of period P for star dict st."""
    return ev.classify_period(P, st["true_p"], ev.REL_TOL, st["prange"])


def candidate_rows(res, st) -> list:
    """Per-candidate rows: t32lib's columns plus P43's (both frac_vertex_tol flavours, drops, drift_a_frac)."""
    rows = L.candidate_rows(res, st["true_p"], st["prange"])
    for row, r in zip(rows, res.candidates):
        fit, d = r.fit, (r.diagnostics or {})
        heights = [pk["height"] for n, pk in sorted(fit.per_peak.items())]
        pos = C._vertex_stats(fit, dict(C.COMB_VARIANTS["local_tol"], tol_count="positive")) if fit.local_info else {}
        row.update(fvt_all=d.get("frac_vertex_tol_all", np.nan), fvt_pos=pos.get("frac_vertex_tol", np.nan),
                   n_dropped=fit.n_peaks_dropped, n_trimmed=fit.n_peaks_trimmed,
                   n_nonpos=int(sum(h <= 0 for h in heights)),
                   drift_frac=C.comb_drift_frac(fit))
    return rows


def run_star(task: dict, variants: tuple, cache_dir: str) -> dict:
    """t32lib.run_star with P43's candidate rows (module-level, picklable)."""
    import time
    import warnings
    warnings.simplefilter("ignore")
    st = L.load_star(task)
    cp = Path(cache_dir) / f"{st['star_id']}.pkl"
    if cp.exists():
        pooled = pickle.loads(cp.read_bytes())
    else:
        pooled = L.pooled_guesses(st)
        tmp = cp.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(pooled))
        os.replace(tmp, cp)
    out = dict(star_id=st["star_id"], true_period=st["true_p"], period_bin=ev.period_bin(st["true_p"]),
               variants={})
    for v in variants:
        t0 = time.perf_counter()
        try:
            res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
            secs = time.perf_counter() - t0
            P = float(res.best_fit.P) if res.success else None
            out["variants"][v] = dict(final_period=P, final_outcome=classify(P, st), fit_seconds=secs,
                                      decline_gates=";".join((res.decline_reason or {}).get("gates", [])),
                                      candidates=candidate_rows(res, st))
        except Exception as exc:  # noqa: BLE001
            out["variants"][v] = dict(final_period=None, final_outcome=ev.NONE, fit_seconds=np.nan,
                                      decline_gates="", candidates=[], error=f"{type(exc).__name__}: {exc}")
    return out


def run_p43_harness(mode: str, variants: tuple, out_pkl: str, n_workers: int = 3,
                    manifest="datasets/dev_medium.csv") -> list:
    """Harness over a manifest with P43's run_star; writes out_pkl atomically."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = L.make_tasks(mode, manifest)
    worker = partial(run_star, variants=tuple(variants), cache_dir=str(cache_dir(manifest, mode)))
    if n_workers <= 1:
        recs = [worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=L.determinism.init_worker) as ex:
            recs = list(ex.map(worker, tasks, chunksize=4))
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_pkl) + ".tmp")
    tmp.write_bytes(pickle.dumps(recs))
    os.replace(tmp, out_pkl)
    return recs
