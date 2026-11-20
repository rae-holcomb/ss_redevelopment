"""P44 sweep harness: cached Stage-1 guesses + memoized comb fits, rescored at every mean_height_snr cutoff.

The comb fit of a candidate does not depend on the mean_height_snr cutoff; only the gates and the
selection do. So per star this runs Stage 1 once (cached on disk, as in P43's harness), then calls the
production `fit_rotation_period` once per (variant, cutoff) with `comb_fit._fit_single_candidate`
memoized in-process: every call after the first reuses the fits and re-runs only gating and selection.
The result is exactly what the production path gives at that cutoff (checked against the `_v8` per-star
tables at 1.0 for the old variant in sweep.py).

Only noise-clean manifests are used (p43lib.make_tasks refuses others). Tuning: dev_medium_b_clean only.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports" / "2026-11-16_p43-clean-rerun"))
import p43lib as L  # noqa: E402  (pins BLAS threads before numpy)

import numpy as np  # noqa: E402
import comb_fit as C  # noqa: E402
import evaluation as ev  # noqa: E402

CACHE_ROOT = ROOT / "runs" / "p44" / "stage1_cache"
VARIANTS = ("local_tol_trim", "local_tol_trim_l0x")
CUTOFFS = tuple(np.round(np.r_[np.arange(0.0, 0.5, 0.1), np.arange(0.5, 1.6001, 0.02),
                               np.arange(1.7, 2.51, 0.1)], 3))
RECORD_CUTOFF = {"local_tol_trim": 1.0}  # per-candidate rows recorded at this cutoff (others: P44_SNR_CUTOFF)

_ORIG_FIT = C._fit_single_candidate
_ORIG_T0 = C._grid_search_t0


def _memo_fit():
    """Install a memoizing wrapper around comb_fit._fit_single_candidate; returns its cache dict."""
    cache = {}

    def wrapped(acf_lags, acf, P0, t0, **kw):
        spec = kw.get("comb_spec") or {}
        key = (float(P0), float(t0), tuple(sorted((k, repr(v)) for k, v in spec.items())),
               tuple(sorted((k, repr(v)) for k, v in kw.items() if k not in ("comb_spec", "weights"))))
        if key not in cache:
            try:
                cache[key] = ("ok", _ORIG_FIT(acf_lags, acf, P0, t0, **kw))
            except Exception as exc:  # noqa: BLE001 -- replay the same exception
                cache[key] = ("err", exc)
        kind, val = cache[key]
        if kind == "err":
            raise val
        return val

    t0_cache = {}

    def t0_wrapped(acf_lags, acf, P0, **kw):
        key = (float(P0), tuple(sorted((k, repr(v)) for k, v in kw.items())))
        if key not in t0_cache:
            t0_cache[key] = _ORIG_T0(acf_lags, acf, P0, **kw)
        return t0_cache[key]

    C._fit_single_candidate = wrapped
    C._grid_search_t0 = t0_wrapped  # same star, same ACF: t0 seeding is cutoff-independent too
    return cache


def cand_rows(res, true_p, prange) -> list:
    """Per-candidate gate values (incl. both SNR means), failed gates and outcome."""
    rows = []
    for rank, r in enumerate(res.candidates):
        fit, d = r.fit, r.diagnostics
        snr = np.asarray(d["height_snr"], float)
        ns = sorted(fit.per_peak)
        lag0 = np.array([C._is_lag0_tooth(fit.per_peak[n]["center"], fit.P) for n in ns], bool)
        rows.append(dict(
            rank=rank, P0=r.period, P=float(fit.P), passed=bool(r.passed_gates),
            selected=bool(res.success and rank == 0),
            outcome=ev.classify_period(fit.P, true_p, ev.REL_TOL, prange),
            failed=";".join(r.failed_gates), failed_ex_snr=";".join(g for g in r.failed_gates if g != "mean_height_snr"),
            snr_gate=float(d["mean_height_snr"]), snr_all=float(np.nanmean(snr)) if snr.size else np.nan,
            snr_nolag0=float(np.nanmean(snr[~lag0])) if (~lag0).any() else np.nan,
            has_lag0=bool(lag0.any()), n_peaks_used=int(fit.n_peaks_used), n_dropped=int(fit.n_peaks_dropped),
            n_trimmed=int(fit.n_peaks_trimmed), redchi=float(fit.redchi),
        ))
    return rows


def run_star(task: dict, cache_dir: str, variants=VARIANTS, cutoffs=CUTOFFS) -> dict:
    """Outcome of one star at every (variant, cutoff), plus per-candidate rows at the record cutoff."""
    warnings.simplefilter("ignore")
    st = L.load_star(task)
    pooled = L.cached_guesses(st, Path(cache_dir))
    out = dict(star_id=st["star_id"], true_period=st["true_p"], period_bin=ev.period_bin(st["true_p"]),
               prange=st["prange"], sweep={}, cands={}, seconds={})
    for v in variants:
        _memo_fit()
        t0 = time.perf_counter()
        rec_cut = RECORD_CUTOFF.get(v, C.COMB_VARIANTS[v].get("min_mean_height_snr", 1.0))
        cuts = sorted(set(cutoffs) | {rec_cut})
        for cut in cuts:
            try:
                res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"],
                                            comb_variant=v, min_mean_height_snr=float(cut))
                P = float(res.best_fit.P) if res.success else None
                outc = ev.classify_period(P, st["true_p"], ev.REL_TOL, st["prange"])
                gates = ";".join((res.decline_reason or {}).get("gates", []))
                if cut == rec_cut:
                    out["cands"][v] = cand_rows(res, st["true_p"], st["prange"])
                    out["seconds"][v] = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                P, outc, gates = None, ev.NONE, f"error:{type(exc).__name__}"
            out["sweep"][(v, float(cut))] = (P, outc, gates)
        C._fit_single_candidate, C._grid_search_t0 = _ORIG_FIT, _ORIG_T0
    return out


def run(mode: str, manifest: str, out_pkl: str, n_workers: int = 3, star_ids=None) -> list:
    """run_star over a manifest (atomic pickle output)."""
    from concurrent.futures import ProcessPoolExecutor
    import determinism
    tasks = L.make_tasks(mode, manifest)
    if star_ids is not None:
        tasks = [t for t in tasks if t["star_id"] in set(star_ids)]
    cache = CACHE_ROOT / (Path(manifest).stem + "_" + mode)
    cache.mkdir(parents=True, exist_ok=True)
    worker = partial(run_star, cache_dir=str(cache))
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["fullyear", "assigned"], required=True)
    ap.add_argument("--manifest", default="datasets/dev_medium_b_clean.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = None
    if a.limit:
        ids = [t["star_id"] for t in L.make_tasks(a.mode, a.manifest)[:a.limit]]
    t = time.time()
    recs = run(a.mode, a.manifest, a.out, a.n_workers, ids)
    print(f"{len(recs)} stars in {time.time() - t:.0f} s -> {a.out}")
