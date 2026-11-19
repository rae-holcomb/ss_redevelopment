"""P43 clean-rerun helpers: a production-path harness with cached Stage-1 guesses, and per-tooth diagnostics.

The harness runs the scorecard's Stage 1 (the pooled methods on both ACF
views) once per star and caches the pooled guesses, then runs Stage 2
(`fit_rotation_period`) under any number of comb variants. Final outcomes
must match the scorecard's per_star.csv for the same variant (checked in
analysis.py); official numbers always come from `benchmarks/scorecard.py`.

Only `datasets/dev_{small,medium}_clean.csv` are used (Rae, 2026-11-16).
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

MANIFEST = "datasets/dev_medium_clean.csv"
CACHE_ROOT = ROOT / "runs" / "p43_clean" / "stage1_cache"


def make_tasks(mode: str, manifest: str = MANIFEST, data_root: str = "data/smarts_dev") -> list:
    """Scorecard task dicts for `manifest` in mode 'fullyear' (ignore manifest sectors) or 'assigned'."""
    if "_clean" not in manifest:
        raise ValueError("develop and judge only on the noise-clean manifests (Rae, 2026-11-16)")
    args = SimpleNamespace(synthetic=None, manifest=ROOT / manifest, ignore_manifest_sectors=(mode == "fullyear"),
                           coverage=cov.FULL, coverage_seed=0, coverage_weighting="frequency",
                           coverage_library=None, data_root=str(ROOT / data_root), noise_table=None,
                           noise_subset="all", allow_noise_excluded=False)
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


def cached_guesses(st: dict, cache_dir: Path) -> list:
    """Pooled guesses from cache_dir/<star_id>.pkl, computing and writing them (atomically) if absent."""
    cp = Path(cache_dir) / f"{st['star_id']}.pkl"
    if cp.exists():
        return pickle.loads(cp.read_bytes())
    pooled = pooled_guesses(st)
    tmp = cp.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(pooled))
    os.replace(tmp, cp)
    return pooled


def tooth_table(cand, spec: dict) -> list:
    """Per-tooth record of one fitted local-solver candidate: n, model centre, free vertex, height, tol, pass."""
    fit = cand.fit
    li = fit.local_info or {}
    P = float(fit.P)
    rows = []
    for n, pk in sorted(fit.per_peak.items()):
        v = li.get("free_vertex", {}).get(n, np.nan) if li else np.nan
        m = (li["model_t0"] + li["model_P"] * n + li["drift_a"] * n * n) if li else pk["center"]
        tol = float(spec.get("tol_base", 0.05)) * P * np.sqrt(max(n, 1))
        rows.append(dict(n=int(n), model_c=float(m), free_vertex=float(v),
                         free_height=float(li.get("free_height", {}).get(n, np.nan)) if li else np.nan,
                         height=float(pk["height"]), curvature=float(pk["curvature"]), tol=float(tol),
                         lag0=bool(m < 0.5 * P),
                         tol_ok=bool(np.isfinite(v) and abs(v - m) <= tol)))
    return rows


def candidate_rows(res, true_p, prange, spec: dict) -> list:
    """Per-candidate summary rows (gate values, pass flags, outcome, per-tooth table)."""
    rows = []
    for rank, r in enumerate(res.candidates):
        g = r.gate_results or {}
        fit = r.fit
        rows.append(dict(
            rank=rank, P0=r.period, t0=r.t0, P=fit.P, passed=bool(r.passed_gates),
            selected=bool(res.success and rank == 0),
            outcome=ev.classify_period(fit.P, true_p, ev.REL_TOL, prange),
            failed=";".join(r.failed_gates), redchi=fit.redchi, n_peaks_used=fit.n_peaks_used,
            n_trimmed=fit.n_peaks_trimmed, n_dropped=fit.n_peaks_dropped, n_required=r.n_peaks_required,
            few=bool(r.few_peaks),
            **{f"v_{k}": v.get("value") for k, v in g.items()},
            **{f"ok_{k}": bool(v.get("passed")) for k, v in g.items() if v.get("enabled", True)},
            teeth=tooth_table(r, spec),
        ))
    return rows


def run_star(task: dict, variants: tuple, cache_dir: str, keep_candidates: bool = True) -> dict:
    """Score one star under every comb variant in `variants` (module-level, picklable)."""
    warnings.simplefilter("ignore")
    st = load_star(task)
    pooled = cached_guesses(st, Path(cache_dir))
    out = dict(star_id=st["star_id"], true_period=st["true_p"], period_bin=ev.period_bin(st["true_p"]),
               period_lo=st["prange"][0] if st["prange"] else None,
               period_hi=st["prange"][1] if st["prange"] else None, variants={})
    for v in variants:
        spec = C.COMB_VARIANTS[v]
        t0 = time.perf_counter()
        try:
            res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
            secs = time.perf_counter() - t0
            P = float(res.best_fit.P) if res.success else None
            rec = dict(final_period=P, final_outcome=ev.classify_period(P, st["true_p"], ev.REL_TOL, st["prange"]),
                       decline_gates=";".join((res.decline_reason or {}).get("gates", [])), fit_seconds=secs)
            if keep_candidates:
                rec["candidates"] = candidate_rows(res, st["true_p"], st["prange"], spec)
        except Exception as exc:  # noqa: BLE001
            rec = dict(final_period=None, final_outcome=ev.NONE, decline_gates="", fit_seconds=np.nan,
                       error=f"{type(exc).__name__}: {exc}")
        out["variants"][v] = rec
    return out


def run_harness(mode: str, variants: tuple, out_pkl: str, n_workers: int = 3, manifest: str = MANIFEST,
                star_ids=None, keep_candidates: bool = True) -> list:
    """Run run_star over a manifest; writes `out_pkl` atomically and returns the records."""
    from concurrent.futures import ProcessPoolExecutor
    tasks = make_tasks(mode, manifest)
    if star_ids is not None:
        keep = set(star_ids)
        tasks = [t for t in tasks if t["star_id"] in keep]
    cache = CACHE_ROOT / (Path(manifest).stem + "_" + mode)
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="P43 harness: cached Stage 1 + Stage 2 under several comb variants.")
    ap.add_argument("--mode", choices=["fullyear", "assigned"], required=True)
    ap.add_argument("--variants", required=True, help="comma-separated COMB_VARIANTS keys")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--star-ids-file", default=None)
    ap.add_argument("--n-workers", type=int, default=3)
    a = ap.parse_args()
    ids = None
    if a.star_ids_file:
        ids = [s.strip() for s in Path(a.star_ids_file).read_text().split() if s.strip()]
    t = time.time()
    recs = run_harness(a.mode, tuple(a.variants.split(",")), a.out, n_workers=a.n_workers, manifest=a.manifest,
                       star_ids=ids)
    print(f"{len(recs)} stars in {time.time() - t:.0f} s -> {a.out}")
