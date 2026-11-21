"""P59 sweep harness: cached Stage-1 guesses + memoized comb fits, rescored over a threshold grid.

Same method as P44's harness (reports/2026-11-20_p44-lag0-out/p44lib.py): per star, Stage 1 runs once
(cached on disk; P44's cache is reused), then the production `fit_rotation_period` is called once per
(variant, min_peaks_required, min_frac_positive_heights, min_mean_height_snr) with
`_fit_single_candidate` and `_grid_search_t0` memoized in-process. min_peaks_required changes the
rejection/trim floors, so it can change fits; the memo key includes those arguments, so that is exact too.

Only noise-clean manifests (p43lib.make_tasks refuses others). Tuning: dev_medium_b_clean only.
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import time
import warnings
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports" / "2026-11-20_p44-lag0-out"))
import p44lib as P44  # noqa: E402  (pins BLAS threads before numpy; memo helpers)

L, C, np = P44.L, P44.C, P44.np
import evaluation as ev  # noqa: E402

CACHE_ROOT = ROOT / "runs" / "p44" / "stage1_cache"  # P44's Stage-1 cache (same manifests, same Stage 1)
VARIANTS = ("local_tol_trim_l0x", "local_tol_trim_l0x2")
COARSE = dict(mp=(3, 4, 5), fp=(0.6, 0.7, 0.75, 0.8, 0.9, 1.0), snr=(0.90, 0.94, 0.98, 1.02, 1.06, 1.10, 1.14))
RECORD = (4, 0.8, 1.02)  # per-candidate rows recorded at the current defaults


def grid(spec: dict) -> list:
    """All (mp, fp, snr) triples of a dict of value lists."""
    return [tuple(float(x) if i else int(x) for i, x in enumerate(t))
            for t in itertools.product(spec["mp"], spec["fp"], spec["snr"])]


def cand_rows(res, true_p, prange) -> list:
    """Per-candidate gate values and outcome."""
    rows = []
    for rank, r in enumerate(res.candidates):
        g = r.gate_results or {}
        rows.append(dict(rank=rank, P=float(r.fit.P), passed=bool(r.passed_gates),
                         outcome=ev.classify_period(r.fit.P, true_p, ev.REL_TOL, prange),
                         failed=";".join(r.failed_gates), n_required=r.n_peaks_required, few=bool(r.few_peaks),
                         **{f"v_{k}": v.get("value") for k, v in g.items()}))
    return rows


def run_star(task: dict, cache_dir: str, combos: dict) -> dict:
    """Outcome of one star for every (variant, mp, fp, snr) in combos[variant]."""
    warnings.simplefilter("ignore")
    st = L.load_star(task)
    pooled = L.cached_guesses(st, Path(cache_dir))
    out = dict(star_id=st["star_id"], true_period=st["true_p"], period_bin=ev.period_bin(st["true_p"]),
               sweep={}, cands={})
    for v, triples in combos.items():
        P44._memo_fit()
        for mp, fp, snr in triples:
            try:
                res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v,
                                            min_peaks_required=mp, min_frac_positive_heights=fp,
                                            min_mean_height_snr=snr)
                Pf = float(res.best_fit.P) if res.success else None
                outc = ev.classify_period(Pf, st["true_p"], ev.REL_TOL, st["prange"])
                if (mp, fp, snr) == RECORD:
                    out["cands"][v] = cand_rows(res, st["true_p"], st["prange"])
            except Exception as exc:  # noqa: BLE001
                Pf, outc = None, ev.NONE
            out["sweep"][(v, mp, fp, snr)] = outc
        C._fit_single_candidate, C._grid_search_t0 = P44._ORIG_FIT, P44._ORIG_T0
    return out


def run(mode: str, manifest: str, out_pkl: str, combos: dict, n_workers: int = 3, limit=None) -> list:
    """run_star over a manifest (atomic pickle output)."""
    from concurrent.futures import ProcessPoolExecutor
    import determinism
    tasks = L.make_tasks(mode, manifest)[:limit] if limit else L.make_tasks(mode, manifest)
    cache = CACHE_ROOT / (Path(manifest).stem + "_" + mode)
    cache.mkdir(parents=True, exist_ok=True)
    worker = partial(run_star, cache_dir=str(cache), combos=combos)
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
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["fullyear", "assigned"], required=True)
    ap.add_argument("--manifest", default="datasets/dev_medium_b_clean.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default=None, help="JSON {variant: {mp: [..], fp: [..], snr: [..]}}; default COARSE "
                                                  "for both variants (+ RECORD)")
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    g = json.loads(a.grid) if a.grid else {v: COARSE for v in VARIANTS}
    combos = {v: sorted(set(grid(spec)) | {RECORD}) for v, spec in g.items()}
    print({v: len(c) for v, c in combos.items()}, "combinations per star")
    t = time.time()
    recs = run(a.mode, a.manifest, a.out, combos, a.n_workers, a.limit)
    print(f"{len(recs)} stars in {time.time() - t:.0f} s -> {a.out}")
