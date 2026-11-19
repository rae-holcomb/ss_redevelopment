"""P34 prototype: 10 GP rotators x F146 {16, 18, 20} on WG6 season 1 (improved noise), through the scorecard per-star path.

Run from the repo root (light: 30 star-seasons, ~1 min at 2 workers):
    nice -n 15 /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-11-13_p34-roman-rotators/run_prototype.py
Writes data/prototype_results.csv, data/prototype_lcs.npz (float32 flux, for figures) and data/prototype_acf.npz
(ACF + pooled candidates of 3 stars). Same GP draw at every magnitude (seeded); only the white noise changes.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tools"))
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import make_roman_rotators as mr  # noqa: E402
import scorecard as sc  # noqa: E402

MAGS = (16.0, 18.0, 20.0)
BACKEND = "celerite2"
ACF_STARS = ("proto02_G_30d", "proto04_K_12d", "proto09_giant_22d")  # 3 stars for the ACF figure (at F146 18)


def star_name(i, p):
    return f"proto{i:02d}_{p['stype']}_{p['P_eq']:g}d"


def build():
    t = mr.roman_season_grid(1)
    lcs = []
    for i, p in enumerate(mr.prototype_params()):
        scatter = p["sigma_mmag"] / float(mr.sigma_f146_mmag(p["f146"]))  # keep the star's crowding factor
        for m in MAGS:
            q = dict(p, f146=m, sigma_mmag=float(mr.sigma_f146_mmag(m) * scatter))
            lc = mr.make_rotator(q, seed=p["seed"], t=t, backend=BACKEND, star_id=f"{star_name(i, p)}_m{m:g}")
            lcs.append(lc)
    return t, lcs


def acf_and_candidates(lc):
    """The scorecard's Stage-1/Stage-2 inputs for one star: ACF, pooled candidate P0s, final period."""
    t, f = lc.time, lc.flux
    lags, acf, counts = sc.compute_acf(t, f, return_counts=True)
    lags1, acf1 = sc.compute_acf(t, f, max_lag_frac=sc.STAGE1_MAX_LAG_FRAC)
    pooled = []
    for name, fn in sc.discover_guess_functions().items():
        if name not in sc.DEFAULT_PIPELINE_METHODS:
            continue
        flux_in = sc._gap_filled(f) if name == "wavelet" else f
        pooled.extend(sc.two_view_guesses(fn, t, flux_in, (lags1, acf1), (lags, acf),
                                          n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT)))
    res = sc.fit_rotation_period(lags, acf, pooled, acf_counts=counts)
    return lags, acf, np.array([g.P0 for g in pooled]), (float(res.best_fit.P) if res.success else np.nan)


def main():
    t, lcs = build()
    worker = partial(sc.score_one_star, pipeline_methods=sc.DEFAULT_PIPELINE_METHODS, rel_tol=sc.ev.REL_TOL)
    with ProcessPoolExecutor(2, initializer=determinism.init_worker) as ex:
        recs = list(ex.map(worker, [mr.scorecard_task(lc) for lc in lcs]))
    rows = []
    for lc, r in zip(lcs, recs):
        p = lc.params
        rows.append(dict(star_id=lc.star_id, stype=p["stype"], P_eq=p["P_eq"], f146=p["f146"],
                         sigma_mmag=p["sigma_mmag"], rvar_f146_mmag=p["rvar_f146_mmag"],
                         rvar_realized_mmag=p["rvar_realized_mmag"], snr_rvar=p["rvar_realized_mmag"] / p["sigma_mmag"],
                         tau_spot_days=p["tau_spot_days"], shear=p["shear"], P_lo=p["period_range"][0],
                         P_hi=p["period_range"][1], band_periods=";".join(f"{x:.4g}" for x in p["band_periods"]),
                         P_found=r.get("final_period"), outcome=r["final_outcome"], error=r.get("error"),
                         seconds=r["seconds"], seed=p["seed"], backend=p["backend"]))
    out = pd.DataFrame(rows)
    (HERE / "data").mkdir(exist_ok=True)
    out.to_csv(HERE / "data/prototype_results.csv", index=False)
    np.savez_compressed(HERE / "data/prototype_lcs.npz", time=t.astype(np.float32),
                        ids=np.array([lc.star_id for lc in lcs]),
                        flux=np.array([lc.flux for lc in lcs], dtype=np.float32))
    acf_store = {}
    for lc in lcs:
        base = lc.star_id.rsplit("_m", 1)[0]
        if base in ACF_STARS and lc.star_id.endswith("_m18"):
            lags, acf, cands, Pfit = acf_and_candidates(lc)
            acf_store[base + "_lags"], acf_store[base + "_acf"] = lags.astype(np.float32), acf.astype(np.float32)
            acf_store[base + "_cands"], acf_store[base + "_Pfit"] = cands, np.array([Pfit])
    np.savez_compressed(HERE / "data/prototype_acf.npz", **acf_store)
    print(out[["star_id", "P_eq", "f146", "snr_rvar", "P_found", "outcome", "seconds"]].to_string())


if __name__ == "__main__":
    main()
