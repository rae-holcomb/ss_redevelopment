"""Run the rotation pipeline on a handful of WG6 resampled (Roman-cadence) stars (T28, 2026-11-12).

For each class in ``--classes`` it picks ``--n`` stars (seeded) from the inventory of one noise variant, takes F146 in one
71-d season, converts mag to relative flux (median 1), and runs the standard pipeline
(compute_acf → gather_initial_guesses, n_guesses=10 → fit_rotation_period). It writes one row per star to
``<out>/pipeline_test.csv``: truth, answer, ``evaluation.classify_period`` outcome against the catalog period (and, for
ECL/ELL/HB, against P_orb/2), the top-3 candidates, and wall time. FL truth is the literature rotation period of the
TESS template (Günther+2020 / Medina+2020); OGLE classes use the OCVS period from ``truth_joined.csv.gz``.
Single process, BLAS pinned to 1 thread.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(REPO / "tools/audit"))
from wg6r_profile import read_member, resolve_root  # noqa: E402

REPORT = REPO / "reports/2026-11-12_wg6-dataset-scoping/data"


def season_slice(t: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k-th contiguous season (split at gaps > 1 d)."""
    cut = np.where(np.diff(t) > 1.0)[0]
    starts, ends = np.r_[0, cut + 1], np.r_[cut, len(t) - 1]
    m = np.zeros(len(t), bool)
    m[starts[k]: ends[k] + 1] = True
    return m


def main(argv=None):
    """CLI: pick stars, run the pipeline on one season each, write pipeline_test.csv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--variant", default="improved")
    ap.add_argument("--classes", nargs="*", default=["FL", "ECL", "ELL", "T2CEP", "RRLYR"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--season", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
    from comb_fit import fit_rotation_period
    from evaluation import classify_period
    from guesses import gather_initial_guesses

    inv = pd.read_csv(REPORT / "resampled/inventory.csv.gz", low_memory=False)
    inv = inv[inv.variant == args.variant]
    truth = pd.read_csv(REPORT / "resampled/truth.csv.gz", low_memory=False)
    p_of = dict(zip(truth.star_id, truth.P_true))
    tdir = next((root / "simple_noise").glob("drive-download-*"))
    times = {"F146": np.load(tdir / "roman_times_shortcadence.npy"),
             "F087": np.load(tdir / "roman_times_longcadence.npy"),
             "F213": np.load(tdir / "roman_times_longcadence2.npy")}
    rng = np.random.default_rng(args.seed)
    rows = []
    for cls in args.classes:
        g = inv[inv.cls == cls].copy()
        g["P_true"] = g.star_id.map(p_of)
        g = g[g.P_true.notna()]
        if cls == "FL":
            g = g.drop_duplicates("fl_template")  # distinct TESS templates
        pick = g.iloc[rng.choice(len(g), min(args.n, len(g)), replace=False)]
        for _, r in pick.iterrows():
            t, y, _ = read_member(root, r.zip, r.member, times)["F146"]
            m = season_slice(t, args.season)
            t, y = t[m] - t[m][0], y[m]
            f = 10 ** (-0.4 * (y - np.nanmedian(y)))
            t0 = time.perf_counter()
            lags, acf, counts = compute_acf(t, f, return_counts=True)
            lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
            guesses, _ = gather_initial_guesses(t, f, lags1, acf1, n_guesses=10, long_view=(lags, acf))
            res = fit_rotation_period(lags, acf, guesses, acf_counts=counts) if guesses else None
            sec = time.perf_counter() - t0
            P = res.best_fit.P if (res is not None and res.success) else np.nan
            ptrue = float(r.P_true)
            row = {"cls": cls, "star_id": r.star_id, "template": r.get("fl_template", ""), "wfmag": r.get("wfmag"),
                   "med_mag": float(np.nanmedian(y)), "P_true": ptrue, "P_found": P, "seconds": sec,
                   "n_points": len(t), "outcome": classify_period(P, ptrue) if np.isfinite(P) else "none"}
            if cls in ("ECL", "ELL", "HB"):
                row["outcome_vs_Porb_half"] = classify_period(P, ptrue / 2) if np.isfinite(P) else "none"
            row["n_guesses"] = len(guesses)
            rows.append(row)
            print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT / "resampled/pipeline_test.csv", index=False)
    print(out.groupby("cls").outcome.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
