"""Cache one example light curve per WG6 class, and time the pipeline on Roman-like single seasons (T28 scoping).

1. For each class, pick the profiled star with a catalog period whose clean LS peak matches (else any profiled star),
   and save its primary-band (t, mag, err) plus every band's times to ``<report>/data/examples/<cls>.npz``, so the
   notebook's light-curve and coverage figures don't need the drive.
2. ``--time-pipeline N``: for N ROT stars, cut the densest 70-d window of OGLE I, convert to relative flux, place it on
   a 15-min grid (NaN where no data), and run the standard pipeline (compute_acf, gather_initial_guesses with
   n_guesses=10 and 5 for acf_fft_highpass, fit_rotation_period). Wall time per star goes to
   ``<report>/data/pipeline_timing.csv``. Single process, BLAS pinned to 1 thread.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_REPO / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPORT = ROOT_REPO / "reports/2026-11-12_wg6-dataset-scoping"
sys.path.insert(0, str(ROOT_REPO / "tools/audit"))
from wg6_profile import primary_band, read_bands  # noqa: E402


def densest_window(t: np.ndarray, length: float = 70.0) -> tuple[float, float]:
    """Return (start, end) of the ``length``-day window holding the most points."""
    t = np.sort(t)
    j = np.searchsorted(t, t + length)
    i = int(np.argmax(j - np.arange(len(t))))
    return t[i], t[i] + length


def roman_like_season(t, mag, cadence_min=15.0, length=70.0):
    """Cut the densest ``length``-d window and place relative flux on a regular grid (NaN where empty)."""
    a, b = densest_window(t, length)
    m = (t >= a) & (t < b)
    tt, ff = t[m] - a, 10 ** (-0.4 * (mag[m] - np.median(mag[m])))
    dt = cadence_min / 1440.0
    grid = np.arange(0, length, dt)
    flux = np.full(len(grid), np.nan)
    idx = np.clip(np.round(tt / dt).astype(int), 0, len(grid) - 1)
    flux[idx] = ff
    return grid, flux, int(m.sum())


def main(argv=None):
    """CLI: cache examples; optionally time the pipeline on N ROT seasons."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("WG6_DATA_ROOT", "/Volumes/Data_Vault/wg6_roman_data_raw"))
    ap.add_argument("--time-pipeline", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive (or set WG6_DATA_ROOT).")
    pc = pd.read_csv(REPORT / "data/period_check.csv")
    ex = REPORT / "data/examples"
    ex.mkdir(parents=True, exist_ok=True)
    for cls, g in pc.groupby("cls"):
        if (ex / f"{cls}.npz").exists():
            continue
        good = g[g.out_clean == "match"]
        r = (good if len(good) else g).sort_values("n", ascending=False).iloc[0]
        bands = read_bands(root / r.relpath)
        pb = primary_band(bands)
        d = bands[pb]
        np.savez_compressed(ex / f"{cls}.npz", name=r.NAME, band=pb, t=d["t"], y=d["y"], e=d["e"],
                            P_true=r.P_true, bands=np.array(list(bands)),
                            **{f"t_{k}": v["t"] for k, v in bands.items()})
        print(f"cached {cls}: {r.NAME} ({pb}, {len(d['t'])} pts)")

    if args.time_pipeline:
        from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf
        from comb_fit import fit_rotation_period
        from guesses import gather_initial_guesses

        from synthetic import spot_signal

        def run(t, f):
            """Run the standard pipeline on (t, f); return (seconds, P or NaN, n_guesses, n_acf_lags)."""
            t0 = time.perf_counter()
            lags, acf, counts = compute_acf(t, f, return_counts=True)
            lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
            guesses, _ = gather_initial_guesses(t, f, lags1, acf1, n_guesses=10, long_view=(lags, acf))
            res = fit_rotation_period(lags, acf, guesses, acf_counts=counts) if guesses else None
            p = res.best_fit.P if (res is not None and res.success) else np.nan
            return time.perf_counter() - t0, p, len(guesses), len(lags)

        rot = pc[(pc.cls == "ROT") & (pc.out_clean == "match") & pc.P_true.between(2, 20)].head(args.time_pipeline)
        rows = []
        rng = np.random.default_rng(20261112)
        for _, r in rot.iterrows():
            d = read_bands(root / r.relpath)["LC_I"]
            # (a) the real OGLE season placed on a regular grid at 15 and 60 min (NaN where no data)
            for cad in (15.0, 60.0):
                t, f, n_in = roman_like_season(d["t"], d["y"], cadence_min=cad)
                sec, p, ng, nl = run(t, f)
                rows.append({"NAME": r.NAME, "P_true": r.P_true, "input": f"OGLE densest 70 d on {cad:.0f}-min grid",
                             "n_grid": len(t), "fill_frac": float(np.isfinite(f).mean()), "n_guesses": ng,
                             "n_acf_lags": nl, "seconds": sec, "P_found": p})
                print(rows[-1], flush=True)
            # (b) a Roman-like season: contiguous 70 d at 12.1 min, spot signal at the ROT period + white noise
            t = np.arange(0, 70.0, 12.1 / 1440)
            sig = spot_signal(t, r.P_true, rng, amplitude=0.01)
            sig = sig[0] if isinstance(sig, tuple) else sig
            f = 1.0 + np.asarray(sig) + rng.normal(0, 0.005, len(t))
            sec, p, ng, nl = run(t, f)
            rows.append({"NAME": r.NAME, "P_true": r.P_true, "input": "synthetic 70 d at 12.1 min (Roman-like)",
                         "n_grid": len(t), "fill_frac": 1.0, "n_guesses": ng, "n_acf_lags": nl,
                         "seconds": sec, "P_found": p})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(REPORT / "data/pipeline_timing.csv", index=False)


if __name__ == "__main__":
    main()
