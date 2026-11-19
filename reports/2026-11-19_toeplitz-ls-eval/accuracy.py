#!/usr/bin/env python3
"""accuracy.py -- Toeplitz-LS vs astropy LombScargle on the exact grids guess_lombscargle uses (P50).

For each star (dev_small_clean only) and each mode (assigned sectors, full year) this
  * loads the light curve the pipeline sees (load_smarts_fits, 30-min regrid, NaN dropped);
  * rebuilds guess_lombscargle's frequency grid (default "lombscargle": [4 dt, B/2], spp=1)
    and guess_lombscargle_short's (max_period=15 d, spp=5);
  * evaluates power with
      astropy_fast  -- LombScargle(...).autopower (method "auto" -> "fast"; today's pipeline),
      astropy_exact -- LombScargle(...).power(freq, method="cython") (direct O(N Nf) sum; reference),
      tls           -- toeplitz_ls.tls.power  (float64, nterms=1, "standard"),
      tlsf          -- toeplitz_ls.tlsf.power (float32 data, nterms=1, "standard");
  * runs guess_lombscargle's own peak picking (find_peaks + guesses._select_peaks, hybrid ranking)
    with n_guesses=10 on each periodogram and records the candidate periods.

Toeplitz-LS is not installed; pass its built python package dir with --tls-path.
Usage:
    python reports/2026-11-19_toeplitz-ls-eval/accuracy.py --tls-path <scratch>/Toeplitz-LS/toeplitz-ls
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import guesses  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402

HERE = Path(__file__).resolve().parent
STARS = ["smarts-tess-v1.0-000732", "smarts-tess-v1.0-004513", "smarts-tess-v1.0-017292",
         "smarts-tess-v1.0-066625", "smarts-tess-v1.0-201481"]


def pipeline_lc(path: Path, sectors):
    """(time, flux) exactly as guess_lombscargle sees them: loader output with non-finite points dropped."""
    pre = load_smarts_fits(path, sectors=sectors)
    m = np.isfinite(pre.time) & np.isfinite(pre.flux)
    return pre.time[m], pre.flux[m]


def ls_grid(time, min_period=None, max_period=None, spp=1):
    """guess_lombscargle's grid: (f0, df, Nf, freq) matching LombScargle.autopower(minimum_frequency=...)."""
    dt = np.median(np.diff(time))
    B = time[-1] - time[0]
    min_period = 4 * dt if min_period is None else min_period
    max_period = B / 2 if max_period is None else max_period
    fmin, fmax = 1.0 / max_period, 1.0 / min_period
    freq = LombScargle(time, np.ones_like(time)).autofrequency(
        samples_per_peak=spp, minimum_frequency=fmin, maximum_frequency=fmax)
    df = 1.0 / (spp * (time.max() - time.min()))
    return float(fmin), float(fmax), df, len(freq), freq


def pick(freq, power, n=10):
    """guess_lombscargle's peak selection (same code path), returning candidate periods in rank order."""
    periods = 1.0 / freq
    order = np.argsort(periods)
    ps, pw = periods[order], np.nan_to_num(power[order], nan=0.0)
    idx, _ = find_peaks(pw)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(pw))])
    idx = guesses._select_peaks(1.0 / ps, pw, idx, n, rank_by="hybrid")
    return [float(ps[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tls-path", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data" / "smarts_dev"))
    args = ap.parse_args()
    sys.path.insert(0, args.tls_path)
    from toeplitz_ls import tls, tlsf

    man = pd.read_csv(ROOT / "datasets" / "dev_small_clean.csv").set_index("star_id")
    rows, cands, curves = [], [], {}
    for sid in STARS:
        r = man.loc[sid]
        assigned = [int(s) for s in str(r["sectors"]).split("-")]
        for mode, sectors in (("assigned", assigned), ("fullyear", None)):
            t, y = pipeline_lc(Path(args.data_root) / r["path"], sectors)
            for grid_name, kw in (("lombscargle", dict(spp=1)), ("lombscargle_short", dict(max_period=15.0, spp=5))):
                f0, fmax, df, nf, freq = ls_grid(t, **kw)
                grid_err = float(np.max(np.abs(f0 + df * np.arange(nf) - freq)))
                ls = LombScargle(t, y, normalization="standard")
                # astropy_fast exactly as guess_lombscargle calls it (min/max frequency from the period window)
                fa, pa = ls.autopower(minimum_frequency=f0, maximum_frequency=fmax, samples_per_peak=kw["spp"])
                assert len(fa) == nf and np.array_equal(fa, freq)
                pe = ls.power(freq, method="cython")
                pt = tls.power(nf, df, f0, t, y, nterms=1, normalization="standard")
                pf = tlsf.power(nf, df, f0, t, y, nterms=1, normalization="standard").astype(np.float64)
                key = f"{sid}|{mode}|{grid_name}"
                curves[key] = dict(freq=freq, astropy_fast=pa, astropy_exact=pe, tls=pt, tlsf=pf)
                row = dict(star_id=sid, true_period=float(r["true_period"]), mode=mode, grid=grid_name,
                           N=len(t), Nf=nf, grid_err=grid_err)
                for name, p in (("astropy_fast", pa), ("tls", pt), ("tlsf", pf)):
                    for ref_name, ref in (("exact", pe), ("fast", pa)):
                        if name == "astropy_fast" and ref_name == "fast":
                            continue
                        d = np.abs(p - ref)
                        rel = d / np.maximum(np.abs(ref), 1e-12)
                        row[f"{name}_vs_{ref_name}_maxabs"] = float(np.nanmax(d))
                        row[f"{name}_vs_{ref_name}_medabs"] = float(np.nanmedian(d))
                        row[f"{name}_vs_{ref_name}_maxrel"] = float(np.nanmax(rel))
                        row[f"{name}_vs_{ref_name}_medrel"] = float(np.nanmedian(rel))
                    row[f"{name}_nan"] = int(np.isnan(p).sum())
                lists = {name: pick(freq, p) for name, p in
                         (("astropy_fast", pa), ("astropy_exact", pe), ("tls", pt), ("tlsf", pf))}
                for name, lst in lists.items():
                    row[f"{name}_top10_same_as_fast"] = lst == lists["astropy_fast"]
                    row[f"{name}_top10_same_as_exact"] = lst == lists["astropy_exact"]
                    row[f"{name}_top5_same_as_fast"] = lst[:5] == lists["astropy_fast"][:5]
                    cands.append(dict(star_id=sid, mode=mode, grid=grid_name, method=name,
                                      periods=json.dumps(lst)))
                rows.append(row)
                print(key, nf, len(t), f"fast-exact {row['astropy_fast_vs_exact_maxabs']:.2e}",
                      f"tls-exact {row['tls_vs_exact_maxabs']:.2e}", f"tls-fast {row['tls_vs_fast_maxabs']:.2e}",
                      "top10 tls==fast", row["tls_top10_same_as_fast"], "tls==exact", row["tls_top10_same_as_exact"],
                      "fast==exact", row["astropy_fast_top10_same_as_exact"], flush=True)
    out = HERE / "data"
    pd.DataFrame(rows).to_csv(out / "accuracy.csv", index=False)
    pd.DataFrame(cands).to_csv(out / "candidates_top10.csv", index=False)
    np.savez_compressed(out / "periodograms.npz", **{f"{k}|{c}": v for k, d in curves.items() for c, v in d.items()})


if __name__ == "__main__":
    main()
