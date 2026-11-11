"""Profile a random sample of WG6 light curves per set (T28 scoping, 2026-11-12).

For each sampled file and each band extension: number of points, time span, median
cadence, season structure (segments split at gaps > ``--season-gap`` d), median
magnitude, point-to-point scatter, median quoted error. For the primary band (OGLE
``LC_I``, else TESS, else the longest band) it also runs a Lomb-Scargle periodogram
and records the top peaks, which are cross-checked against literature truth periods
where those exist.

Also handles the time-stamp quirks found in scoping: values > 3e6 (+2450000 applied
twice in the WG6 code) are shifted back by 2450000.

Outputs (small, committed): ``<out>/profile_bands.csv`` (one row per file × band),
``<out>/profile_ls.csv`` (one row per file: primary band + top LS peaks) and
``<out>/ls_spectra/<set>__<name>.npz`` (frequency grid + power, for the figures).
Resumable: files already in profile_ls.csv are skipped. Serial and light.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle

DEFAULT_ROOT = "/Volumes/Data_Vault/wg6_roman_data"


def resolve_root(cli_root):
    """Return the WG6 root (CLI > env WG6_DATA_ROOT > default); exit if the drive is absent."""
    root = Path(cli_root or os.environ.get("WG6_DATA_ROOT", DEFAULT_ROOT))
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive (or set WG6_DATA_ROOT).")
    return root


def fix_time(t: np.ndarray) -> np.ndarray:
    """Undo the WG6 double +2450000 offset (values > 3e6) so all times are HJD/BJD-like."""
    t = np.asarray(t, float)
    return np.where(t > 3.0e6, t - 2450000.0, t)


def read_bands(path: Path) -> dict[str, dict]:
    """Return {extname: {'t', 'y', 'e', 'cols'}} for every non-empty table extension.

    ``y`` is the second column (magnitude for OGLE/UKIRT, see ``cols``), ``e`` the third.
    """
    out = {}
    with fits.open(path) as h:
        for x in h[1:]:
            if x.data is None or len(x.data) == 0:
                continue
            names = x.columns.names
            d = x.data
            out[x.name] = {"t": fix_time(d[names[0]]), "y": np.asarray(d[names[1]], float),
                           "e": np.asarray(d[names[2]], float) if len(names) > 2 else np.full(len(d), np.nan),
                           "cols": names}
    return out


def segments(t: np.ndarray, gap: float) -> list[tuple[float, float, int]]:
    """Split sorted times at gaps > ``gap`` days; return (start, end, n) per segment."""
    t = np.sort(t)
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > gap)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(t) - 1]
    return [(t[a], t[b], b - a + 1) for a, b in zip(starts, ends)]


def p2p_scatter(y: np.ndarray) -> float:
    """Point-to-point scatter: 1.4826 * MAD(diff(y)) / sqrt(2) (robust white-noise level)."""
    d = np.diff(y[np.isfinite(y)])
    if len(d) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def primary_band(bands: dict) -> str | None:
    """Pick the band used for periods: OGLE I, then any TESS band, then the longest."""
    if "LC_I" in bands and len(bands["LC_I"]["t"]) >= 30:
        return "LC_I"
    tess = [k for k in bands if k.startswith(("LC_T", "TESS"))]
    if tess:
        return max(tess, key=lambda k: len(bands[k]["t"]))
    return max(bands, key=lambda k: len(bands[k]["t"])) if bands else None


def ls_peaks(t, y, pmin=0.02, pmax=1000.0, n_peaks=5, max_n=20000):
    """Lomb-Scargle on (t, y); return (freq, power, top periods, top powers).

    Grid: uniform in frequency from 1/pmax (or 1/baseline) to 1/pmin with 5 samples per
    1/baseline resolution element, capped at 2e6 frequencies (coarser if needed).
    Very long series are subsampled to ``max_n`` points (random, seeded) to stay light.
    """
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if len(t) > max_n:
        idx = np.sort(np.random.default_rng(0).choice(len(t), max_n, replace=False))
        t, y = t[idx], y[idx]
    base = t.max() - t.min()
    fmin = max(1.0 / pmax, 1.0 / base)
    fmax = 1.0 / pmin
    nf = int(min(2e6, 5 * base * (fmax - fmin)))
    freq = np.linspace(fmin, fmax, max(nf, 1000))
    power = LombScargle(t, y).power(freq, method="fast")
    # local maxima, strongest first, at least 1 resolution element apart
    lm = np.where((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:]))[0] + 1
    lm = lm[np.argsort(power[lm])[::-1]]
    keep = []
    for i in lm:
        if all(abs(freq[i] - freq[j]) > 1.5 / base for j in keep):
            keep.append(i)
        if len(keep) == n_peaks:
            break
    return freq, power, [1.0 / freq[i] for i in keep], [float(power[i]) for i in keep]


def main(argv=None):
    """CLI: sample N files per set from the inventory CSVs and profile them."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--inventory", default="reports/2026-11-12_wg6-dataset-scoping/data/inventory")
    ap.add_argument("--out", default="reports/2026-11-12_wg6-dataset-scoping/data")
    ap.add_argument("--n", type=int, default=30, help="files per set")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--season-gap", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    out = Path(args.out)
    (out / "ls_spectra").mkdir(parents=True, exist_ok=True)
    import pandas as pd

    done = set()
    ls_path, band_path = out / "profile_ls.csv", out / "profile_bands.csv"
    if ls_path.exists():
        done = set(pd.read_csv(ls_path)["relpath"])
    band_cols = ["set", "relpath", "NAME", "band", "cols", "n", "t0", "t1", "span_d", "med_dt_min",
                 "n_seasons", "med_season_len_d", "med_season_n", "med_y", "p2p", "med_err"]
    ls_cols = ["set", "relpath", "NAME", "VARTYPE", "band", "n", "span_d",
               "P1", "P2", "P3", "P4", "P5", "pow1", "pow2", "pow3"]
    fb = open(band_path, "a", newline="")
    fl = open(ls_path, "a", newline="")
    wb, wl = csv.DictWriter(fb, band_cols), csv.DictWriter(fl, ls_cols)
    if fb.tell() == 0:
        wb.writeheader()
    if fl.tell() == 0:
        wl.writeheader()
    for inv in sorted(Path(args.inventory).glob("*.csv*")):
        s = inv.name.split(".")[0]
        if args.sets and s not in args.sets:
            continue
        df = pd.read_csv(inv)
        df = df[df["error"].isna()] if "error" in df else df
        samp = df.sample(min(args.n, len(df)), random_state=args.seed)
        print(f"{s}: profiling {len(samp)} of {len(df)}", flush=True)
        for _, r in samp.iterrows():
            if r["relpath"] in done:
                continue
            bands = read_bands(root / r["relpath"])
            for b, d in bands.items():
                tt = np.sort(d["t"])
                seg = segments(tt, args.season_gap)
                wb.writerow({"set": s, "relpath": r["relpath"], "NAME": r["NAME"], "band": b,
                             "cols": "|".join(d["cols"]), "n": len(tt), "t0": tt[0], "t1": tt[-1],
                             "span_d": tt[-1] - tt[0],
                             "med_dt_min": np.median(np.diff(tt)) * 1440 if len(tt) > 1 else np.nan,
                             "n_seasons": len(seg),
                             "med_season_len_d": np.median([b1 - a1 for a1, b1, _ in seg]),
                             "med_season_n": np.median([n for *_, n in seg]),
                             "med_y": np.nanmedian(d["y"]), "p2p": p2p_scatter(d["y"][np.argsort(d["t"])]),
                             "med_err": np.nanmedian(d["e"])})
            pb = primary_band(bands)
            row = {"set": s, "relpath": r["relpath"], "NAME": r["NAME"], "VARTYPE": r["VARTYPE"], "band": pb}
            if pb is not None and len(bands[pb]["t"]) >= 30:
                d = bands[pb]
                freq, power, P, pw = ls_peaks(d["t"], d["y"])
                row.update(n=len(d["t"]), span_d=np.ptp(d["t"]))
                for i, p in enumerate(P[:5]):
                    row[f"P{i+1}"] = p
                for i, p in enumerate(pw[:3]):
                    row[f"pow{i+1}"] = p
                # downsample the spectrum for storage (max-pool into 4000 bins)
                k = max(1, len(freq) // 4000)
                m = len(freq) // k * k
                np.savez_compressed(out / "ls_spectra" / f"{s}__{r['NAME']}.npz",
                                    freq=freq[:m].reshape(-1, k).mean(1), power=power[:m].reshape(-1, k).max(1))
            wl.writerow(row)
            fb.flush(); fl.flush()
    fb.close(); fl.close()


if __name__ == "__main__":
    main()
