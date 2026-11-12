"""Inventory and profile the WG6 *resampled* (Roman-cadence, Roman-filter) light curves, read in place from the zips.

Root: ``$WG6_RESAMPLED_ROOT`` (default ``/Volumes/Data_Vault/wg6_roman_data_resampled``), holding two variants:

- ``simple_noise/RGES_filters_<CLS>_lightcurves[_shifted]-*.zip`` (times in ``drive-download-*/roman_times_*.npy``);
- ``improved noise/RGES_filters_<CLS>_lightcurves_final-*.zip`` (JD column inside each FITS). ``FL`` also has
  non-``_final`` zips in that folder, in the simple-noise format.

Nothing is extracted: each sampled member is read into memory (~1.3 MB). Outputs go to ``--out`` (default
``reports/2026-11-12_wg6-dataset-scoping/data/resampled``):

- ``inventory.csv.gz``: every member (variant, cls, star_id, zip, member, WFmag/template tags for FL);
- ``profile.csv``: one row per sampled (variant, member): median mag, median error, p2p scatter, top-5 LS peaks
  on F146 (all 6 seasons), and, for stars also present in ``simple``, the offset, correlation and noise-difference
  statistics against the simple version;
- ``noise_diff.csv``: whiteness tests of (variant − simple − offset), i.e. the difference of two noise draws on the
  same underlying signal: lag-1..5 autocorrelation and the RMS of bin means vs the white expectation;
- ``spectra/*.npz`` and ``examples/*.npz``: periodograms and light curves for the figures.

Resumable (``profile.csv`` rows already present are skipped). Serial and light.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle

DEFAULT_ROOT = "/Volumes/Data_Vault/wg6_roman_data_resampled"
OUT = "reports/2026-11-12_wg6-dataset-scoping/data/resampled"
ID_RE = re.compile(r"RGES_filters_(OGLE-[A-Z]+-[A-Z0-9]+-\d+)(?:_ind(\d+)_(\d+))?")
FL_RE = re.compile(r"FL_(\d+)_([^_]+)_(\d+)_WFmag_(\d+)_lightcurves")


def resolve_root(cli):
    """Resampled root (CLI > env > default); exit cleanly if the drive is absent."""
    root = Path(cli or os.environ.get("WG6_RESAMPLED_ROOT", DEFAULT_ROOT))
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive (or set WG6_RESAMPLED_ROOT).")
    return root


def build_inventory(root: Path) -> pd.DataFrame:
    """One row per FITS member of every zip (names only; no decompression)."""
    rows = []
    for z in sorted(glob.glob(str(root / "*" / "*.zip"))):
        zrel = str(Path(z).relative_to(root))
        folder = zrel.split("/")[0]
        cls = re.search(r"RGES_filters_([A-Z0-9]+)_lightcurves", z).group(1)
        for n in zipfile.ZipFile(z).namelist():
            if not n.endswith(".fits"):
                continue
            if folder == "simple_noise":
                variant = "simple_shifted" if "_shifted" in n else "simple"
            else:
                variant = "improved" if "_final" in n else "improved_folder_nonfinal"
            r = {"variant": variant, "cls": cls, "zip": zrel, "member": n}
            m = ID_RE.search(n)
            if m:
                r["star_id"] = m.group(1)
                r["cv_ind"] = f"{m.group(2)}_{m.group(3)}" if m.group(2) else ""
            f = FL_RE.search(n)
            if f:
                r.update(star_id=f"FL_{f.group(1)}", fl_template=f.group(2), fl_k=int(f.group(3)),
                         wfmag=int(f.group(4)))
            rows.append(r)
    return pd.DataFrame(rows)


def read_member(root: Path, zrel: str, member: str, times: dict) -> dict:
    """Return {band: (t, mag, err)} for one FITS member; times from the JD column or the simple-noise .npy grids."""
    with zipfile.ZipFile(root / zrel) as zf, fits.open(io.BytesIO(zf.read(member))) as h:
        out = {}
        for band in ("F146", "F087", "F213"):
            if band not in h:
                continue
            d = h[band].data
            t = np.asarray(d["JD"], float) if "JD" in d.columns.names else times[band]
            out[band] = (t, np.asarray(d["mag"], float), np.asarray(d["mag_error"], float))
        return out


def p2p(y):
    """Robust point-to-point scatter: 1.4826 * MAD(diff) / sqrt(2)."""
    d = np.diff(y[np.isfinite(y)])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def whiteness(r):
    """Lag-1..5 autocorrelation and binned-RMS / white-expectation ratios (bins of 4, 16, 64) of a residual series."""
    r = r - np.mean(r)
    v = np.mean(r * r)
    ac = [float(np.mean(r[:-k] * r[k:]) / v) for k in range(1, 6)]
    ratio = {}
    for b in (4, 16, 64):
        m = len(r) // b * b
        bm = r[:m].reshape(-1, b).mean(1)
        ratio[b] = float(np.std(bm) / (np.std(r) / np.sqrt(b)))
    return ac, ratio


def ls_peaks(t, y, pmin=0.02, pmax=300.0, n_peaks=5):
    """Lomb–Scargle top peaks; uniform frequency grid with 5 samples per 1/baseline; returns (f, p, P, pow)."""
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    base = np.ptp(t)
    fmin, fmax = max(1 / pmax, 1 / base), 1 / pmin
    freq = np.linspace(fmin, fmax, int(min(2e6, 5 * base * (fmax - fmin))))
    pw = LombScargle(t, y).power(freq, method="fast")
    lm = np.where((pw[1:-1] > pw[:-2]) & (pw[1:-1] >= pw[2:]))[0] + 1
    lm = lm[np.argsort(pw[lm])[::-1]]
    keep = []
    for i in lm:
        if all(abs(freq[i] - freq[j]) > 1.5 / base for j in keep):
            keep.append(i)
        if len(keep) == n_peaks:
            break
    return freq, pw, [1 / freq[i] for i in keep], [float(pw[i]) for i in keep]


def main(argv=None):
    """CLI: build the inventory, then profile N sampled stars per (variant, class)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=12, help="stars per class (same stars in every variant)")
    ap.add_argument("--n-fl", type=int, default=30, help="FL objects to sample")
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    out = Path(args.out)
    for sub in ("spectra", "examples"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    inv_path = out / "inventory.csv.gz"
    if inv_path.exists():
        inv = pd.read_csv(inv_path, low_memory=False)
    else:
        inv = build_inventory(root)
        inv.to_csv(str(inv_path) + ".part", index=False, compression="gzip")
        os.replace(str(inv_path) + ".part", inv_path)
    print(inv.groupby(["variant", "cls"]).size().to_string(), flush=True)

    tdir = next((root / "simple_noise").glob("drive-download-*"))
    times = {"F146": np.load(tdir / "roman_times_shortcadence.npy"),
             "F087": np.load(tdir / "roman_times_longcadence.npy"),
             "F213": np.load(tdir / "roman_times_longcadence2.npy")}

    rng = np.random.default_rng(args.seed)
    picks = []  # (star_id) per class, same stars in every variant that has them
    for cls, g in inv[inv.cls != "FL"].groupby("cls"):
        ids = sorted(set(g.star_id))
        picks += list(rng.choice(ids, min(args.n, len(ids)), replace=False))
    fl = inv[inv.cls == "FL"]
    fl_ids = sorted(set(fl.star_id))
    picks += list(rng.choice(fl_ids, min(args.n_fl, len(fl_ids)), replace=False))
    sel = inv[inv.star_id.isin(picks)]
    # one member per (star, variant) for OGLE classes; every magnitude copy for FL (noise vs magnitude on one object)
    sel = sel.sort_values(["star_id", "variant", "member"])
    sel = pd.concat([sel[sel.cls != "FL"].groupby(["star_id", "variant"]).head(1), sel[sel.cls == "FL"]])

    prof_path, nd_path = out / "profile.csv", out / "noise_diff.csv"
    done = set(pd.read_csv(prof_path).member) if prof_path.exists() else set()
    pcols = ["variant", "cls", "star_id", "member", "fl_template", "wfmag", "band", "n", "med_mag", "med_err",
             "p2p", "nan_frac", "p2p_over_err", "P1", "P2", "P3", "P4", "P5", "pow1", "off_vs_simple", "corr_vs_simple"]
    ncols = ["variant", "cls", "star_id", "med_mag", "std_diff", "expected_white", "ac1", "ac2", "ac3", "ac4", "ac5",
             "bin4", "bin16", "bin64"]
    fp, fn = open(prof_path, "a", newline=""), open(nd_path, "a", newline="")
    wp, wn = csv.DictWriter(fp, pcols), csv.DictWriter(fn, ncols)
    if fp.tell() == 0:
        wp.writeheader()
    if fn.tell() == 0:
        wn.writeheader()
    for sid, g in sel.groupby("star_id"):
        lcs = {}
        for _, r in g.iterrows():
            lcs[r.member] = (r, read_member(root, r.zip, r.member, times))
        base = next((v for v in lcs.values() if v[0].variant == "simple"), None)
        for _, (r, bands) in lcs.items():
            var = r.variant
            if r.member in done:
                continue
            for band, (t, y, e) in bands.items():
                row = {k: r.get(k) for k in ("variant", "cls", "star_id", "member", "fl_template", "wfmag")}
                row.update(band=band, n=len(y), med_mag=np.nanmedian(y), med_err=np.nanmedian(e), p2p=p2p(y),
                           nan_frac=float(np.mean(~np.isfinite(y))))
                row["p2p_over_err"] = row["p2p"] / row["med_err"]
                if band == "F146":
                    f, pw, P, PW = ls_peaks(t, y)
                    for i, p in enumerate(P):
                        row[f"P{i+1}"] = p
                    row["pow1"] = PW[0]
                    k = max(1, len(f) // 4000)
                    m = len(f) // k * k
                    np.savez_compressed(out / "spectra" / f"{var}__{sid}{'_m' + str(int(r.wfmag)) if r.cls == 'FL' else ''}.npz",
                                        freq=f[:m].reshape(-1, k).mean(1), power=pw[:m].reshape(-1, k).max(1))
                    if sid == sorted(sel[sel.cls == r.cls].star_id)[0] and not (out / "examples" / f"{r.cls}__{var}.npz").exists():
                        np.savez_compressed(out / "examples" / f"{r.cls}__{var}.npz", t=t, y=y.astype(np.float32),
                                            e=e.astype(np.float32), star_id=sid,
                                            **{f"t_{b}": bands[b][0] for b in bands})
                if base is not None and var != "simple" and band in base[1]:
                    yb = base[1][band][1]
                    if len(yb) == len(y):
                        off = np.nanmedian(y - yb)
                        row["off_vs_simple"] = off
                        ok = np.isfinite(y) & np.isfinite(yb)
                        row["corr_vs_simple"] = np.corrcoef(y[ok], yb[ok])[0, 1]
                        if band == "F146":
                            d = y - yb - off
                            d = d[np.isfinite(d)]
                            ac, ratio = whiteness(d)
                            eb = base[1][band][2]
                            wn.writerow({"variant": var, "cls": r.cls, "star_id": sid, "med_mag": np.median(y),
                                         "std_diff": np.nanstd(d),
                                         "expected_white": float(np.sqrt(np.median(e) ** 2 + np.median(eb) ** 2)),
                                         **{f"ac{i+1}": a for i, a in enumerate(ac)},
                                         **{f"bin{b}": v for b, v in ratio.items()}})
                wp.writerow(row)
            fp.flush(); fn.flush()
        print(f"{sid}: {sorted(lcs)}", flush=True)
    fp.close(); fn.close()


if __name__ == "__main__":
    main()
