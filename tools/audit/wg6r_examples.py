"""Cache figure inputs for the WG6 resampled-set section of the T28 report (no drive needed afterwards).

Writes to ``reports/2026-11-12_wg6-dataset-scoping/data/resampled/examples/``:

- ``same_star.npz``: one star in the raw OGLE I set and in all three resampled variants (simple, simple_shifted,
  improved), F146 season 1 plus the full F146/F087/F213 time stamps (for the coverage figure);
- ``class__<CLS>.npz``: improved-noise F146 season 1 for one star per class with a catalog period, its spectrum and truth.

Mount-checked; reads the zips in place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/audit"))
from wg6_profile import read_bands  # noqa: E402
from wg6r_pipeline import season_slice  # noqa: E402
from wg6r_profile import ls_peaks, read_member, resolve_root  # noqa: E402

D = REPO / "reports/2026-11-12_wg6-dataset-scoping/data"


def main(argv=None):
    """CLI: cache the same-star comparison and the per-class examples."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--raw-root", default="/Volumes/Data_Vault/wg6_roman_data_raw")
    ap.add_argument("--star", default="OGLE-BLG-T2CEP-0234")
    args = ap.parse_args(argv)
    root = resolve_root(args.root)
    out = D / "resampled/examples"
    out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(D / "resampled/inventory.csv.gz", low_memory=False)
    truth = pd.read_csv(D / "resampled/truth.csv.gz", low_memory=False)
    P = dict(zip(truth.star_id, truth.P_true))
    tdir = next((root / "simple_noise").glob("drive-download-*"))
    times = {"F146": np.load(tdir / "roman_times_shortcadence.npy"),
             "F087": np.load(tdir / "roman_times_longcadence.npy"),
             "F213": np.load(tdir / "roman_times_longcadence2.npy")}

    # same star in raw + 3 variants
    sid = args.star
    z = {"star_id": sid, "P_true": P.get(sid, np.nan)}
    raw_inv = pd.read_csv(D / f"inventory/{sid.split('-')[2]}_lightcurves.csv.gz")
    rr = raw_inv[raw_inv.NAME == sid].iloc[0]
    b = read_bands(Path(args.raw_root) / rr.relpath)["LC_I"]
    z.update(raw_t=b["t"], raw_y=b["y"].astype(np.float32))
    for v in ("simple", "simple_shifted", "improved"):
        r = inv[(inv.star_id == sid) & (inv.variant == v)].iloc[0]
        bands = read_member(root, r.zip, r.member, times)
        t, y, e = bands["F146"]
        m = season_slice(t, 0)
        z.update({f"{v}_t": t[m], f"{v}_y": y[m].astype(np.float32), f"{v}_e": e[m].astype(np.float32)})
        if v == "improved":
            for bb in ("F146", "F087", "F213"):
                z[f"times_{bb}"] = bands[bb][0]
            z["improved_F087_y"] = bands["F087"][1].astype(np.float32)
            z["improved_F213_y"] = bands["F213"][1].astype(np.float32)
            z["improved_F146_all_y"] = y.astype(np.float32)
    np.savez_compressed(out / "same_star.npz", **z)
    print("cached same_star", sid)

    # one improved star per class (FL: a rotating template at WFmag 19)
    rng = np.random.default_rng(3)
    for cls in ["FL", "ECL", "ELL", "T2CEP", "RRLYR", "DSCT", "HB", "LPV", "CV"]:
        g = inv[(inv.variant == "improved") & (inv.cls == cls)].copy()
        g["P_true"] = g.star_id.map(P)
        if cls == "FL":
            g = g[(g.wfmag == 19) & g.P_true.between(1, 10)]
        elif cls != "CV":
            g = g[g.P_true.notna()]
        r = g.iloc[rng.integers(len(g))]
        t, y, e = read_member(root, r.zip, r.member, times)["F146"]
        f, pw, Ps, _ = ls_peaks(t, y)
        m = season_slice(t, 0)
        k = max(1, len(f) // 4000)
        n = len(f) // k * k
        np.savez_compressed(out / f"class__{cls}.npz", star_id=r.star_id, P_true=r.P_true, t=t[m], y=y[m].astype(np.float32),
                            freq=f[:n].reshape(-1, k).mean(1), power=pw[:n].reshape(-1, k).max(1), P_ls=Ps[0],
                            template=str(r.get("fl_template", "")))
        print("cached", cls, r.star_id, r.P_true, Ps[0])


if __name__ == "__main__":
    main()
