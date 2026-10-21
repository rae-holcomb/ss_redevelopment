"""Precompute the small tables the T2 report notebook plots (needs the drive for the catalog).

Writes to ``data/``:

- ``corpus_hists.csv``: histograms of the SMARTS parameters for the whole corpus and the
  SIM_ID >= 800k pool, on the same edges the notebook uses for each set;
- ``set_hists.csv``: the same histograms for each set, unweighted and weighted by
  ``sampling_weight``;
- ``corpus_strata.csv``: pool share per stratum and per (pbin, amp_tier, shear_class);
- ``example_lcs.npz``: raw and loaded light curves (assigned pattern and full year) for a
  few dev_small stars per coverage class, for the example-light-curve figure.

Usage (repo root)::

    python reports/2026-10-15_t2-dev-sets/make_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "audit"))
sys.path.insert(0, str(ROOT / "claude_files"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_dev_sets as bds  # noqa: E402
import coverage as cov  # noqa: E402
import smarts_data as sd  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
HIST_EDGES = {
    "true_period": np.linspace(0, 180, 73),
    "log_period": np.linspace(np.log10(0.1), np.log10(180), 41),
    "amp_proxy": np.geomspace(1e-3, 10, 41),
    "shear": np.linspace(-1, 1, 41),
    "incl": np.linspace(0, np.pi / 2, 31),
    "activity": np.geomspace(0.1, 10, 31),
    "minlat": np.linspace(0, 40, 21),
    "maxlat": np.linspace(5, 80, 26),
}


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column ``name`` of a manifest-style frame (log_period derived)."""
    return np.log10(df["true_period"].to_numpy()) if name == "log_period" else df[name].to_numpy(float)


def hist_rows(df: pd.DataFrame, label: str, weights=None) -> list:
    """Normalised histograms (density per bin, summing to 1) of every HIST_EDGES column."""
    rows = []
    for name, edges in HIST_EDGES.items():
        h, _ = np.histogram(_col(df, name), edges, weights=weights)
        h = h / h.sum()
        rows += [dict(set=label, param=name, lo=lo, hi=hi, frac=f) for lo, hi, f in zip(edges[:-1], edges[1:], h)]
    return rows


def main():
    """Write the report's data tables."""
    root = sd.require_drive(sd.data_root(None))
    OUT.mkdir(exist_ok=True)
    pool, _ = bds.load_pool(root)
    from catalog_to_table import load_catalog
    cat = load_catalog(root)
    ren = {"PERIOD": "true_period", "DIFFROT": "shear", "INCL": "incl", "ACTIVITY": "activity",
           "MINLAT": "minlat", "MAXLAT": "maxlat"}
    cat = cat.rename(columns=ren)
    from catalog_audit import amplitude_proxy
    cat["amp_proxy"] = amplitude_proxy(load_catalog(root))
    pool = pool.rename(columns=ren)
    rows = hist_rows(cat, "corpus (1M)") + hist_rows(pool, "pool (SIM_ID>=800k)")
    pd.DataFrame(rows).to_csv(OUT / "corpus_hists.csv", index=False)

    rows = []
    for name in bds.SET_ORDER:
        m = pd.read_csv(ROOT / "datasets" / f"{name}.csv")
        rows += hist_rows(m, name) + hist_rows(m, f"{name} (weighted)", m.sampling_weight.to_numpy())
    pd.DataFrame(rows).to_csv(OUT / "set_hists.csv", index=False)

    st = pool.groupby(["pbin", "amp_tier", "shear_class"]).size().rename("n_pool").reset_index()
    st["pool_frac"] = st.n_pool / len(pool)
    st.to_csv(OUT / "corpus_strata.csv", index=False)

    # example light curves: two dev_small stars per coverage class (fixed choice: lowest row numbers)
    from astropy.io import fits
    from preprocessing import load_smarts_fits
    m = pd.read_csv(ROOT / "datasets" / "dev_small.csv", dtype={"sectors": str})
    ex = m.sort_values("row").groupby("coverage_class").head(2)
    arrs, meta = {}, []
    for i, r in enumerate(ex.itertuples()):
        p = bds.LOCAL_DIR / r.path
        with fits.open(p) as h:
            arrs[f"raw_t_{i}"] = np.asarray(h["LIGHTCURVE"].data["time"], float)
            arrs[f"raw_f_{i}"] = np.asarray(h["LIGHTCURVE"].data["flux"], float)
        full = load_smarts_fits(p)
        asg = load_smarts_fits(p, sectors=list(cov.parse_pattern(r.sectors)))
        arrs[f"full_t_{i}"], arrs[f"full_f_{i}"] = full.time, full.flux
        arrs[f"asg_t_{i}"], arrs[f"asg_f_{i}"] = asg.time, asg.flux
        meta.append(dict(i=i, star_id=r.star_id, true_period=r.true_period, coverage_class=r.coverage_class,
                         sectors=r.sectors, native_sectors=r.native_sectors, pbin=r.pbin))
    np.savez_compressed(OUT / "example_lcs.npz", **arrs)
    pd.DataFrame(meta).to_csv(OUT / "example_lcs_meta.csv", index=False)
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
