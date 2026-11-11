"""Download plan for the P27 draft manifest: real file sizes from MAST product metadata (no downloads).

For a random sample of manifest stars, lists the HLSP light-curve observations (TESS-SPOC, QLP, TGLC, CDIPS),
asks MAST for their product lists, and records the size of each science light-curve file by sector. The
per-product median size per cadence era x the manifest's file counts gives the plan.
Writes reports/2026-11-12_validation-set-scoping/data/{lc_file_sizes.csv,download_plan.csv}.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
D = os.path.join(ROOT, "reports", "2026-11-12_validation-set-scoping", "data")
PRODUCTS = ["TESS-SPOC", "QLP", "TGLC", "CDIPS"]


def era(s):
    """FFI cadence era of sector ``s``."""
    return "S1-26 (30 min)" if s <= 26 else ("S27-55 (10 min)" if s <= 55 else "S56+ (200 s)")


def main(n_sample: int = 40, seed: int = 27):
    """Sample stars, fetch product sizes, write the size table and the plan."""
    m = pd.read_csv(os.path.join(ROOT, "datasets", "proposed", "p27_validation_draft.csv"))
    tics = [str(t) for t in m["tic"].sample(n_sample, random_state=seed)]
    js = vo.mast_invoke({"service": "Mast.Caom.Filtered", "format": "json",
                         "params": {"columns": "obsid,target_name,provenance_name,sequence_number",
                                    "filters": [{"paramName": "target_name", "values": tics},
                                                {"paramName": "provenance_name", "values": PRODUCTS}]},
                         "pagesize": 50000, "page": 1})
    obs = pd.DataFrame(js["data"])
    rows = []
    for i in range(0, len(obs), 50):
        sub = obs.iloc[i:i + 50]
        pj = vo.mast_invoke({"service": "Mast.Caom.Products", "format": "json",
                             "params": {"obsid": ",".join(str(o) for o in sub["obsid"])}, "pagesize": 50000, "page": 1})
        p = pd.DataFrame(pj["data"])
        if len(p) == 0:
            continue
        p = p[p["productType"] == "SCIENCE"]
        p = p[p["productFilename"].str.contains("llc|lc|cdips", case=False)]
        p = p.merge(sub[["obsid", "provenance_name", "sequence_number"]].astype({"obsid": str}),
                    left_on=p["parent_obsid"].astype(str), right_on="obsid", how="left", suffixes=("", "_o"))
        rows.append(p[["provenance_name", "sequence_number", "productFilename", "size"]])
    sz = pd.concat(rows, ignore_index=True).drop_duplicates("productFilename")
    sz["era"] = sz["sequence_number"].astype(int).map(era)
    sz["mb"] = sz["size"] / 1e6
    sz.to_csv(os.path.join(D, "lc_file_sizes.csv"), index=False)
    med = sz.groupby(["provenance_name", "era"])["mb"].median().unstack()
    print(med.round(2))
    # Plan: files per product from the manifest counts, size = median over eras observed (conservative: mean of eras)
    plan = []
    for prod in PRODUCTS:
        col = "n_" + prod.lower().replace("-", "_")
        n_files = int(m[col].sum())
        mb = float(np.nanmean(med.loc[prod].values)) if prod in med.index else np.nan
        plan.append(dict(product=prod, stars_with_any=int((m[col] > 0).sum()), files=n_files,
                         median_mb_per_file=round(mb, 2), est_gb=round(n_files * mb / 1e3, 2)))
    # TESScut: 10x10 px cutout per sector, ~4 float32 arrays (flux, flux_err, bkg, ...) per cadence + overhead
    secs = [int(s) for ss in m["sectors"].dropna() for s in str(ss).split()]
    ncad = {"S1-26 (30 min)": 1300, "S27-55 (10 min)": 3900, "S56+ (200 s)": 11700}
    cut_mb = sum(ncad[era(s)] * 100 * 4 * 4 / 1e6 + 0.1 for s in secs)
    plan.append(dict(product="TESScut 10x10 cutouts", stars_with_any=int((m["n_sectors"] > 0).sum()), files=len(secs),
                     median_mb_per_file=round(cut_mb / len(secs), 2), est_gb=round(cut_mb / 1e3, 1)))
    out = pd.DataFrame(plan)
    out.to_csv(os.path.join(D, "download_plan.csv"), index=False)
    print(out.to_string())
    print(pd.Series([era(s) for s in secs]).value_counts())


if __name__ == "__main__":
    main()
