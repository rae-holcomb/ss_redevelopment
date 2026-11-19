"""Summarise P27 non-rotator / contaminant source pools into small CSVs for the report.

Writes to reports/2026-11-12_validation-set-scoping/data/:
  * zach_list_tic.csv.gz   slim TIC properties of Zach Claytor's 23,332 TIC IDs
  * nonrotator_pools.csv   pool sizes per source and Tmag limit
  * mcquillan_detfrac.csv  Kepler detection fraction vs Teff (McQuillan+2014)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
D = os.path.join(ROOT, "reports", "2026-11-12_validation-set-scoping", "data")


def main():
    """Build the summary tables."""
    z = pd.read_csv(os.path.join(ROOT, "data", "p27_cache", "zach_tic.csv"), low_memory=False)
    z[["ID", "ra", "dec", "Tmag", "Teff", "logg", "contratio", "lumclass", "GAIA"]].to_csv(
        os.path.join(D, "zach_list_tic.csv.gz"), index=False)
    rows = []
    for tmax in (12, 13, 14, 15):
        rows.append(dict(source="Zach list (Claytor+2024 SCVZ templates)", tmag_max=tmax, n=int((z.Tmag <= tmax).sum())))
    y = vo.vizier_table("J/ApJS/236/42/giants")
    rotg = set(vo.vizier_table("J/A+A/605/A111/table1")["KIC"]) | set(vo.vizier_table("J/A+A/605/A111/table2")["KIC"])
    rc = y[(y.Phase == 2) & ~y.KIC.isin(rotg)]
    for kmax in (11, 12, 13):
        rows.append(dict(source="Kepler HeB giants, no Ceillier rotation (Yu+2018)", tmag_max=kmax,
                         n=int((rc.Kpmag <= kmax).sum()), note="Kp used as proxy for Tmag (giants: T ~ Kp - 0.6)"))
    m2 = vo.vizier_table("J/ApJS/211/24/table2", "KIC,Teff,log(g),Mass,Prot,w")
    nd = m2[m2.Prot.isna() & (m2["log(g)"] > 4.2) & m2.Teff.between(4000, 6200)]
    rows.append(dict(source="Kepler dwarfs, no McQuillan period", tmag_max=np.nan, n=len(nd),
                     note="4000-6200 K, logg>4.2; Kp not in table; before Santos+2019/2021 cleaning"))
    wd = vo.vizier_table("J/MNRAS/508/3877/maincat", "GaiaEDR3,RA_ICRS,DE_ICRS,Plx,Pwd,Gmag,BPmag,RPmag",
                         extra="&Pwd=%3E0.75&Gmag=%3C17")
    for g in (13, 14, 15, 16, 17):
        rows.append(dict(source="Gaia WDs Pwd>0.75 (Gentile Fusillo+2021)", tmag_max=g, n=int((wd.Gmag < g).sum()),
                         note="G used (WD T ~ G)"))
    rc_g = pd.read_csv(os.path.join(D, "gaia_rc_counts.csv"))
    for _, r in rc_g.iterrows():
        rows.append(dict(source=f"Gaia DR3 red-clump box, {r.region}", tmag_max=r.gmax, n=int(r.n),
                         note="-0.5<M_G<1.5, 0.9<BP-RP<1.6, plx/err>10; G limit"))
    pd.DataFrame(rows).to_csv(os.path.join(D, "nonrotator_pools.csv"), index=False)
    m1 = vo.vizier_table("J/ApJS/211/24/table1")
    edges = np.arange(3500, 6800, 250)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = int(m1.Teff.between(lo, hi, inclusive="left").sum())
        b = int(m2.Teff.between(lo, hi, inclusive="left").sum())
        out.append(dict(teff_lo=lo, teff_hi=hi, n_periodic=a, n_nonperiodic=b, frac=a / max(1, a + b)))
    pd.DataFrame(out).to_csv(os.path.join(D, "mcquillan_detfrac.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
