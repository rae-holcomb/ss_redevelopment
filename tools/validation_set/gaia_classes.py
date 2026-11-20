"""Gaia DR3 variability class of every P27 validation star (Rae, 2026-11-19: "for hot stars, use the Gaia classification").

TIC -> Gaia source id (MAST TIC ``GAIA``, DR2; equal to the DR3 id for the vast majority), then the Gaia DR3 classifier
(VizieR I/358/vclassre, best ``Class`` + score) and the main-sequence-oscillator table (I/358/vmsosc: delta Sct / gamma Dor /
SX Phe specific object studies) by source id, from the CfA VizieR mirror. Only stars with a table row are listed; a star
missing from the output is not a Gaia DR3 variable (Gaia's sampling makes that a lower bound on pulsators).

Input: the draft and replacement manifests' TIC IDs (``datasets/proposed``). Output: ``datasets/proposed/p27_v1_gaia_classes.csv``
(tic, gaia_id, gaia_class, gaia_class_score, in_vmsosc). Network, metadata only; caches under ``data/p27_cache``.
Usage: python tools/validation_set/gaia_classes.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "datasets", "proposed", "p27_v1_gaia_classes.csv")


def by_source(table: str, cols: str, ids) -> pd.DataFrame:
    """VizieR rows of ``table`` for Gaia source ids (comma list; a single id needs padding, so ids are chunked >= 2)."""
    ids = [int(i) for i in ids]
    out = []
    for i in range(0, len(ids), 80):
        ch = ids[i:i + 80]
        if len(ch) == 1:
            ch = ch + [1]  # the ASU list syntax needs >= 2 values; 1 is not a Gaia source id
        try:
            out.append(vo.vizier_table(table, cols, extra="&Source=" + ",".join(map(str, ch))))
        except RuntimeError as e:  # vizier_table raises when the answer has no data rows (and on a real outage: see the log)
            print(f"  {table}: chunk {i // 80} gave no rows ({str(e)[-60:]})")
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols.split(","))


def main():
    tics = set(pd.read_csv(os.path.join(ROOT, "datasets", "proposed", "p27_validation_draft.csv"))["tic"])
    rep = os.path.join(ROOT, "datasets", "proposed", "p27_v1_replacements.csv")
    if os.path.exists(rep):
        tics |= set(pd.read_csv(rep)["tic"])
    tics = sorted(int(t) for t in tics)
    frames = []
    for i in range(0, len(tics), 400):
        js = vo.mast_invoke({"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                             "params": {"columns": "ID,GAIA", "filters": [{"paramName": "ID",
                                                                           "values": [str(t) for t in tics[i:i + 400]]}]},
                             "pagesize": 5000, "page": 1})
        frames.append(pd.DataFrame(js["data"]))
    t = pd.concat(frames, ignore_index=True)[["ID", "GAIA"]].rename(columns={"ID": "tic", "GAIA": "gaia_id"})
    t["tic"] = t["tic"].astype("int64")
    t = t[pd.to_numeric(t["gaia_id"], errors="coerce").notna()].copy()
    t["gaia_id"] = pd.to_numeric(t["gaia_id"]).astype("int64")
    print(len(tics), "TICs;", len(t), "with a Gaia id")
    cl = by_source("I/358/vclassre", "Source,Class,ClassSc", t["gaia_id"])
    ms = by_source("I/358/vmsosc", "Source,Freq1", t["gaia_id"])
    cl = cl.rename(columns={"Source": "gaia_id", "Class": "gaia_class", "ClassSc": "gaia_class_score"}).drop_duplicates("gaia_id")
    ms = ms.assign(in_vmsosc=1).rename(columns={"Source": "gaia_id"})[["gaia_id", "in_vmsosc"]].drop_duplicates("gaia_id")
    m = t.merge(cl, on="gaia_id", how="left").merge(ms, on="gaia_id", how="left")
    m["in_vmsosc"] = m["in_vmsosc"].fillna(0).astype(int)
    m = m[m["gaia_class"].notna() | (m["in_vmsosc"] > 0)].sort_values("tic")
    m.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    print(m["gaia_class"].value_counts(dropna=False).to_string(), "\nin_vmsosc:", int(m["in_vmsosc"].sum()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
