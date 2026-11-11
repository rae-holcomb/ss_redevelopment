"""Draft the P27 real-TESS validation manifest (<= 500 stars) (download plan: download_plan.py).

Inputs (all built by the other scripts in this folder, cached under data/p27_cache/):
  * reports/2026-11-12_validation-set-scoping/data/rotators_all.csv   (build_rotators.py)
  * data/p27_cache/zach_tic.csv                                       (tic_lookup.py on Zach's list)
  * VizieR: Yu+2018 Kepler giants, Ceillier+2017 rotating giants, McQuillan+2014 non-periodic,
    Gentile Fusillo+2021 white dwarfs, Prsa+2022 TESS EBs.

For every selected star it records TIC, source, truth label/period, period method and data set, Tmag,
the TESScut FFI sectors (through the latest sector MAST knows), and which HLSP light-curve products exist
at MAST (TESS-SPOC, QLP, TGLC, CDIPS sector counts).

Output: datasets/proposed/p27_validation_draft.csv. Draft only: Rae freezes it.

Usage: python tools/validation_set/build_manifest.py [--seed 27]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402
from build_rotators import attach_tic  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
REPORT = os.path.join(ROOT, "reports", "2026-11-12_validation-set-scoping")
PBINS = [0.1, 1, 3, 10, 20, 40, 71]
N_ROT_PER_BIN = 48          # 6 bins x 48 = 288 rotators
N_NONROT = {"kepler_rc_quiet": 50, "kepler_dwarf_nodetect": 50, "claytor_scvz_template": 40, "white_dwarf": 22}
N_EB = 50
TMAG_MAX = 14.0
PRODUCTS = ["TESS-SPOC", "QLP", "TGLC", "CDIPS"]


def pick_rotators(rot: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stratify TESS-independent, TIC-matched, gyro-consistent rotators into log-period bins.

    Within each bin, stars are drawn round-robin across sources (clusters, Kepler field) so no single
    cluster dominates; brighter stars are preferred (sorted by Tmag after a random shuffle).
    """
    r = rot[rot["tic"].notna() & rot["tess_independent"] & (rot["Tmag"] <= TMAG_MAX)
            & ~rot["gyro_flag"].isin(["too_slow"])].copy()
    r = r[(r["contratio"].isna()) | (r["contratio"] < 0.2)]
    r["pbin"] = pd.cut(r["prot"], PBINS)
    picks = []
    for _, g in r.groupby("pbin", observed=True):
        g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
        queues = {c: list(sub.sort_values("Tmag").index) for c, sub in g.groupby("cluster")}
        chosen = []
        while len(chosen) < N_ROT_PER_BIN and any(queues.values()):
            for c in list(queues):
                if queues[c] and len(chosen) < N_ROT_PER_BIN:
                    chosen.append(queues[c].pop(0))
        picks.append(g.loc[chosen])
    out = pd.concat(picks)
    return pd.DataFrame({
        "tic": out["tic"].astype("Int64"), "label": "rotator", "source": out["cluster"],
        "truth_period": out["prot"], "period_method": out["period_method"],
        "period_dataset": out["period_dataset"], "period_ref": out["period_ref"],
        "Tmag": out["Tmag"], "Teff": out["Teff"], "logg": out["logg"], "ra": out["ra"], "dec": out["dec"],
        "contratio": out["contratio"], "gyro_flag": out["gyro_flag"]})


def pick_kepler_nonrot(rng: np.random.Generator) -> pd.DataFrame:
    """Kepler red-clump giants with no Ceillier+2017 rotation, and Kepler dwarfs with no McQuillan period."""
    y = vo.vizier_table("J/ApJS/236/42/giants")
    rotg = set(vo.vizier_table("J/A+A/605/A111/table1")["KIC"]) | set(vo.vizier_table("J/A+A/605/A111/table2")["KIC"])
    rc = y[(y["Phase"] == 2) & ~y["KIC"].isin(rotg) & (y["Kpmag"] < 12.5)]
    rc = rc.sample(min(len(rc), 4 * N_NONROT["kepler_rc_quiet"]), random_state=int(rng.integers(1e9)))
    m2 = vo.vizier_table("J/ApJS/211/24/table2", "KIC,Teff,log(g),Mass,Prot,w")
    # Santos+2019 (M/K) and Santos+2021 (F/G) recover periods McQuillan missed: exclude those stars.
    santos = (set(vo.vizier_table("J/ApJS/244/21/table3", "KIC,Prot")["KIC"])
              | set(vo.vizier_table("J/ApJS/255/17/table1", "KIC,Prot")["KIC"]))
    dw = m2[m2["Prot"].isna() & m2["Teff"].between(4000, 6200) & (m2["log(g)"] > 4.2) & ~m2["KIC"].isin(santos)]
    dw = dw.sample(min(len(dw), 6 * N_NONROT["kepler_dwarf_nodetect"]), random_state=int(rng.integers(1e9)))
    cand = pd.concat([
        pd.DataFrame({"kic": rc["KIC"].astype("Int64"), "src": "kepler_rc_quiet"}),
        pd.DataFrame({"kic": dw["KIC"].astype("Int64"), "src": "kepler_dwarf_nodetect"})], ignore_index=True)
    t = attach_tic(cand.assign(prot=np.nan))
    t = t[t["tic"].notna() & (t["Tmag"] <= TMAG_MAX)]
    t = t[(t["contratio"].isna()) | (t["contratio"] < 0.2)]
    out = []
    for src, g in t.groupby("src"):
        g = g.sort_values("Tmag").head(N_NONROT[src])
        lab = ("Yu+2018 HeB (asteroseismic) & not in Ceillier+2017 rotating/polluted"
               if src == "kepler_rc_quiet" else "McQuillan+2014 no period (Kepler Q3-Q14 ACF)")
        out.append(pd.DataFrame({"tic": g["tic"].astype("Int64"), "label": "nonrotator", "source": src,
                                 "truth_period": np.nan, "period_method": "none detected (" + lab + ")",
                                 "period_dataset": "Kepler", "period_ref": lab, "Tmag": g["Tmag"],
                                 "Teff": g["Teff"], "logg": g["logg"], "ra": g["ra"], "dec": g["dec"],
                                 "contratio": g["contratio"], "gyro_flag": "n/a"}))
    return pd.concat(out)


def pick_claytor(rng: np.random.Generator) -> pd.DataFrame:
    """Sample Zach Claytor's SCVZ list (Claytor+2024 noise templates), brightest-first within Tmag bins."""
    z = pd.read_csv(os.path.join(ROOT, "data", "p27_cache", "zach_tic.csv"))
    z = z[(z["Tmag"] <= TMAG_MAX) & ((z["contratio"].isna()) | (z["contratio"] < 0.2))]
    z = z.sample(N_NONROT["claytor_scvz_template"], random_state=int(rng.integers(1e9)))
    return pd.DataFrame({"tic": z["ID"].astype("Int64"), "label": "nonrotator_candidate",
                         "source": "claytor_scvz_template", "truth_period": np.nan,
                         "period_method": "assumed quiet (hot star; Claytor+2024 noise template), unvetted",
                         "period_dataset": "TESS (SCVZ)", "period_ref": "Claytor+2024 (ApJ 962, 47) sec. III.1.2",
                         "Tmag": z["Tmag"], "Teff": z["Teff"], "logg": z["logg"], "ra": z["ra"], "dec": z["dec"],
                         "contratio": z["contratio"], "gyro_flag": "n/a"})


def pick_wd(rng: np.random.Generator) -> pd.DataFrame:
    """Brightest high-probability Gaia white dwarfs (Gentile Fusillo+2021), mapped to TIC by Gaia DR2 ID."""
    wd = vo.vizier_table("J/MNRAS/508/3877/maincat", "GaiaEDR3,GaiaDR2,RA_ICRS,DE_ICRS,Plx,Pwd,Gmag,BPmag,RPmag",
                         extra="&Pwd=%3E0.9&Gmag=%3C14.5")
    wd = wd[wd["GaiaDR2"].notna()].sort_values("Gmag")
    t = attach_tic(pd.DataFrame({"gaia_dr2": wd["GaiaDR2"].astype("Int64"), "prot": np.nan, "cluster": "WD"}))
    t = t[t["tic"].notna() & ((t["contratio"].isna()) | (t["contratio"] < 0.2))].head(N_NONROT["white_dwarf"])
    return pd.DataFrame({"tic": t["tic"].astype("Int64"), "label": "nonrotator_candidate", "source": "white_dwarf",
                         "truth_period": np.nan,
                         "period_method": "assumed no 0.1-180 d signal (WD; rotation is minutes-hours)",
                         "period_dataset": "Gaia EDR3 (selection only)", "period_ref": "Gentile Fusillo+2021",
                         "Tmag": t["Tmag"], "Teff": t["Teff"], "logg": t["logg"], "ra": t["ra"], "dec": t["dec"],
                         "contratio": t["contratio"], "gyro_flag": "n/a"})


def pick_ebs(rng: np.random.Generator) -> pd.DataFrame:
    """Prsa+2022 TESS EBs (contaminant class), stratified in orbital period, Tmag <= TMAG_MAX."""
    e = vo.vizier_table("J/ApJS/258/16/tess-ebs")
    e = e[(e["Tmag"] <= TMAG_MAX) & e["Per"].between(0.1, 180)].copy()
    e["pbin"] = pd.cut(e["Per"], PBINS)
    per = int(np.ceil(N_EB / e["pbin"].nunique()))
    s = (e.groupby("pbin", observed=True, group_keys=False)
         .apply(lambda g: g.sample(min(len(g), per), random_state=int(rng.integers(1e9))))).head(N_EB)
    t = vo.mast_invoke({"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                        "params": {"columns": "ID", "filters": [{"paramName": "ID",
                                                                 "values": [str(int(x)) for x in s["TIC"]]}]},
                        "pagesize": 5000, "page": 1})
    tt = pd.DataFrame(t["data"])[["ID", "Teff", "logg", "contratio"]].rename(columns={"ID": "TIC"})
    s = s.merge(tt, on="TIC", how="left")
    return pd.DataFrame({"tic": s["TIC"].astype("Int64"), "label": "contaminant_eb", "source": "prsa2022_eb",
                         "truth_period": s["Per"], "period_method": "eclipse timing (orbital period)",
                         "period_dataset": "TESS 2-min (S1-26)", "period_ref": "Prsa+2022 (ApJS 258, 16)",
                         "Tmag": s["Tmag"], "Teff": s["Teff"], "logg": s["logg"], "ra": s["RAJ2000"],
                         "dec": s["DEJ2000"], "contratio": s["contratio"], "gyro_flag": "n/a"})


def products(tics) -> pd.DataFrame:
    """Return per-TIC sector counts of each HLSP product in ``PRODUCTS`` from MAST CAOM (cached)."""
    tics = [str(int(t)) for t in tics]
    frames = []
    for i in range(0, len(tics), 100):
        js = vo.mast_invoke({"service": "Mast.Caom.Filtered", "format": "json",
                             "params": {"columns": "target_name,provenance_name,sequence_number",
                                        "filters": [{"paramName": "target_name", "values": tics[i:i + 100]},
                                                    {"paramName": "provenance_name", "values": PRODUCTS}]},
                             "pagesize": 50000, "page": 1})
        frames.append(pd.DataFrame(js["data"]))
    d = pd.concat(frames, ignore_index=True)
    d["tic"] = pd.to_numeric(d["target_name"], errors="coerce")
    c = d.drop_duplicates(["tic", "provenance_name", "sequence_number"]).groupby(
        ["tic", "provenance_name"]).size().unstack(fill_value=0)
    c = c.reindex(columns=PRODUCTS, fill_value=0)
    c.columns = ["n_" + p.lower().replace("-", "_") for p in c.columns]
    return c.reset_index()


def main():
    """Assemble the draft manifest, attach sectors/products, and write the CSV + download plan."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "proposed", "p27_validation_draft.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rot = pd.read_csv(os.path.join(REPORT, "data", "rotators_all.csv"))
    parts = [pick_rotators(rot, rng), pick_kepler_nonrot(rng), pick_claytor(rng), pick_wd(rng), pick_ebs(rng)]
    m = pd.concat(parts, ignore_index=True).drop_duplicates("tic")
    sectors = []
    for _, r in m.iterrows():
        sectors.append(vo.tesscut_sectors(float(r["ra"]), float(r["dec"])))
    m["sectors"] = [" ".join(map(str, s)) for s in sectors]
    m["n_sectors"] = [len(s) for s in sectors]
    m = m.merge(products(m["tic"]), on="tic", how="left").fillna({f"n_{p.lower().replace('-', '_')}": 0
                                                                   for p in PRODUCTS})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    m.to_csv(a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)
    print(m.groupby(["label", "source"]).size())
    # The download plan (real file sizes) is made by download_plan.py from this manifest.


if __name__ == "__main__":
    main()
