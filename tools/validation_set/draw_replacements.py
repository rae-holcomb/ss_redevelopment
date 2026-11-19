"""P27 v1: draw replacements for the unscorable and crowded stars of the draft (Rae, 2026-11-19; network, metadata only).

Removes from ``datasets/proposed/p27_validation_draft.csv``
  * every star with no TESS-SPOC *and* no QLP light curve at MAST (72), and
  * every EB with TIC ``contratio`` >= 0.2 (7; "ditch the crowded EBs"),
and draws one replacement per removed star from the same pool, keeping each (label, source, truth-period bin) count.
A replacement must have >= 1 TESS-SPOC or QLP file at MAST (``build_manifest.products``) and, for every block except the EBs'
own pool, satisfies the draft's cuts (Tmag <= 14, ``contratio`` < 0.2); EB replacements also need ``contratio`` < 0.2.

Pools and order (all seeded, ``--seed``):
  rotators        ``rotators_all.csv`` (committed), the draft's filters, per truth-period bin; clusters round-robin, brightest first
  kepler_*        fresh random Kepler samples (4x/6x the draft's, 2x for slack), brightest first
  claytor         ``zach_list_tic.csv.gz`` minus the 294 Gaia DR3 variables, random order
  white_dwarf     Gentile Fusillo+2021 by Gmag (next brightest unused)
  prsa EB         Prsa+2022 in the removed star's period bin, random order

Output: ``datasets/proposed/p27_v1_replacements.csv`` = the draft's columns for each replacement plus ``replaces`` (the removed
TIC). ``tools/build_tess_validation.py --version v1`` applies it. Needs the network; caches under ``data/p27_cache``.

Usage: python tools/validation_set/draw_replacements.py [--seed 271]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402
import build_manifest as bm  # noqa: E402
from build_rotators import attach_tic  # noqa: E402

ROOT = bm.ROOT
REPORT = bm.REPORT
DRAFT = os.path.join(ROOT, "datasets", "proposed", "p27_validation_draft.csv")
OUT = os.path.join(ROOT, "datasets", "proposed", "p27_v1_replacements.csv")
PB_LABELS = ["0.1-1", "1-3", "3-10", "10-20", "20-40", "40-71"]


def products_safe(tics) -> pd.DataFrame:
    """``build_manifest.products`` that tolerates a batch with no HLSP rows at all (all-zero counts)."""
    tics = [int(t) for t in tics]
    try:
        return bm.products(tics)
    except KeyError:
        return pd.DataFrame({"tic": tics, **{f"n_{p.lower().replace('-', '_')}": 0 for p in bm.PRODUCTS}})


def pbin(p):
    return pd.cut(pd.Series(p), bm.PBINS, labels=PB_LABELS).astype(object).where(pd.notna(p), "")


def scorable_filter(cand: pd.DataFrame, need: int, used: set, batch: int = 60) -> pd.DataFrame:
    """Walk ``cand`` in order, check MAST products batch by batch, return the first ``need`` with a SPOC or QLP file."""
    got = []
    cand = cand[~cand["tic"].isin(used)].drop_duplicates("tic")
    for i in range(0, len(cand), batch):
        sub = cand.iloc[i:i + batch]
        pr = products_safe(sub["tic"])
        pr = pr.assign(tic=pr["tic"].astype("int64")).assign(ok=(pr["n_tess_spoc"] + pr["n_qlp"]) > 0).set_index("tic")["ok"]
        for r in sub.itertuples():
            if pr.get(int(r.tic), False):
                got.append(r.Index)
                if len(got) == need:
                    return cand.loc[got]
    return cand.loc[got]


def extra_kepler(b: str, need: int, used: set, have_kic: set, n_pool: int = 600, seed: int = 2719) -> pd.DataFrame:
    """More McQuillan+2014 periodic Kepler stars in period bin ``b`` (not in rotators_all), brightest first, with products."""
    lo, hi = {k: v for k, v in zip(PB_LABELS, zip(bm.PBINS[:-1], bm.PBINS[1:]))}[b]
    m = vo.vizier_table("J/ApJS/211/24/table1")
    m = m[m["Prot"].notna() & (m["Prot"] > lo) & (m["Prot"] <= hi) & ~m["KIC"].isin(have_kic)]
    m = m.sample(min(len(m), n_pool), random_state=seed)
    t = attach_tic(pd.DataFrame({"kic": m["KIC"].astype("Int64"), "prot": m["Prot"].to_numpy()}))
    t = t[t["tic"].notna() & (t["Tmag"] <= bm.TMAG_MAX)]
    t = t[(t["contratio"].isna()) | (t["contratio"] < 0.2)].sort_values("Tmag")
    t["tic"] = t["tic"].astype("int64")
    sel = scorable_filter(t, need, used)
    used |= set(sel["tic"])
    return sel.assign(cluster="Kepler-field", period_method="spot modulation; ACF", period_dataset="Kepler Q3-Q14",
                      period_ref="McQuillan+2014 (ApJS 211, 24)", gyro_flag="field")


def draw_rotators(need_by_bin: dict, used: set) -> pd.DataFrame:
    rot = pd.read_csv(os.path.join(REPORT, "data", "rotators_all.csv"))
    r = rot[rot["tic"].notna() & rot["tess_independent"] & (rot["Tmag"] <= bm.TMAG_MAX) & ~rot["gyro_flag"].isin(["too_slow"])].copy()
    r = r[(r["contratio"].isna()) | (r["contratio"] < 0.2)]
    r["tic"] = r["tic"].astype("int64")
    r["pbin"] = pbin(r["prot"])
    out = []
    for b, n in need_by_bin.items():
        g = r[(r["pbin"] == b) & ~r["tic"].isin(used)]
        queues = {c: list(sub.sort_values("Tmag").index) for c, sub in g.groupby("cluster")}
        order = []
        while any(queues.values()):
            for c in list(queues):
                if queues[c]:
                    order.append(queues[c].pop(0))
        cand = g.loc[order]
        sel = scorable_filter(cand, n, used)
        used |= set(sel["tic"])
        out.append(sel)
        if len(sel) < n:  # long-period bins: the 600-star McQuillan sample of rotators_all is exhausted; draw more
            ext = extra_kepler(b, n - len(sel), used, set(rot["kic"].dropna().astype("int64")))
            print(f"  bin {b}: {len(sel)}/{n} from rotators_all, {len(ext)} more from McQuillan+2014")
            out.append(ext)
    out = pd.concat(out)
    return pd.DataFrame({
        "tic": out["tic"], "label": "rotator", "source": out["cluster"], "truth_period": out["prot"],
        "period_method": out["period_method"], "period_dataset": out["period_dataset"], "period_ref": out["period_ref"],
        "Tmag": out["Tmag"], "Teff": out["Teff"], "logg": out["logg"], "ra": out["ra"], "dec": out["dec"],
        "contratio": out["contratio"], "gyro_flag": out["gyro_flag"]})


def draw_kepler(src: str, need: int, used: set, rng) -> pd.DataFrame:
    n_pool = {"kepler_rc_quiet": 4 * 50 * 3, "kepler_dwarf_nodetect": 6 * 50 * 3}[src]
    y = vo.vizier_table("J/ApJS/236/42/giants")
    if src == "kepler_rc_quiet":
        rotg = set(vo.vizier_table("J/A+A/605/A111/table1")["KIC"]) | set(vo.vizier_table("J/A+A/605/A111/table2")["KIC"])
        pool = y[(y["Phase"] == 2) & ~y["KIC"].isin(rotg) & (y["Kpmag"] < 12.5)]
        lab = "Yu+2018 HeB (asteroseismic) & not in Ceillier+2017 rotating/polluted"
    else:
        m2 = vo.vizier_table("J/ApJS/211/24/table2", "KIC,Teff,log(g),Mass,Prot,w")
        santos = (set(vo.vizier_table("J/ApJS/244/21/table3", "KIC,Prot")["KIC"])
                  | set(vo.vizier_table("J/ApJS/255/17/table1", "KIC,Prot")["KIC"]))
        pool = m2[m2["Prot"].isna() & m2["Teff"].between(4000, 6200) & (m2["log(g)"] > 4.2) & ~m2["KIC"].isin(santos)]
        lab = "McQuillan+2014 no period (Kepler Q3-Q14 ACF)"
    pool = pool.sample(min(len(pool), n_pool), random_state=int(rng.integers(1e9)))
    t = attach_tic(pd.DataFrame({"kic": pool["KIC"].astype("Int64"), "src": src, "prot": np.nan}))
    t = t[t["tic"].notna() & (t["Tmag"] <= bm.TMAG_MAX)]
    t = t[(t["contratio"].isna()) | (t["contratio"] < 0.2)].sort_values("Tmag")
    t["tic"] = t["tic"].astype("int64")
    sel = scorable_filter(t, need, used)
    used |= set(sel["tic"])
    return pd.DataFrame({"tic": sel["tic"], "label": "nonrotator", "source": src, "truth_period": np.nan,
                         "period_method": "none detected (" + lab + ")", "period_dataset": "Kepler", "period_ref": lab,
                         "Tmag": sel["Tmag"], "Teff": sel["Teff"], "logg": sel["logg"], "ra": sel["ra"], "dec": sel["dec"],
                         "contratio": sel["contratio"], "gyro_flag": "n/a"})


def draw_claytor(need: int, used: set, rng) -> pd.DataFrame:
    z = pd.read_csv(os.path.join(REPORT, "data", "zach_list_tic.csv.gz"))
    z = z[(z["Tmag"] <= bm.TMAG_MAX) & ((z["contratio"].isna()) | (z["contratio"] < 0.2))]
    var = pd.read_csv(os.path.join(REPORT, "data", "zach_gaia_variability.csv.gz"))
    known = set(var.loc[var.filter(like="in_").sum(axis=1) > 0, "ID"])
    z = z[~z["ID"].isin(known)].rename(columns={"ID": "tic"})
    z = z.sample(frac=1, random_state=int(rng.integers(1e9)))
    sel = scorable_filter(z, need, used)
    used |= set(sel["tic"])
    return pd.DataFrame({"tic": sel["tic"], "label": "expected_undetectable", "source": "claytor_scvz_template",
                         "truth_period": np.nan,
                         "period_method": "expected undetectable (5800-6000 K dwarf; Claytor+2024 noise template; not in Gaia DR3 variability tables), otherwise unvetted",
                         "period_dataset": "TESS (SCVZ)", "period_ref": "Claytor+2024 (ApJ 962, 47) sec. III.1.2",
                         "Tmag": sel["Tmag"], "Teff": sel["Teff"], "logg": sel["logg"], "ra": sel["ra"], "dec": sel["dec"],
                         "contratio": sel["contratio"], "gyro_flag": "n/a"})


def draw_wd(need: int, used: set) -> pd.DataFrame:
    wd = vo.vizier_table("J/MNRAS/508/3877/maincat", "GaiaEDR3,GaiaDR2,RA_ICRS,DE_ICRS,Plx,Pwd,Gmag,BPmag,RPmag",
                         extra="&Pwd=%3E0.9&Gmag=%3C14.5")
    wd = wd[wd["GaiaDR2"].notna()].sort_values("Gmag")
    t = attach_tic(pd.DataFrame({"gaia_dr2": wd["GaiaDR2"].astype("Int64").head(400), "prot": np.nan, "cluster": "WD"}))
    t = t[t["tic"].notna() & ((t["contratio"].isna()) | (t["contratio"] < 0.2))].sort_values("Tmag")
    t["tic"] = t["tic"].astype("int64")
    sel = scorable_filter(t, need, used)
    used |= set(sel["tic"])
    return pd.DataFrame({"tic": sel["tic"], "label": "nonrotator_candidate", "source": "white_dwarf", "truth_period": np.nan,
                         "period_method": "assumed no 0.1-180 d signal (WD; rotation is minutes-hours)",
                         "period_dataset": "Gaia EDR3 (selection only)", "period_ref": "Gentile Fusillo+2021",
                         "Tmag": sel["Tmag"], "Teff": sel["Teff"], "logg": sel["logg"], "ra": sel["ra"], "dec": sel["dec"],
                         "contratio": sel["contratio"], "gyro_flag": "n/a"})


def draw_ebs(need_by_bin: dict, used: set, rng) -> pd.DataFrame:
    e = vo.vizier_table("J/ApJS/258/16/tess-ebs")
    e = e[(e["Tmag"] <= bm.TMAG_MAX) & e["Per"].between(0.1, 71)].copy()
    e["pbin"] = pbin(e["Per"])
    e = e[~e["TIC"].isin(used)]
    out = []
    for b, n in need_by_bin.items():
        g = e[e["pbin"] == b].sample(frac=1, random_state=int(rng.integers(1e9))).head(12 * n + 20)
        t = vo.mast_invoke({"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                            "params": {"columns": "ID", "filters": [{"paramName": "ID", "values": [str(int(x)) for x in g["TIC"]]}]},
                            "pagesize": 5000, "page": 1})
        tt = pd.DataFrame(t["data"])[["ID", "Teff", "logg", "contratio"]].rename(columns={"ID": "TIC"})
        g = g.merge(tt, on="TIC", how="left")
        g = g[(g["contratio"].isna()) | (g["contratio"] < 0.2)].rename(columns={"TIC": "tic"})
        g["tic"] = g["tic"].astype("int64")
        sel = scorable_filter(g, n, used)
        used |= set(sel["tic"])
        out.append(sel)
    s = pd.concat(out)
    return pd.DataFrame({"tic": s["tic"], "label": "contaminant_eb", "source": "prsa2022_eb", "truth_period": s["Per"],
                         "period_method": "eclipse timing (orbital period)", "period_dataset": "TESS 2-min (S1-26)",
                         "period_ref": "Prsa+2022 (ApJS 258, 16)", "Tmag": s["Tmag"], "Teff": s["Teff"], "logg": s["logg"],
                         "ra": s["RAJ2000"], "dec": s["DEJ2000"], "contratio": s["contratio"], "gyro_flag": "n/a"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=271)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = pd.read_csv(DRAFT)
    unscor = (d["n_tess_spoc"] == 0) & (d["n_qlp"] == 0)
    crowded = (d["label"] == "contaminant_eb") & (d["contratio"] >= 0.2)
    rem = d[unscor | crowded].copy()
    rem["pbin"] = pbin(rem["truth_period"])
    print("removing", len(rem), "stars:\n", rem.groupby(["label", "source", "pbin"]).size().to_string())
    used = set(d["tic"])
    need = lambda lab, src: rem[(rem["label"] == lab) & (rem["source"] == src)]  # noqa: E731
    parts = []
    rot = rem[rem["label"] == "rotator"]
    parts.append(draw_rotators(rot.groupby("pbin").size().to_dict(), used))
    for src in ("kepler_rc_quiet", "kepler_dwarf_nodetect"):
        n = len(need("nonrotator", src))
        if n:
            parts.append(draw_kepler(src, n, used, rng))
    parts.append(draw_claytor(len(need("expected_undetectable", "claytor_scvz_template")), used, rng))
    parts.append(draw_wd(len(need("nonrotator_candidate", "white_dwarf")), used))
    eb = rem[rem["label"] == "contaminant_eb"]
    parts.append(draw_ebs(eb.groupby("pbin").size().to_dict(), used, rng))
    m = pd.concat(parts, ignore_index=True)
    m["_pb"] = pbin(m["truth_period"])
    grp = lambda df: np.where(df["label"] == "rotator", "*", df["source"])  # noqa: E731  rotators match on bin only
    rem_n = rem.assign(g=grp(rem)).groupby(["label", "g", "pbin"]).size()
    got_n = m.assign(g=grp(m)).groupby(["label", "g", "_pb"]).size().rename_axis(["label", "g", "pbin"])
    short = (rem_n - got_n.reindex(rem_n.index).fillna(0)).loc[lambda x: x > 0]
    if len(short):
        print("SHORTFALL (needed - found):\n", short.to_string())
    m = m.drop(columns="_pb")
    assert len(m) == len(rem), (len(m), len(rem))
    sectors = [vo.tesscut_sectors(float(r.ra), float(r.dec)) for r in m.itertuples()]
    m["sectors"] = [" ".join(map(str, s)) for s in sectors]
    m["n_sectors"] = [len(s) for s in sectors]
    pr = products_safe(m["tic"])
    m = m.merge(pr, on="tic", how="left")
    for c in ("n_tess_spoc", "n_qlp", "n_tglc", "n_cdips"):
        m[c] = m[c].fillna(0).astype(int)
    assert ((m["n_tess_spoc"] + m["n_qlp"]) > 0).all()
    # pair each replacement with a removed star of the same (label, source, pbin), for the audit column `replaces`
    m["pbin"] = pbin(m["truth_period"])
    pool = {k: list(g["tic"]) for k, g in rem.assign(g=grp(rem)).groupby(["label", "g", "pbin"])}
    m["replaces"] = [pool[(r.label, "*" if r.label == "rotator" else r.source, r.pbin)].pop(0) for r in m.itertuples()]
    m = m.drop(columns="pbin")
    m.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)
    print(m.groupby(["label", "source"]).size())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
