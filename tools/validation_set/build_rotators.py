"""Build the P27 known-rotator source table: cluster members + Kepler-field periods.

Pulls published rotation-period catalogues from VizieR (CfA mirror), keeps
members with a period, records the period's *method* and *data set*, maps each
star to a TIC ID (Gaia DR2 source_id -> TIC ``GAIA`` column; KIC -> TIC ``KIC``),
and attaches TIC Tmag / Gaia colours. Also computes the Mamajek & Hillenbrand
(2008) gyrochronology period for each star and an outlier flag.

Output: ``reports/2026-11-12_validation-set-scoping/data/rotators_all.csv``.

Usage: python tools/validation_set/build_rotators.py [--kepler-sample N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "2026-11-12_validation-set-scoping")

# Cluster metadata: age (Myr) and E(B-V) are representative literature values, used only for
# the gyro sanity check. Method / data-set strings describe how the *published* periods were made.
CLUSTERS = {
    "Pleiades": dict(age=125, ebv=0.04, ref="Rebull+2016 (AJ 152, 113) via Godoy-Rivera+2021",
                     method="spot modulation; Lomb-Scargle (+ by-eye vetting)", dataset="K2 C4"),
    "Blanco1": dict(age=115, ebv=0.01, ref="Gillen+2020 (MNRAS 492, 1008)",
                    method="spot modulation; GP / Lomb-Scargle / G-ACF (adopted)", dataset="NGTS (ground)"),
    "NGC2516": dict(age=150, ebv=0.12, ref="Fritzewski+2020 (A&A 641, A51) via Godoy-Rivera+2021",
                    method="spot modulation; periodogram", dataset="CTIO 4m Blanco (ground)"),
    "NGC2547": dict(age=38, ebv=0.06, ref="Irwin+2008 (MNRAS 383, 1588) via Godoy-Rivera+2021",
                    method="spot modulation; least-squares sine fit", dataset="CTIO 1m (ground)"),
    "NGC3532": dict(age=300, ebv=0.04, ref="Fritzewski+2021 (A&A 652, A60)",
                    method="spot modulation; periodogram", dataset="CTIO 1m (ground)"),
    "Praesepe": dict(age=670, ebv=0.027, ref="Douglas+2019 (ApJ 879, 100)",
                     method="spot modulation; Lomb-Scargle (+ by-eye vetting)", dataset="K2 C5 (+PTF/SuperWASP fill-in)"),
    "Hyades": dict(age=650, ebv=0.0, ref="Douglas+2019 (ApJ 879, 100) + compiled ground periods",
                   method="spot modulation; Lomb-Scargle (+ by-eye vetting)", dataset="K2 C4/C13"),
    "NGC6811": dict(age=1000, ebv=0.07, ref="Curtis+2019 (ApJ 879, 49) via Godoy-Rivera+2021",
                    method="spot modulation; Lomb-Scargle", dataset="Kepler"),
    "Rup147": dict(age=2700, ebv=0.11, ref="Curtis+2020 (ApJ 904, 140)",
                   method="spot modulation; Lomb-Scargle (+ by-eye vetting)", dataset="K2 C7 (+PTF)"),
    "PscEri": dict(age=120, ebv=0.0, ref="Curtis+2019 (AJ 158, 77)",
                   method="spot modulation; Lomb-Scargle", dataset="TESS FFI"),
}
KEPLER = dict(age=np.nan, ebv=0.0, ref="McQuillan+2014 (ApJS 211, 24)",
              method="spot modulation; ACF", dataset="Kepler Q3-Q14")


def load_clusters() -> pd.DataFrame:
    """Return one row per cluster star with a published period: cluster, gaia_dr2, prot, period_ref, flags."""
    rows = []
    gr = vo.vizier_table("J/ApJS/257/46/tablef1")
    gr = gr[gr["Class"].isin(["probable-member", "possible-member"]) & gr["GaiaDR2"].notna()]
    for cl in ["Pleiades", "NGC2516", "NGC2547", "NGC6811"]:
        sub = gr[gr["Cluster"] == cl]
        rows.append(pd.DataFrame({"cluster": cl, "gaia_dr2": sub["GaiaDR2"].astype("int64"),
                                  "prot": sub["Period"], "member_class": sub["Class"]}))
    # Blanco 1 (adopted period; all are NGTS-selected members)
    b = vo.vizier_table("J/MNRAS/492/1008/table1")
    b = b[b["Padop"].notna() & b["GaiaDR2"].notna()]
    rows.append(pd.DataFrame({"cluster": "Blanco1", "gaia_dr2": b["GaiaDR2"].astype("int64"),
                              "prot": b["Padop"], "member_class": "member (Gillen+2020)"}))
    # NGC 3532
    n = vo.vizier_table("J/A+A/652/A60/table2")
    n = n[n["Prot"].notna() & n["GaiaDR2"].notna()]
    rows.append(pd.DataFrame({"cluster": "NGC3532", "gaia_dr2": n["GaiaDR2"].astype("int64"),
                              "prot": n["Prot"], "member_class": "member (" + n["Class"].astype(str) + ")"}))
    # Hyades (Douglas+2019 table3): column P says which catalogue's period was adopted (decoded from which
    # period column is populated for each code): 3 -> Prot1 (K2 C13, this work), 2 -> Prot4 (K2 C4, Douglas+2016),
    # D -> SuperWASP (Delorme+2011), H -> HATNet (Hartman+2011), K -> ASAS (Kiraga 2012), P -> Prosser+1995,
    # R -> Radick+1987/1995 (Lowell). All ground codes are photometric spot-modulation periods.
    h = vo.vizier_table("J/ApJ/879/100/table3")
    code_col = {"3": "Prot1", "2": "Prot4", "D": "ProtD", "H": "ProtH", "K": "ProtA", "P": "ProtP", "R": "ProtR"}
    code_ds = {"3": "K2 C13", "2": "K2 C4", "D": "SuperWASP (ground)", "H": "HATNet (ground)",
               "K": "ASAS (ground)", "P": "Prosser+1995 (ground)", "R": "Lowell / Radick (ground)"}
    h = h[h["Gaia"].notna() & h["P"].astype(str).isin(list(code_col))].copy()
    h["prot"] = [r[code_col[str(r["P"])]] for _, r in h.iterrows()]
    h = h[h["prot"].notna() & (h["Mm-Gaia"].astype(str).str.strip() == "True")]
    rows.append(pd.DataFrame({"cluster": "Hyades", "gaia_dr2": h["Gaia"].astype("int64"), "prot": h["prot"],
                              "member_class": "member (Douglas+2019 Gaia)",
                              "period_dataset_star": h["P"].astype(str).map(code_ds)}))
    # Praesepe (Douglas+2019 table4): adopted Prot; mostly K2 C5 (Douglas+2017) with PTF/SuperWASP/HATNet fill-in.
    pr = vo.vizier_table("J/ApJ/879/100/table4")
    pr = pr[pr["Gaia"].notna() & pr["Prot"].notna()]
    rows.append(pd.DataFrame({"cluster": "Praesepe", "gaia_dr2": pr["Gaia"].astype("int64"), "prot": pr["Prot"],
                              "member_class": "member (Douglas+2019)"}))
    # Ruprecht 147 (Cl == 0 rows are R147; Cl == 1 rows are the comparison clusters)
    r = vo.vizier_table("J/ApJ/904/140/table1")
    r = r[(r["Cl"] == 0) & r["Prot"].notna() & r["GaiaDR2"].notna()]
    rows.append(pd.DataFrame({"cluster": "Rup147", "gaia_dr2": r["GaiaDR2"].astype("int64"), "prot": r["Prot"],
                              "member_class": "member (" + r["Member"].astype(str) + ")"}))
    # Psc-Eri (TESS-derived; kept, but flagged as not TESS-independent)
    p = vo.vizier_table("J/AJ/158/77/table2")
    p = p[p["Prot"].notna()]
    rows.append(pd.DataFrame({"cluster": "PscEri", "gaia_dr2": p["Source"].astype("int64"), "prot": p["Prot"],
                              "member_class": "member (Curtis+2019)"}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df["prot"] > 0]  # Rup147 table uses negative values as flags
    for k in ("ref", "method", "dataset", "age", "ebv"):
        df["period_" + k if k in ("ref", "method", "dataset") else k] = df["cluster"].map(
            {c: v[k] for c, v in CLUSTERS.items()})
    star_ds = df.pop("period_dataset_star")
    df["period_dataset"] = star_ds.fillna(df["period_dataset"])
    return df


def load_kepler(n_sample: int, seed: int = 27) -> pd.DataFrame:
    """Return a random sample of McQuillan+2014 periodic Kepler stars, stratified in log period."""
    m = vo.vizier_table("J/ApJS/211/24/table1")
    m = m[m["Prot"].notna()].copy()
    bins = np.array([0.1, 1, 3, 10, 20, 40, 71])
    m["pbin"] = pd.cut(m["Prot"], bins)
    per = max(1, n_sample // (len(bins) - 1))
    s = (m.groupby("pbin", observed=True, group_keys=False)
         .apply(lambda g: g.sample(min(len(g), per), random_state=seed)))
    out = pd.DataFrame({"cluster": "Kepler-field", "kic": s["KIC"].astype("int64"), "prot": s["Prot"],
                        "member_class": "field (McQuillan+2014 periodic)",
                        "period_ref": KEPLER["ref"], "period_method": KEPLER["method"],
                        "period_dataset": KEPLER["dataset"], "age": np.nan, "ebv": 0.0,
                        "teff_lit": s["Teff"], "logg_lit": s["log(g)"]})
    return out


def attach_tic(df: pd.DataFrame) -> pd.DataFrame:
    """Add TIC columns by Gaia DR2 ID (TIC v8.2 ``GAIA``) or KIC; unmatched stars keep NaN TIC."""
    from tic_lookup import COLS  # noqa: WPS433

    def by(param, values):
        frames = []
        values = [str(int(v)) for v in values]
        for i in range(0, len(values), 400):
            js = vo.mast_invoke({"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                                 "params": {"columns": COLS, "filters": [{"paramName": param,
                                                                          "values": values[i:i + 400]}]},
                                 "pagesize": 5000, "page": 1})
            frames.append(pd.DataFrame(js["data"]))
        return pd.concat(frames, ignore_index=True)

    out = []
    g = df[df.get("gaia_dr2").notna()] if "gaia_dr2" in df else df.iloc[:0]
    if len(g):
        t = by("GAIA", g["gaia_dr2"].unique())
        t["GAIA"] = pd.to_numeric(t["GAIA"], errors="coerce").astype("Int64")
        # keep the primary TIC entry (drop duplicates / artifacts)
        t = t[t["disposition"].isna() | (t["disposition"] == "")].drop_duplicates("GAIA")
        out.append(g.merge(t, left_on="gaia_dr2", right_on="GAIA", how="left"))
    k = df[df["kic"].notna()] if "kic" in df else df.iloc[:0]
    if len(k):
        t = by("KIC", k["kic"].unique())
        t["KIC"] = pd.to_numeric(t["KIC"], errors="coerce").astype("Int64")
        t = t[t["disposition"].isna() | (t["disposition"] == "")].drop_duplicates("KIC")
        out.append(k.merge(t, left_on="kic", right_on="KIC", how="left", suffixes=("", "_tic")))
    res = pd.concat(out, ignore_index=True).rename(columns={"ID": "tic"})
    return res


def bv_from_bprp(bprp0: np.ndarray) -> np.ndarray:
    """(B-V)0 from (BP-RP)0 for dwarfs, interpolated in Mamajek's dwarf colour table (subset below).

    Values (BP-RP, B-V) transcribed from E. Mamajek, "A Modern Mean Dwarf Stellar Color and Effective
    Temperature Sequence" (EEM_dwarf_UBVIJHK_colors_Teff.txt, v2022.04.16) for F5V-M2V.
    """
    tab = np.array([
        [0.587, 0.440], [0.627, 0.480], [0.674, 0.530], [0.731, 0.590], [0.754, 0.610],
        [0.784, 0.640], [0.823, 0.680], [0.858, 0.720], [0.921, 0.790], [0.977, 0.840],
        [1.044, 0.910], [1.084, 0.950], [1.139, 1.000], [1.213, 1.060], [1.345, 1.150],
        [1.511, 1.260], [1.705, 1.350], [1.837, 1.400], [1.992, 1.430], [2.137, 1.460],
    ])
    return np.interp(bprp0, tab[:, 0], tab[:, 1], left=np.nan, right=np.nan)


def gyro_mh08(bv0: np.ndarray, age_myr: np.ndarray) -> np.ndarray:
    """Mamajek & Hillenbrand (2008) gyro period: P = 0.407 (B-V - 0.495)^0.325 t^0.566 (t in Myr, P in d)."""
    x = np.clip(bv0 - 0.495, 1e-6, None)
    p = 0.407 * x ** 0.325 * age_myr ** 0.566
    return np.where(bv0 > 0.5, p, np.nan)


def flag_gyro(df: pd.DataFrame) -> pd.DataFrame:
    """Add (BP-RP)0, P_gyro (MH08) and ``gyro_flag``.

    * ``slow_seq``: 0.6 < P/P_gyro < 1.6 (on the gyro sequence);
    * ``fast``: P/P_gyro <= 0.6 (unconverged fast rotator; real in young clusters, suspicious in old ones);
    * ``too_slow``: P/P_gyro >= 1.6 (likely 2P alias, blend or non-member);
    * ``out_of_range``: outside the 0.5 < (B-V)0 < 1.4 calibration range (no gyro check).
    """
    bprp = df["gaiabp"] - df["gaiarp"]
    df["bprp0"] = bprp - 1.339 * df["ebv"].fillna(0)
    df["bv0"] = bv_from_bprp(df["bprp0"].to_numpy())
    df["p_gyro"] = gyro_mh08(df["bv0"].to_numpy(), df["age"].to_numpy())
    ratio = df["prot"] / df["p_gyro"]
    df["p_over_gyro"] = ratio
    flag = np.select([ratio.isna(), ratio <= 0.6, ratio >= 1.6], ["out_of_range", "fast", "too_slow"], "slow_seq")
    df["gyro_flag"] = np.where(df["cluster"] == "Kepler-field", "field", flag)
    return df


def main():
    """Build and save the rotator source table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kepler-sample", type=int, default=600, help="McQuillan stars to draw (stratified)")
    ap.add_argument("--out", default=os.path.join(REPORT, "data", "rotators_all.csv"))
    a = ap.parse_args()
    cl = load_clusters()
    kp = load_kepler(a.kepler_sample)
    df = attach_tic(pd.concat([cl, kp], ignore_index=True))
    df = flag_gyro(df)
    df["tess_independent"] = ~df["period_dataset"].str.contains("TESS")
    keep = ["tic", "cluster", "gaia_dr2", "kic", "member_class", "prot", "period_method", "period_dataset",
            "period_ref", "tess_independent", "age", "ebv", "ra", "dec", "Tmag", "Teff", "logg", "GAIAmag",
            "gaiabp", "gaiarp", "bprp0", "bv0", "p_gyro", "p_over_gyro", "gyro_flag", "contratio", "lumclass", "d"]
    df = df[[c for c in keep if c in df.columns]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)
    print(df.groupby("cluster").agg(n=("prot", "size"), tic=("tic", lambda s: s.notna().sum()),
                                    tmag_med=("Tmag", "median"), pmin=("prot", "min"), pmax=("prot", "max")))
    print(df.groupby(["cluster", "gyro_flag"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
