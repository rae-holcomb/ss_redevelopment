"""Build the WG6 truth table: join OGLE OCVS catalog periods (and the flare literature periods) to WG6 star names.

Inputs: ``$WG6_DATA_ROOT/extracted/_truth/`` (from ``tools/fetch_ogle_truth.py``); the Günther+2020/Medina+2020 per-star
CSVs in the report ``data/``; the inventory CSVs.
Output: ``<report>/data/truth_joined.csv.gz`` with columns NAME, P_true, amp, imag, subtype, truth_src, p_convention,
and ``truth_match_rates.csv`` (per class: files, matched, rate).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path("reports/2026-11-12_wg6-dataset-scoping")
# file -> (period col, amp col, I-mag col, convention, source label); whitespace-split token indices
TABLES = {
    "rot/rot.dat": (5, 4, 3, "photometric (spot) period", "OCVS rot (Iwanek+2024)"),
    "ecl/ecl.dat": (3, 5, 1, "orbital period", "OCVS ecl (Soszynski+2016)"),
    "ecl/ell.dat": (3, 5, 1, "orbital period (light curve shows P/2)", "OCVS ell (Soszynski+2016)"),
    "lpv/Miras.dat": (3, 4, 1, "pulsation period", "OCVS lpv Miras (Iwanek+2022)"),
    "hb/hb.dat": (3, 5, 1, "orbital period", "OCVS hb (Wrona+2022)"),
    "cep/cepF.dat": (3, 6, 1, "pulsation period (F)", "OCVS cep"),
    "cep/cep1O.dat": (3, 6, 1, "pulsation period (1O)", "OCVS cep"),
    "cep/cepF1O.dat": (3, 6, 1, "longer pulsation period", "OCVS cep"),
    "cep/cep1O2O.dat": (3, 6, 1, "longer pulsation period", "OCVS cep"),
    "cep/cep1O2O3O.dat": (3, 6, 1, "longest pulsation period", "OCVS cep"),
    "cep/cep2O3O.dat": (3, 6, 1, "longer pulsation period", "OCVS cep"),
    "t2cep/t2cep.dat": (3, 6, 1, "pulsation period", "OCVS t2cep"),
    "rrlyr/RRab.dat": (3, 6, 1, "pulsation period (F)", "OCVS rrlyr"),
    "rrlyr/RRc.dat": (3, 6, 1, "pulsation period (1O)", "OCVS rrlyr"),
    "rrlyr/RRd.dat": (3, 6, 1, "first-overtone period", "OCVS rrlyr"),
    "rrlyr/aRRd.dat": (3, 6, 1, "first-overtone period", "OCVS rrlyr"),
    "dsct/dsct.dat": (3, 6, 1, "primary pulsation period", "OCVS dsct (Pietrukowicz+2020)"),
}


def num(tok: str) -> float:
    """Parse a catalog token; '-' or junk -> NaN."""
    try:
        return float(tok)
    except ValueError:
        return np.nan


def read_ocvs(truth_dir: Path) -> pd.DataFrame:
    """Parse every OCVS table in TABLES into one frame keyed by OGLE NAME."""
    rows = []
    for rel, (ip, ia, im, conv, src) in TABLES.items():
        p = truth_dir / rel
        if not p.exists():
            print(f"missing {p}")
            continue
        sub = Path(rel).stem
        for line in open(p):
            t = line.split()
            if len(t) <= ip:
                continue
            rows.append({"NAME": t[0], "P_true": num(t[ip]), "amp": num(t[ia]) if len(t) > ia else np.nan,
                         "imag": num(t[im]), "subtype": sub, "truth_src": src, "p_convention": conv})
    df = pd.DataFrame(rows)
    # subtypes from ident.dat (EB contact/non-contact, ROT has none)
    ecl_id = truth_dir / "ecl/ident.dat"
    if ecl_id.exists():
        st = {l.split()[0]: l.split()[1] for l in open(ecl_id) if l.strip()}
        m = df.truth_src.str.startswith("OCVS ecl")
        df.loc[m, "subtype"] = df.loc[m, "NAME"].map(st).fillna(df.loc[m, "subtype"])
    return df


def read_flares(data_dir: Path) -> pd.DataFrame:
    """Literature rotation periods for the TESS flare stars (Günther+2020 'Per', else Medina+2020 'Prot')."""
    g = pd.read_csv(data_dir / "truth_gunther2020_perstar.csv")
    g = pd.DataFrame({"NAME": g.TESS.astype(str), "P_true": g.Per, "amp": g.amp_med, "imag": g.Tmag,
                      "subtype": "flare", "truth_src": "Gunther+2020 Per", "p_convention": "rotation period",
                      "n_flares": g.n_flares})
    m = pd.read_csv(data_dir / "truth_medina2020.csv")
    m = pd.DataFrame({"NAME": m.TIC.astype(str), "P_true": m.Prot, "amp": m.Amp, "imag": m.Tmag,
                      "subtype": "flare", "truth_src": "Medina+2020 Prot", "p_convention": "rotation period",
                      "ln_flare_rate": m.rate})
    both = pd.concat([g, m[~m.NAME.isin(g.NAME[g.P_true.notna()])]], ignore_index=True)
    return both.sort_values("P_true", na_position="last").drop_duplicates("NAME")


def parse_flare_catalogs(wg6_repo: Path, data_dir: Path) -> None:
    """Write the Günther+2020 per-star and Medina+2020 CSVs from the WG6 repo's CDS tables (``data/*_mrt.txt``)."""
    from astropy.io import ascii

    g = ascii.read(wg6_repo / "data/Gunther_TESS_flare_catalog_ajab5d3at1_mrt.txt", format="cds").to_pandas()
    gs = g.groupby("TESS").agg(n_flares=("Amp", "size"), amp_med=("Amp", "median"), amp_max=("Amp", "max"),
                               Per=("Per", "first"), Tmag=("Tmag", "first"), SpT=("SpT", "first"),
                               sectors=("sector", "nunique"))
    gs.to_csv(data_dir / "truth_gunther2020_perstar.csv")
    m = ascii.read(wg6_repo / "data/Medina_TESS_var_catalog_apjabc686t1_mrt.txt", format="cds").to_pandas()
    m.to_csv(data_dir / "truth_medina2020.csv", index=False)


def period_check(data_dir: Path) -> pd.DataFrame:
    """Join the profiled LS peaks to the truth and classify; writes ``period_check.csv``.

    ``P1c`` is the first of the top-5 LS peaks outside the ground-based window peaks (0.95–1.05 d, 0.49–0.51 d,
    > 300 d); ``out_clean`` compares it with ``P_true`` (match / half / double within 10%, else other; na if missing).
    """
    s = pd.read_csv(data_dir / "profile_ls.csv").drop_duplicates("relpath", keep="last")
    tr = pd.read_csv(data_dir / "truth_joined.csv.gz")[["NAME", "cls", "P_true", "truth_src"]].drop_duplicates("NAME")
    s["NAME"] = s["NAME"].astype(str)
    tr["NAME"] = tr["NAME"].astype(str)
    s = s.merge(tr, on="NAME", how="left")

    def first_clean(r):
        for i in range(1, 6):
            p = r[f"P{i}"]
            if np.isfinite(p) and not (0.95 < p < 1.05 or p > 300 or 0.49 < p < 0.51):
                return p
        return np.nan

    def outcome(r):
        if not (np.isfinite(r.P_true) and np.isfinite(r.P1c)):
            return "na"
        x = r.P1c / r.P_true
        for k, n in ((1, "match"), (0.5, "half"), (2, "double")):
            if abs(x / k - 1) < 0.1:
                return n
        return "other"

    s["P1c"] = s.apply(first_clean, axis=1)
    s["out_clean"] = s.apply(outcome, axis=1)
    s.to_csv(data_dir / "period_check.csv", index=False)
    return s


def main(argv=None):
    """CLI: write truth_joined.csv and truth_match_rates.csv (and, on request, the flare CSVs / period check)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("WG6_DATA_ROOT", "/Volumes/Data_Vault/wg6_roman_data"))
    ap.add_argument("--wg6-repo", default=None, help="clone of rges-pit/variable_catalog_xmatch: (re)parse flare catalogs")
    ap.add_argument("--period-check", action="store_true", help="after profiling: write period_check.csv")
    args = ap.parse_args(argv)
    if args.period_check:
        period_check(REPORT / "data")
        return
    if args.wg6_repo:
        parse_flare_catalogs(Path(args.wg6_repo), REPORT / "data")
    tdir = Path(args.root) / "extracted" / "_truth"
    if not tdir.is_dir():
        sys.exit(f"{tdir} not found: plug in the Data_Vault drive / run tools/fetch_ogle_truth.py.")
    sys.path.insert(0, str(REPORT))
    from figlib import load_inventory  # noqa: E402

    truth = pd.concat([read_ocvs(tdir), read_flares(REPORT / "data")], ignore_index=True)
    inv = load_inventory()[["NAME", "cls", "set"]]
    inv["NAME"] = inv["NAME"].astype(str)
    j = inv.merge(truth.drop_duplicates("NAME"), on="NAME", how="left")
    j.to_csv(REPORT / "data" / "truth_joined.csv.gz", index=False)
    rate = j.groupby("cls").agg(files=("NAME", "size"), matched=("truth_src", lambda s: s.notna().sum()),
                                with_period=("P_true", lambda s: s.notna().sum()))
    rate["period_rate"] = rate.with_period / rate.files
    rate.to_csv(REPORT / "data" / "truth_match_rates.csv")
    print(rate)


if __name__ == "__main__":
    main()
