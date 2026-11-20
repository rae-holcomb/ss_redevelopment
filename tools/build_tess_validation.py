#!/usr/bin/env python3
"""Freeze the P27 real-TESS validation manifests: ``datasets/tess_validation_v0.csv`` and ``tess_validation_v1.csv``.

**v1 (2026-11-19, current)** applies Rae's answers to v0's open questions: the 72 unscorable stars (no TESS-SPOC or QLP
light curve at MAST) and the 7 crowded EBs (TIC ``contratio`` >= 0.2) are replaced by stars from the same pools, strata
counts unchanged (``datasets/proposed/p27_v1_replacements.csv``, ``tools/validation_set/draw_replacements.py``); the
Teff >= 6500 K ``hot_star`` flag is replaced by the Gaia DR3 classification (``datasets/proposed/p27_v1_gaia_classes.csv``,
``tools/validation_set/gaia_classes.py``); and a stratified, seeded 30% of the stars is locked as ``split == "holdout"`` (the
other 70% is ``validation``). **Never score the holdout without Rae's word.** v0 (superseded, never scored) still reproduces
with ``--version v0``.

Inputs (all committed, no network):
  * ``datasets/proposed/p27_validation_draft.csv``  (built by ``tools/validation_set/build_manifest.py``, seed 27)
  * ``reports/2026-11-12_validation-set-scoping/data/zach_gaia_variability.csv.gz``  (Gaia DR3 variability flags for
    Zach Claytor's 23,332 IDs; 294 are known variables)
  * ``datasets/external/claytor2025_nonrotator_ticids.txt``

The draft is NOT re-drawn here. This script re-checks it against Rae's P27 decisions (see ``check_decisions``), applies the
mechanical fixes listed in ``FIXES`` and adds the derived columns below. A change to the frozen file means a new version
(``tess_validation_v1.csv``), never an edit.

Columns added to the draft (the draft's own columns are kept, in order, after ``star_id``):
  star_id            ``TIC<tic>``; the key the downloader, loader and scorecard use.
  truth_kind         ``spot_rotation`` (literature P_rot), ``orbital`` (Prsa+2022 P_orb), ``none``.
  pbin               truth-period bin for rotators and EBs (0.1-1, 1-3, 3-10, 10-20, 20-40, 40-71 d); empty otherwise.
  star_type          ``rc_giant`` (kepler_rc_quiet block), ``white_dwarf``, ``hot_star`` (Teff >= 6500 K: delta Sct / gamma Dor
                     strips), ``giant`` (logg < 3.5), else ``dwarf``.  Priority in that order.
  pulsation_eligible True for rc_giant, hot_star, giant: a detection there can be scored as the ``pulsation`` outcome
                     (the scoring window per star_type is in the P27 freeze report, not in this file).
  gaia_variable      0/1 for ``expected_undetectable`` stars: in the Gaia DR3 variability tables (always 0: those 294 were
                     excluded before the draw; kept as an audit column). Empty for other blocks.
  product_primary    ``spoc`` if the star has >= 1 TESS-SPOC FFI light curve at MAST, else ``qlp`` if >= 1 QLP, else ``none``.
  scorable           product_primary != none (the star can be scored from SPOC or QLP at all).

v1 columns (changes from v0): ``pulsation_eligible`` and ``hot_star`` are gone; added ``gaia_class`` (Gaia DR3 classifier best
class, empty if not a Gaia variable), ``gaia_pulsator`` (class DSCT|GDOR|SXPHE or in the main-sequence-oscillator table),
``pulsation_class`` (``rc_giant`` = the ``kepler_rc_quiet`` block, ``gaia_pulsator``, else empty: selects the pulsation window
in ``evaluation.classify_negative``), ``split`` (``validation``/``holdout``). ``star_type`` is now rc_giant / white_dwarf /
giant (logg < 3.5) / dwarf. ``gaia_variable`` (the Zach-block audit) is kept.

Usage:
    python tools/build_tess_validation.py [--version v1]            # write datasets/tess_validation_v1.csv
    python tools/build_tess_validation.py [--version v1] --check    # rebuild in memory, fail if it differs from the file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "datasets" / "proposed" / "p27_validation_draft.csv"
GAIA = ROOT / "reports" / "2026-11-12_validation-set-scoping" / "data" / "zach_gaia_variability.csv.gz"
ZACH_IDS = ROOT / "datasets" / "external" / "claytor2025_nonrotator_ticids.txt"
OUT = ROOT / "datasets" / "tess_validation_v0.csv"
OUT_V1 = ROOT / "datasets" / "tess_validation_v1.csv"
REPLACEMENTS = ROOT / "datasets" / "proposed" / "p27_v1_replacements.csv"
GAIA_CLASSES = ROOT / "datasets" / "proposed" / "p27_v1_gaia_classes.csv"
HOLDOUT_FRAC = 0.30
SPLIT_SEED = 20261119
PULSATOR_CLASSES = ("DSCT", "GDOR", "SXPHE")

PBINS = [0.1, 1, 3, 10, 20, 40, 71]
HOT_TEFF = 6500.0
GIANT_LOGG = 3.5
EXPECTED_COUNTS = {"rotator": 268, "nonrotator": 100, "nonrotator_candidate": 22,
                   "expected_undetectable": 40, "contaminant_eb": 50}
EXPECTED_SOURCES = {"kepler_rc_quiet": 50, "kepler_dwarf_nodetect": 50, "claytor_scvz_template": 40,
                    "white_dwarf": 22, "prsa2022_eb": 50}
# Mechanical fixes applied to the draft (each is listed in the report):
FIXES = [
    "n_tess_spoc / n_qlp / n_tglc / n_cdips cast from float to int (the draft stored 3.0, not 3)",
    "tic cast to plain int; rows sorted by (label order, source, tic) so the file is stable under re-runs",
]
LABEL_ORDER = ["rotator", "nonrotator", "nonrotator_candidate", "expected_undetectable", "contaminant_eb"]
COUNT_COLS = ["n_tess_spoc", "n_qlp", "n_tglc", "n_cdips"]


def check_decisions(d: pd.DataFrame, gaia: pd.DataFrame, zach_ids: set) -> None:
    """Raise AssertionError if the draft contradicts a P27 decision (Rae, 2026-11-12/13/19)."""
    assert len(d) == 480 and d["tic"].is_unique, "480 unique TIC IDs expected"
    assert d["label"].value_counts().to_dict() == EXPECTED_COUNTS, d["label"].value_counts().to_dict()
    assert d["source"].isin(list(EXPECTED_SOURCES) + sorted(set(d.loc[d.label == "rotator", "source"]))).all()
    for src, n in EXPECTED_SOURCES.items():
        assert (d["source"] == src).sum() == n, (src, (d["source"] == src).sum())
    # Zach's stars: labelled expected_undetectable, in his list, none a Gaia DR3 variable.
    z = d[d["source"] == "claytor_scvz_template"]
    assert (z["label"] == "expected_undetectable").all() and z["tic"].isin(zach_ids).all()
    flagged = gaia.loc[gaia[["in_vclassre", "in_vmsosc", "in_vrm", "in_veb"]].sum(axis=1) > 0, "ID"]
    assert len(flagged) == 294, len(flagged)
    assert not z["tic"].isin(set(flagged)).any(), "an expected_undetectable star is a Gaia DR3 variable"
    assert z["tic"].isin(gaia["ID"]).all(), "Gaia check does not cover every expected_undetectable star"
    # No other rotator/negative block shares a star with Zach's list. (EBs may: TIC 358111274 is a Prsa EB with a P_orb
    # truth, which is a fine EB test whatever the list says about it.)
    other = d[~d["source"].isin(["claytor_scvz_template", "prsa2022_eb"])]
    assert not other["tic"].isin(zach_ids).any()
    # Provenance: every literature period has a method, a data set and a reference.
    r = d[d["label"] == "rotator"]
    assert r["truth_period"].between(0.1, 71).all(), "rotator period outside 0.1-71 d (70-180 d block is skipped, 2026-11-19)"
    for c in ("period_method", "period_dataset", "period_ref"):
        assert d[c].notna().all() and (d[c].astype(str).str.len() > 0).all(), c
    assert r["period_method"].str.startswith("spot modulation").all()
    # Truth: periods only for rotators (literature) and EBs (orbital); none for every negative block.
    e = d[d["label"] == "contaminant_eb"]
    assert e["truth_period"].between(0.1, 71).all()
    neg = d[~d["label"].isin(["rotator", "contaminant_eb"])]
    assert neg["truth_period"].isna().all()
    # Selection cuts documented in the scoping report: Tmag <= 14; contratio < 0.2 for rotators and negatives (EBs exempt).
    assert (d["Tmag"] <= 14).all()
    assert (d.loc[d["label"] != "contaminant_eb", "contratio"].fillna(0) < 0.2).all()
    # Coverage columns are consistent.
    assert (d["sectors"].str.split().str.len() == d["n_sectors"]).all()


def build_v0() -> pd.DataFrame:
    d = pd.read_csv(DRAFT)
    gaia = pd.read_csv(GAIA)
    zach_ids = set(pd.read_csv(ZACH_IDS, header=None, comment="#")[0].astype(int))
    check_decisions(d, gaia, zach_ids)

    d["tic"] = d["tic"].astype("int64")
    for c in COUNT_COLS:
        d[c] = d[c].astype(int)
    d["_lab"] = d["label"].map({k: i for i, k in enumerate(LABEL_ORDER)})
    d = d.sort_values(["_lab", "source", "tic"]).drop(columns="_lab").reset_index(drop=True)

    d.insert(0, "star_id", "TIC" + d["tic"].astype(str))
    kind = np.where(d["label"] == "rotator", "spot_rotation", np.where(d["label"] == "contaminant_eb", "orbital", "none"))
    d.insert(d.columns.get_loc("truth_period") + 1, "truth_kind", kind)

    has_p = d["truth_period"].notna()
    names = ["0.1-1", "1-3", "3-10", "10-20", "20-40", "40-71"]
    pb = pd.cut(d["truth_period"].where(has_p), PBINS, labels=names).astype(object)
    d["pbin"] = np.where(has_p, pb, "")

    st = np.where(d["source"] == "kepler_rc_quiet", "rc_giant",
         np.where(d["source"] == "white_dwarf", "white_dwarf",
         np.where(d["Teff"] >= HOT_TEFF, "hot_star",
         np.where(d["logg"] < GIANT_LOGG, "giant", "dwarf"))))
    d["star_type"] = st
    d["pulsation_eligible"] = pd.Series(st).isin(["rc_giant", "hot_star", "giant"]).to_numpy()

    gv = gaia.set_index("ID")[["in_vclassre", "in_vmsosc", "in_vrm", "in_veb"]].sum(axis=1).gt(0).astype(int)
    is_z = d["label"] == "expected_undetectable"
    d["gaia_variable"] = pd.array([gv.get(t) if z else None for t, z in zip(d["tic"], is_z)], dtype="Int64")

    prim = np.where(d["n_tess_spoc"] > 0, "spoc", np.where(d["n_qlp"] > 0, "qlp", "none"))
    d["product_primary"] = prim
    d["scorable"] = prim != "none"
    return d


def _split(d: pd.DataFrame) -> pd.Series:
    """Stratified seeded holdout: per (label, pbin or source) stratum, round(30%) of the stars are ``holdout``."""
    rng = np.random.default_rng(SPLIT_SEED)
    strat = d["label"] + "|" + np.where(d["pbin"] != "", d["pbin"], d["source"])
    out = pd.Series("validation", index=d.index)
    for _, g in d.groupby(strat, sort=True):
        k = int(round(HOLDOUT_FRAC * len(g)))
        pick = rng.permutation(g.index.to_numpy())[:k]
        out.loc[pick] = "holdout"
    return out


def build_v1() -> pd.DataFrame:
    """v1: the draft minus unscorable/crowded stars, plus their replacements, Gaia class, split (see module docstring)."""
    draft = pd.read_csv(DRAFT)
    rep = pd.read_csv(REPLACEMENTS)
    gaia = pd.read_csv(GAIA)
    zach_ids = set(pd.read_csv(ZACH_IDS, header=None, comment="#")[0].astype(int))
    check_decisions(draft, gaia, zach_ids)  # the draft still satisfies every P27 decision
    unscor = (draft["n_tess_spoc"] == 0) & (draft["n_qlp"] == 0)
    crowded = (draft["label"] == "contaminant_eb") & (draft["contratio"] >= 0.2)
    removed = draft[unscor | crowded]
    assert set(rep["replaces"]) == set(removed["tic"]) and len(rep) == len(removed), "replacements do not match the removed stars"
    assert not set(rep["tic"]) & set(draft["tic"]), "a replacement is already in the draft"
    new = pd.concat([draft[~(unscor | crowded)], rep.drop(columns="replaces")], ignore_index=True)
    # strata counts are unchanged by construction; re-assert them and the decisions on the new rows
    new_pb = pd.cut(new["truth_period"], PBINS).astype(str)
    old_pb = pd.cut(draft["truth_period"], PBINS).astype(str)
    # rotators match on (label, period bin) only (a replacement may come from another cluster); other blocks also on source
    key = lambda df, pb: df.assign(_pb=pb, _s=np.where(df["label"] == "rotator", "*", df["source"])).groupby(
        ["label", "_s", "_pb"]).size().to_dict()  # noqa: E731
    assert key(new, new_pb) == key(draft, old_pb), "strata counts changed"
    assert ((new["n_tess_spoc"] + new["n_qlp"]) > 0).all(), "a star without SPOC or QLP is still in the set"
    assert (new["contratio"].fillna(0) < 0.2).all(), "contratio >= 0.2 star remains (EBs now included)"
    z = new[new["source"] == "claytor_scvz_template"]
    flagged = gaia.loc[gaia[["in_vclassre", "in_vmsosc", "in_vrm", "in_veb"]].sum(axis=1) > 0, "ID"]
    assert z["tic"].isin(zach_ids).all() and not z["tic"].isin(set(flagged)).any()
    assert new["tic"].is_unique and len(new) == 480 and (new["Tmag"] <= 14).all()
    assert new["truth_period"].dropna().between(0.1, 71).all()

    new["tic"] = new["tic"].astype("int64")
    for c in COUNT_COLS:
        new[c] = new[c].astype(int)
    new["_lab"] = new["label"].map({k: i for i, k in enumerate(LABEL_ORDER)})
    new = new.sort_values(["_lab", "source", "tic"]).drop(columns="_lab").reset_index(drop=True)
    d = new
    d.insert(0, "star_id", "TIC" + d["tic"].astype(str))
    has_p = d["truth_period"].notna()
    d.insert(d.columns.get_loc("truth_period") + 1, "truth_kind",
             np.where(d["label"] == "rotator", "spot_rotation", np.where(d["label"] == "contaminant_eb", "orbital", "none")))
    names = ["0.1-1", "1-3", "3-10", "10-20", "20-40", "40-71"]
    d["pbin"] = np.where(has_p, pd.cut(d["truth_period"].where(has_p), PBINS, labels=names).astype(object), "")
    d["star_type"] = np.where(d["source"] == "kepler_rc_quiet", "rc_giant", np.where(d["source"] == "white_dwarf", "white_dwarf",
                     np.where(d["logg"] < GIANT_LOGG, "giant", "dwarf")))
    g = pd.read_csv(GAIA_CLASSES)[["tic", "gaia_class", "gaia_class_score", "in_vmsosc"]]
    d = d.merge(g, on="tic", how="left")
    cls = d["gaia_class"].fillna("")
    d["gaia_pulsator"] = (cls.apply(lambda c: any(p in c for p in PULSATOR_CLASSES)) | (d["in_vmsosc"].fillna(0) > 0))
    d["gaia_class"] = cls
    d["pulsation_class"] = np.where(d["source"] == "kepler_rc_quiet", "rc_giant", np.where(d["gaia_pulsator"], "gaia_pulsator", ""))
    d = d.drop(columns=["gaia_class_score", "in_vmsosc"])
    gv = pd.read_csv(GAIA).set_index("ID")[["in_vclassre", "in_vmsosc", "in_vrm", "in_veb"]].sum(axis=1).gt(0).astype(int)
    is_z = d["label"] == "expected_undetectable"
    d["gaia_variable"] = pd.array([gv.get(t) if zz else None for t, zz in zip(d["tic"], is_z)], dtype="Int64")
    d["product_primary"] = np.where(d["n_tess_spoc"] > 0, "spoc", "qlp")
    d["scorable"] = True
    d["split"] = _split(d)
    return d


def build(version: str = "v1") -> pd.DataFrame:
    return build_v1() if version == "v1" else build_v0()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", choices=["v0", "v1"], default="v1")
    ap.add_argument("--check", action="store_true", help="rebuild in memory and fail if it differs from the committed file")
    args = ap.parse_args(argv)
    out = OUT_V1 if args.version == "v1" else OUT
    d = build(args.version)
    text = d.to_csv(index=False, float_format="%.10g")
    if args.check:
        if not out.exists() or out.read_text() != text:
            print(f"FAIL: {out} is missing or differs from the rebuild", file=sys.stderr)
            return 1
        print(f"OK: {out.name} reproduces from the committed inputs ({len(d)} rows)")
        return 0
    out.write_text(text)
    print(f"wrote {out} ({len(d)} rows)")
    print(d["label"].value_counts().to_string())
    print(f"scorable: {int(d['scorable'].sum())}/{len(d)}")
    if args.version == "v1":
        print(pd.crosstab(d["label"], d["split"]).to_string())
        print("pulsation_class:", d["pulsation_class"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
