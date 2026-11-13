"""Build ``datasets/wg6_roman_v0.csv``: the first frozen-candidate manifest of the WG6 Roman-resampled light curves (P35).

Rae's decisions (2026-11-13): **improved** noise only; development on the **three early seasons** (seasons 1-3 of WG6's six
consecutive 71-d seasons, an assumption recorded in ``claude_files/wg6.py``); EBs/ELLs are a flagged class (``is_eb``);
rotators beyond the 54 FL templates come later (P34), so the schema just has to be able to hold them (``is_rotator``,
``cls='ROT'``, ``variant``).

One row per (file, season). Columns:

    star_id           unique row id: ``<obj_id>[_WF<mag>]_s<season>``
    obj_id            the star/object (OGLE id, or ``FL_<idx>``); seasons and magnitude copies share it
    zip, member       FITS location, relative to ``$WG6_RESAMPLED_ROOT`` (default /Volumes/Data_Vault/wg6_roman_data_resampled)
    variant           ``improved`` (only value in v0)
    cls               ECL, ELL, T2CEP, RRLYR, HB, DSCT, LPV, CV, FL  (ROT/CEP reserved: absent from the improved set)
    subtype           OCVS subtype (NC/C for ECL, RRab/RRc/..., ``flare``)
    P_true            catalogue period (OGLE OCVS, or the TESS template's literature rotation period for FL); NaN = none
    p_convention      what P_true means (orbital / pulsation / rotation / ...), from the truth join
    P_phot_expected   the period a photometric method should lock to: P_true/2 for ECL and ELL (two minima per orbit),
                      else P_true. Score EBs against both.
    fl_template       TIC id of the FL template (``flat`` = no rotation); blank for OGLE classes
    wfmag             FL magnitude copy; blank for OGLE classes
    season            1-3
    is_rotator        True for FL objects with a literature rotation period
    is_eb             True for ECL and ELL (flagged class, not plain negatives)
    in_raw            star is also in the raw archival WG6 set
    eval_set          dev | holdout. Split by FL template (and by OGLE star): nothing is on both sides.
                      The holdout is never evaluated without Rae's go-ahead.

Selection (seeded, deterministic): ~300 stars per contaminant class (ECL, ELL, T2CEP, RRLYR; CEP has no improved
files); all of HB/DSCT/LPV/CV up to ``--n-stress`` (100); every FL rotator object at ``--fl-mags`` (17 and 20); the 79 FL
``flat`` objects as non-rotator negatives. ~20% of templates / stars go to the holdout.

Inputs: the scoping report's committed inventory + truth tables (``--inventory``, ``--truth``). If the inventory file is
missing it is rebuilt from the zip directories on the drive (cheap: names only, no decompression) and cached atomically.
``--verify N`` opens N random members from the zips to confirm they exist and hold the expected 6 seasons (drive step).
The build is a pure table job and resumable: an up-to-date output is kept unless ``--force``; every write is
temp-file-then-rename. Run: ``python tools/build_wg6_manifest.py [--verify 20]``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "claude_files"))
sys.path.insert(0, str(REPO / "tools/audit"))

SCOPING = REPO / "reports/2026-11-12_wg6-dataset-scoping/data/resampled"
DEFAULT_ROOT = "/Volumes/Data_Vault/wg6_roman_data_resampled"
CONTAM_CLASSES = ("ECL", "ELL", "T2CEP", "RRLYR")      # ~n-per-class stars each (CEP: no improved files)
STRESS_CLASSES = ("HB", "DSCT", "LPV", "CV")           # smaller stress sets
EB_CLASSES = ("ECL", "ELL")
COLUMNS = ["star_id", "obj_id", "zip", "member", "variant", "cls", "subtype", "P_true", "p_convention",
           "P_phot_expected", "fl_template", "wfmag", "season", "is_rotator", "is_eb", "in_raw", "eval_set"]


def _h(key: str, salt: str) -> int:
    """Stable integer hash of ``salt|key`` (md5), used for seeded selection and splits independent of row order."""
    return int(hashlib.md5(f"{salt}|{key}".encode()).hexdigest()[:12], 16)


def atomic_write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    """Write ``df`` to ``path`` via a temp file in the same directory, then rename (never leaves a partial file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def holdout_keys(keys, frac: float, seed: str) -> set:
    """Exactly ``round(frac * n)`` of the unique ``keys`` (at least 1 if n >= 2), chosen by hash order: the holdout side."""
    uniq = sorted(set(keys), key=lambda k: _h(str(k), seed + "|split"))
    n = int(round(frac * len(uniq))) if len(uniq) >= 2 else 0
    n = max(n, 1) if len(uniq) >= 2 else 0
    return set(uniq[:n])


def pick(keys, n: int, seed: str, tag: str) -> list:
    """The ``n`` keys with the smallest salted hash (a seeded sample without replacement, stable under reordering)."""
    return sorted(sorted(set(keys), key=lambda k: _h(str(k), f"{seed}|{tag}"))[:n])


def load_inventory(root: Path, path: Path) -> pd.DataFrame:
    """Inventory (one row per FITS member). Read ``path`` if present, else scan the drive zips and cache it atomically."""
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    if not root.is_dir():
        sys.exit(f"{root} not found and no inventory at {path}: plug in the Data_Vault drive.")
    from wg6r_profile import build_inventory
    inv = build_inventory(root)
    atomic_write_csv(inv, path, compression="gzip")
    return inv


def build(inv: pd.DataFrame, truth: pd.DataFrame, n_per_class: int = 300, n_stress: int = 100,
          fl_mags=(17, 20), seasons=(1, 2, 3), holdout_frac: float = 0.2, seed: str = "wg6r_v0") -> pd.DataFrame:
    """Select stars and return the manifest DataFrame (see the module docstring for rules and columns)."""
    inv = inv[inv.variant == "improved"].copy()
    tr = truth.set_index("star_id")
    inv["obj_id"] = inv["star_id"]
    rows = []

    def add(r, rotator: bool, eval_set: str, cls: str):
        t = tr.loc[r.obj_id] if r.obj_id in tr.index else None
        P = float(t.P_true) if t is not None and pd.notna(t.P_true) else np.nan
        conv = "" if t is None or pd.isna(t.p_convention) else str(t.p_convention)
        sub = "" if t is None or pd.isna(t.subtype) else str(t.subtype)
        in_raw = bool(t.in_raw) if t is not None and pd.notna(t.in_raw) else False
        pexp = P / 2 if cls in EB_CLASSES and np.isfinite(P) else P
        wf = "" if pd.isna(r.wfmag) else int(r.wfmag)
        tmpl = "" if pd.isna(r.fl_template) else str(r.fl_template)
        base = r.obj_id + (f"_WF{wf}" if wf != "" else "") + (f"_{r.cv_ind}" if isinstance(r.cv_ind, str) else "")
        for k in seasons:
            rows.append(dict(star_id=f"{base}_s{k}", obj_id=r.obj_id, zip=r.zip, member=r.member, variant="improved",
                             cls=cls, subtype=sub, P_true=P, p_convention=conv, P_phot_expected=pexp, fl_template=tmpl,
                             wfmag=wf, season=k, is_rotator=rotator, is_eb=cls in EB_CLASSES, in_raw=in_raw,
                             eval_set=eval_set))

    # --- FL: split by template (rotators) / by object (flat) -------------------------------------------------------
    fl = inv[(inv.cls == "FL") & inv.wfmag.isin(fl_mags)].copy()
    fl["P_true"] = fl.obj_id.map(tr.P_true)
    rot = fl[fl.P_true.notna()]
    hold_t = holdout_keys(rot.fl_template.astype(str), holdout_frac, seed + "|FLtemplate")
    for _, r in rot.iterrows():
        add(r, True, "holdout" if str(r.fl_template) in hold_t else "dev", "FL")
    flat = fl[fl.fl_template.astype(str) == "flat"]
    hold_f = holdout_keys(flat.obj_id, holdout_frac, seed + "|FLflat")
    for _, r in flat.iterrows():
        add(r, False, "holdout" if r.obj_id in hold_f else "dev", "FL")

    # --- OGLE classes: pick stars, split by star -------------------------------------------------------------------
    for cls in CONTAM_CLASSES + STRESS_CLASSES:
        g = inv[inv.cls == cls]
        if cls == "CV":                      # 10 copies per star: take the first copy of each star
            g = g.sort_values(["obj_id", "cv_ind"]).drop_duplicates("obj_id")
        n = n_per_class if cls in CONTAM_CLASSES else n_stress
        chosen = pick(g.obj_id, n, seed, cls)
        hold = holdout_keys(chosen, holdout_frac, seed + "|" + cls)
        for _, r in g[g.obj_id.isin(chosen)].iterrows():
            add(r, False, "holdout" if r.obj_id in hold else "dev", cls)

    man = pd.DataFrame(rows, columns=COLUMNS).sort_values(["cls", "obj_id", "wfmag", "season"], key=lambda s: s.astype(str))
    return man.reset_index(drop=True)


def check_split(man: pd.DataFrame) -> None:
    """Raise AssertionError if any OGLE star, FL object or FL template (non-flat) is on both sides of dev/holdout."""
    for col in ("obj_id", "fl_template"):
        m = man[man[col].astype(str).ne("") & man[col].astype(str).ne("flat") & man[col].notna()]
        n_sets = m.groupby(col).eval_set.nunique()
        bad = n_sets[n_sets > 1]
        assert bad.empty, f"{col} on both sides of the split: {list(bad.index[:5])}"


def verify_members(man: pd.DataFrame, root: Path, n: int, seed: str) -> None:
    """Open ``n`` random manifest members from the zips and check each has an F146 table with 6 seasons (drive step)."""
    if not root.is_dir():
        sys.exit(f"{root} not found: plug in the Data_Vault drive.")
    from wg6 import load_wg6_roman
    sub = man.drop_duplicates(["zip", "member"])
    pick_idx = sorted(range(len(sub)), key=lambda i: _h(sub.member.iloc[i], seed + "|verify"))[:n]
    for i in pick_idx:
        r = sub.iloc[i]
        lc = load_wg6_roman(root / r.zip, r.member, "F146", season=1)
        assert lc.meta["n_seasons_in_file"] == 6, (r.member, lc.meta)
        assert 8000 < len(lc.time) < 9000, (r.member, len(lc.time))
    print(f"verified {len(pick_idx)} members: F146 present, 6 seasons, ~8450 points/season")


def main(argv=None):
    """CLI entry: build, check the split, write ``datasets/wg6_roman_v0.csv`` atomically."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("WG6_RESAMPLED_ROOT", DEFAULT_ROOT))
    ap.add_argument("--inventory", type=Path, default=SCOPING / "inventory.csv.gz")
    ap.add_argument("--truth", type=Path, default=SCOPING / "truth.csv.gz")
    ap.add_argument("--out", type=Path, default=REPO / "datasets/wg6_roman_v0.csv")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--n-stress", type=int, default=100)
    ap.add_argument("--fl-mags", type=int, nargs="+", default=[17, 20])
    ap.add_argument("--seasons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", default="wg6r_v0")
    ap.add_argument("--verify", type=int, default=0, help="open N random members on the drive to check them")
    ap.add_argument("--force", action="store_true", help="rebuild even if --out exists")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.out.exists() and not a.force:
        man = pd.read_csv(a.out)
        print(f"{a.out} exists ({len(man)} rows); keeping it (use --force to rebuild).")
    else:
        inv = load_inventory(root, a.inventory)
        truth = pd.read_csv(a.truth, low_memory=False)
        man = build(inv, truth, a.n_per_class, a.n_stress, tuple(a.fl_mags), tuple(a.seasons), a.holdout_frac, a.seed)
        check_split(man)
        atomic_write_csv(man, a.out)
        print(f"wrote {a.out}: {len(man)} rows, {man.obj_id.nunique()} objects")
    print(man.groupby(["cls", "eval_set"]).obj_id.nunique().unstack(fill_value=0))
    if a.verify:
        verify_members(man, root, a.verify, a.seed)


if __name__ == "__main__":
    main()
