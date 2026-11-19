#!/usr/bin/env python3
"""Download the TESS light curves for the frozen P27 validation manifest (``datasets/tess_validation_v0.csv``) from MAST.

Rae runs this locally. Products (MAST HLSP, via ``astroquery.mast``):
  * TESS-SPOC FFI light curves (``*_lc.fits``): hold ``SAP_FLUX`` and ``PDCSAP_FLUX``, so one download serves both scores;
  * QLP light curves (``*_llc.fits``): ``SAP_FLUX`` only. Fallback: by default fetched only for stars that have no TESS-SPOC
    light curve at all (``--qlp fallback``); ``--qlp all`` fetches every QLP sector of every star, ``--qlp none`` skips it.

Layout (paths in the index are relative to the root, so the root can move):
    <root>/spoc/TIC<tic>/<mast filename>
    <root>/qlp/TIC<tic>/<mast filename>
    <root>/files_index.csv      one row per file: star_id, tic, product, sector, relpath, size_bytes, data_uri, status
    <root>/download.log         append-only log

<root> = ``--root``, else ``$TESS_VALIDATION_ROOT``, else ``/Volumes/Data_Vault/tess_validation``. If the root is on a
``/Volumes/<drive>`` that is not mounted, the script stops with "plug in the Data_Vault drive".

Resumable and atomic: each file is fetched to ``<name>.part`` and renamed only when complete; a file already present at the
expected size is skipped, so a re-run after an interruption (or after new sectors appear at MAST with ``--refresh``) only
fetches what is missing.

``--dry-run`` queries MAST **metadata only** (observation and product lists, no file downloads), prints the file count and
total size per product and writes the plan to ``--plan-out`` (default ``runs/tess_validation_plan.csv``). It needs no drive.
A real run re-uses that plan if it is at ``<root>/files_plan.csv`` (copied there by the first real run), else it queries first;
``--refresh`` always re-queries.

Usage:
    python tools/download_tess_validation.py --dry-run
    python tools/download_tess_validation.py --dry-run --limit 5          # quick metadata check on 5 stars
    nice -n 15 python tools/download_tess_validation.py                   # the real download
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "tess_validation_v0.csv"
DEFAULT_ROOT = "/Volumes/Data_Vault/tess_validation"
ENV_ROOT = "TESS_VALIDATION_ROOT"
PROVENANCE = {"spoc": "TESS-SPOC", "qlp": "QLP"}
FILE_SUFFIX = {"spoc": "_lc.fits", "qlp": "_llc.fits"}
SECTOR_RE = {"spoc": re.compile(r"-s(\d{4})_"), "qlp": re.compile(r"_s(\d{4})-")}
BATCH = 40
PLAN_COLS = ["star_id", "tic", "product", "sector", "filename", "relpath", "size_bytes", "data_uri"]
MB_PER_S = (1.0, 5.0)   # assumed MAST throughput range for the time estimate (not measured)
SEC_PER_FILE = 0.5      # assumed per-file overhead


# --- root / mount check ---------------------------------------------------------------------------------------------

def resolve_root(arg: str | None) -> Path:
    return Path(arg or os.environ.get(ENV_ROOT) or DEFAULT_ROOT).expanduser()


def check_mount(root: Path) -> None:
    """Exit with 'plug in the Data_Vault drive' if ``root`` is under an unmounted /Volumes/<drive>."""
    parts = root.parts
    if len(parts) >= 3 and parts[1] == "Volumes" and not Path("/", "Volumes", parts[2]).exists():
        sys.exit(f"{parts[2]} is not mounted: plug in the Data_Vault drive (root {root}).")


# --- logging --------------------------------------------------------------------------------------------------------

class Log:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(line + "\n")


# --- planning (MAST metadata only) ----------------------------------------------------------------------------------

def _sector_of(product: str, filename: str, seq) -> int | None:
    m = SECTOR_RE[product].search(filename)
    if m:
        return int(m.group(1))
    try:
        return int(seq)
    except (TypeError, ValueError):
        return None


def query_plan(stars: pd.DataFrame, products: list[str], log: Log) -> pd.DataFrame:
    """MAST metadata queries -> one row per light-curve file (``PLAN_COLS``). No downloads."""
    from astroquery.mast import Observations
    rows = []
    tics = [str(t) for t in stars["tic"]]
    for i in range(0, len(tics), BATCH):
        chunk = tics[i:i + BATCH]
        obs = Observations.query_criteria(obs_collection="HLSP", target_name=chunk,
                                          provenance_name=[PROVENANCE[p] for p in products])
        log(f"metadata: stars {i + 1}-{i + len(chunk)} of {len(tics)}: {len(obs)} observations")
        if len(obs) == 0:
            continue
        prov = {str(o): str(p) for o, p in zip(obs["obsid"], obs["provenance_name"])}
        tgt = {str(o): str(t) for o, t in zip(obs["obsid"], obs["target_name"])}
        seq = {str(o): s for o, s in zip(obs["obsid"], obs["sequence_number"])}
        for j in range(0, len(obs), BATCH):
            prods = Observations.get_product_list(obs[j:j + BATCH])
            for r in prods:
                fn = str(r["productFilename"])
                pid = str(r["parent_obsid"])
                p = next((k for k, v in PROVENANCE.items() if prov.get(pid) == v), None)
                if p is None or str(r["productType"]) != "SCIENCE" or not fn.endswith(FILE_SUFFIX[p]):
                    continue
                tic = int(tgt[pid])
                rows.append(dict(star_id=f"TIC{tic}", tic=tic, product=p, sector=_sector_of(p, fn, seq.get(pid)),
                                 filename=fn, relpath=f"{p}/TIC{tic}/{fn}", size_bytes=int(r["size"] or 0),
                                 data_uri=str(r["dataURI"])))
    plan = pd.DataFrame(rows, columns=PLAN_COLS).drop_duplicates("relpath")
    return plan.sort_values(["tic", "product", "sector"]).reset_index(drop=True)


def apply_qlp_policy(plan: pd.DataFrame, qlp: str) -> pd.DataFrame:
    """``fallback``: keep QLP files only for stars with no TESS-SPOC file in the plan; ``all``; ``none``."""
    if qlp == "none":
        return plan[plan["product"] != "qlp"]
    if qlp == "all":
        return plan
    has_spoc = set(plan.loc[plan["product"] == "spoc", "tic"])
    return plan[(plan["product"] == "spoc") | ~plan["tic"].isin(has_spoc)]


def summarize(plan: pd.DataFrame, stars: pd.DataFrame) -> str:
    lines = [f"{len(plan)} files for {plan['tic'].nunique()} of {len(stars)} stars"]
    for p, g in plan.groupby("product"):
        gb = g["size_bytes"].sum() / 1e9
        lines.append(f"  {p:5s}: {len(g):5d} files, {g['tic'].nunique():3d} stars, {gb:6.2f} GB")
    tot = plan["size_bytes"].sum() / 1e6
    lo, hi = tot / MB_PER_S[1] + len(plan) * SEC_PER_FILE, tot / MB_PER_S[0] + len(plan) * SEC_PER_FILE
    lines.append(f"  total: {tot / 1e3:.2f} GB; at an assumed {MB_PER_S[0]:g}-{MB_PER_S[1]:g} MB/s that is "
                 f"{lo / 60:.0f}-{hi / 60:.0f} min (a guess, not measured)")
    none = stars[~stars["tic"].isin(plan["tic"])]
    lines.append(f"  stars with no SPOC/QLP file in the plan: {len(none)}")
    return "\n".join(lines)


# --- download (resumable, atomic) -----------------------------------------------------------------------------------

def astroquery_fetch(uri: str, dest: Path) -> None:
    """Fetch one MAST data URI to ``dest`` (called with a ``.part`` path)."""
    from astroquery.mast import Observations
    status, msg, _ = Observations.download_file(uri, local_path=str(dest), cache=False, verbose=False)
    if status != "COMPLETE":
        raise RuntimeError(f"{status}: {msg}")


def download(plan: pd.DataFrame, root: Path, log: Log, fetch=astroquery_fetch, retries: int = 3, sleep=time.sleep) -> pd.DataFrame:
    """Download every planned file under ``root``; returns the plan with a ``status`` column (done/skipped/failed)."""
    status = []
    n = len(plan)
    for k, r in enumerate(plan.itertuples(), 1):
        dest = root / r.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (r.size_bytes in (0, dest.stat().st_size)):
            status.append("skipped")
            continue
        part = dest.with_name(dest.name + ".part")
        ok = False
        for attempt in range(1, retries + 1):
            try:
                if part.exists():
                    part.unlink()
                fetch(r.data_uri, part)
                if r.size_bytes and part.stat().st_size != r.size_bytes:
                    raise RuntimeError(f"size {part.stat().st_size} != expected {r.size_bytes}")
                os.replace(part, dest)
                ok = True
                break
            except Exception as e:  # noqa: BLE001 - network errors: log, back off, retry
                log(f"FAIL {r.relpath} attempt {attempt}/{retries}: {e}")
                sleep(5 * attempt)
        status.append("done" if ok else "failed")
        if ok and (k % 25 == 0 or k == n):
            log(f"{k}/{n} files ({status.count('done')} new, {status.count('skipped')} present, {status.count('failed')} failed)")
    out = plan.copy()
    out["status"] = status
    return out


def write_index(df: pd.DataFrame, root: Path) -> None:
    tmp = root / "files_index.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, root / "files_index.csv")


# --- main -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--root", default=None, help=f"destination root (default ${ENV_ROOT} or {DEFAULT_ROOT})")
    ap.add_argument("--dry-run", action="store_true", help="MAST metadata only: list files and total size, download nothing")
    ap.add_argument("--qlp", choices=["fallback", "all", "none"], default="fallback")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest stars (testing)")
    ap.add_argument("--refresh", action="store_true", help="re-query MAST even if a plan exists")
    ap.add_argument("--plan-out", default=str(ROOT / "runs" / "tess_validation_plan.csv"), help="where --dry-run writes the plan")
    args = ap.parse_args(argv)

    stars = pd.read_csv(args.manifest)
    if args.limit:
        stars = stars.head(args.limit)
    root = resolve_root(args.root)
    products = ["spoc"] if args.qlp == "none" else ["spoc", "qlp"]

    if args.dry_run:
        log = Log(None)
        plan = apply_qlp_policy(query_plan(stars, products, log), args.qlp)
        Path(args.plan_out).parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(args.plan_out, index=False)
        print(summarize(plan, stars))
        print(f"plan written to {args.plan_out}")
        return 0

    check_mount(root)
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "download.log")
    plan_path = root / "files_plan.csv"
    if plan_path.exists() and not args.refresh:
        plan = pd.read_csv(plan_path)
        plan = plan[plan["tic"].isin(stars["tic"])]
        log(f"re-using plan {plan_path} ({len(plan)} files)")
    else:
        plan = apply_qlp_policy(query_plan(stars, products, log), args.qlp)
        tmp = plan_path.with_suffix(".tmp")
        plan.to_csv(tmp, index=False)
        os.replace(tmp, plan_path)
    log(summarize(plan, stars).replace("\n", " | "))
    res = download(plan, root, log)
    write_index(res, root)
    nfail = int((res["status"] == "failed").sum())
    log(f"finished: {int((res['status'] == 'done').sum())} downloaded, {int((res['status'] == 'skipped').sum())} already present, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
