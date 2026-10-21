"""T20 follow-up (f): fetch independent TESS FFI light curves (QLP / TASOC / ELEANOR-LITE HLSPs) for selected noise sources.

The SMARTS noise light curves came from Zach Claytor's custom FFI reduction. If a
periodic signal we found in a SMARTS noise source also shows up, at the same
period, in an independent reduction of the same TIC, it is intrinsic to the sky
at that position (the source itself or a contaminating neighbour), not an
artefact of Zach's pipeline. No SPOC products exist for these TIC EXTENDED
objects (checked 2026-10-21); the MAST HLSPs below do.

Run with an env that has lightkurve (``codeastro``). For each TIC in
``--select`` it searches MAST, takes up to ``--n-lc`` light curves (preferring
QLP, then TASOC, then GSFC-ELEANOR-LITE; distinct sectors), downloads them to
``<outdir>/cache`` (lightkurve's own download cache, so reruns are free) and
writes each as a plain CSV ``<outdir>/lc/<tic>_<author>_s<sector>.csv``
(time, flux, normalized, quality-masked, NaNs dropped) plus
``<outdir>/ffi_index.csv``. Analysis happens in ``ss_openblas``.

Usage::

    /opt/anaconda3/envs/codeastro/bin/python -I tools/audit/ffi_lc_check.py \\
        --select reports/<folder>/data/ffi_selection.csv --outdir /Volumes/Data_Vault/smarts_data/extracted/ffi_check
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

AUTHOR_PREF = ["QLP", "TASOC", "GSFC-ELEANOR-LITE"]


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically (temp file in the same directory, then rename)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part.csv")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_tic(tic: int, n_lc: int, outdir: Path, pause: float = 1.0) -> list:
    """Search, download and save up to ``n_lc`` HLSP light curves of one TIC. Returns index records."""
    import lightkurve as lk
    res = lk.search_lightcurve(f"TIC {tic}")
    recs = []
    if len(res) == 0:
        return [dict(tic=tic, status="no products")]
    tab = res.table.to_pandas()
    tab["i"] = np.arange(len(tab))
    tab["pref"] = tab.author.map({a: k for k, a in enumerate(AUTHOR_PREF)}).fillna(99)
    tab = tab.sort_values(["pref", "mission"]).drop_duplicates("mission")
    for _, row in tab.head(n_lc).iterrows():
        sector = str(row.mission).replace("TESS Sector ", "")
        path = outdir / "lc" / f"{tic}_{row.author}_s{sector}.csv"
        rec = dict(tic=tic, author=row.author, sector=sector, exptime=row.exptime, path=str(path))
        if not path.exists():
            try:
                lc = res[int(row.i)].download(download_dir=str(outdir / "cache"))
                lc = lc.remove_nans()
                if "quality" in lc.colnames:
                    lc = lc[lc.quality == 0]
                lc = lc.normalize()
                _atomic_csv(pd.DataFrame(dict(time=np.asarray(lc.time.value, float),
                                              flux=np.asarray(lc.flux.value, float))), path)
                time.sleep(pause)
            except Exception as e:  # noqa: BLE001 -- record and continue
                rec["status"] = f"error: {type(e).__name__}: {e}"[:200]
                recs.append(rec)
                continue
        rec["status"] = "ok"
        recs.append(rec)
    return recs


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", required=True, help="CSV with a 'tic' column")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-lc", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "lc").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    recs = []
    for tic in pd.read_csv(args.select).tic:
        r = fetch_tic(int(tic), args.n_lc, out)
        recs += r
        print(tic, [(x.get("author"), x.get("sector"), x["status"][:20]) for x in r], flush=True)
    _atomic_csv(pd.DataFrame(recs), out / "ffi_index.csv")


if __name__ == "__main__":
    main()
