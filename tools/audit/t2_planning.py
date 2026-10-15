"""Planning numbers for T2 (dev sets / holdout) from the SMARTS catalog. Not a set builder.

Computes, and writes to ``--outdir``:

- ``t2_pool_balance.csv``: parameter medians and KS distance between the two
  galaxy pools (SIM_ID < 800,000 = training galaxies; >= 800,000 = val/test
  galaxies; see ``noise_fingerprint.py --split-analysis``), to check the
  pools are parameter-identical.
- ``t2_strata_by_pool.csv``: per-stratum availability (audit period bin x
  amplitude-proxy tercile x shear class) in each pool.
- ``t2_draft_draws.csv``: for illustrative stratified draws of different sizes,
  how many distinct period bundles must be scanned, the extracted size, and the
  rough extraction time (one ~6 s streaming pass per bundle, measured here).

The draws are throwaway illustrations with a fixed seed; T2 owns the real
manifests.

Usage::

    python tools/audit/t2_planning.py --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smarts_data as sd  # noqa: E402
from catalog_to_table import load_catalog  # noqa: E402
from catalog_audit import AUDIT_PBIN_LABELS, amplitude_proxy, audit_pbin, shear_class  # noqa: E402

FILE_BYTES = 155_520     # every FITS file checked so far (55,390 in 10 bundle indexes)
BUNDLE_SCAN_S = 6.0      # measured: ~5.6-7.8 s per bundle stream on the Data_Vault drive
POOL_CUT = 800_000


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    out = Path(args.outdir)
    df = load_catalog(root)
    df["row"] = np.arange(len(df))
    df["pool"] = np.where(df.SIM_ID < POOL_CUT, "train_gal", "valtest_gal")
    df["pbin"] = audit_pbin(df.PERIOD.values)
    prox = amplitude_proxy(df)
    terc = np.quantile(prox, [1 / 3, 2 / 3])
    df["amp_tier"] = np.where(prox < terc[0], "low", np.where(prox < terc[1], "mid", "high"))
    df["shear"] = shear_class(df.DIFFROT.values)
    df["bundle"] = np.clip(np.ceil(df.PERIOD.values).astype(int), 1, 180)

    a, b = df[df.pool == "train_gal"], df[df.pool == "valtest_gal"]
    bal = []
    for c in ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL", "MINLAT", "MAXLAT", "DIFFROT",
              "TSPOT", "BFLY"]:
        bal.append(dict(param=c, median_train_gal=float(a[c].astype(float).median()),
                        median_valtest_gal=float(b[c].astype(float).median()),
                        ks_D=float(stats.ks_2samp(a[c].astype(float), b[c].astype(float)).statistic)))
    bal.append(dict(param="n_rows", median_train_gal=len(a), median_valtest_gal=len(b), ks_D=np.nan))
    pd.DataFrame(bal).to_csv(out / "t2_pool_balance.csv", index=False)

    st = df.groupby(["pool", "pbin", "amp_tier", "shear"]).size().rename("n").reset_index()
    st.to_csv(out / "t2_strata_by_pool.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for name, per_bin in [("dev_small ~42", 6), ("dev_medium ~1000", 143), ("holdout ~3500", 500)]:
        pick = []
        for pb in AUDIT_PBIN_LABELS:
            cell = df[(df.pool == "valtest_gal") & (df.pbin == pb)]
            pick.append(cell.sample(min(per_bin, len(cell)), random_state=rng.integers(1 << 31)))
        pick = pd.concat(pick)
        nb = pick.bundle.nunique()
        rows.append(dict(draw=name, n_files=len(pick), n_bundles=nb,
                         extracted_MB=len(pick) * FILE_BYTES / 1e6,
                         scan_minutes=nb * BUNDLE_SCAN_S / 60))
    rows.append(dict(draw="local budget 300 MB", n_files=int(300e6 // FILE_BYTES), n_bundles=np.nan,
                     extracted_MB=300.0, scan_minutes=np.nan))
    rows.append(dict(draw="full corpus", n_files=len(df), n_bundles=180,
                     extracted_MB=len(df) * FILE_BYTES / 1e6, scan_minutes=180 * BUNDLE_SCAN_S / 60))
    pd.DataFrame(rows).to_csv(out / "t2_draft_draws.csv", index=False)
    print(pd.DataFrame(bal).round(4).to_string())
    print(pd.DataFrame(rows).round(1).to_string())
    print(st[st.pool == "valtest_gal"].groupby("pbin").n.agg(["sum", "min"]))


if __name__ == "__main__":
    main()
