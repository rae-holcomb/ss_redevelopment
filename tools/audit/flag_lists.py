"""Write T1 exclusion/flag lists: per-file (from scanned files) and rule-based (whole catalog).

Outputs to ``--outdir``:

- ``exclusion_candidates.csv``: one row per *scanned* file (``flux_outliers.csv``)
  with corrupt galaxy-noise orbits; columns row, member (path inside the
  period bundle, i.e. relative to ``tess_retry/<bundle tar>``), reason, action.
- ``flag_rules.csv``: catalog-level rules (hard-but-valid stars), with the
  number and fraction of the 1M corpus each one flags. Per-row flags for the
  rules live in ``<root>/extracted/catalog/catalog_derived.npz`` on the drive.

Usage::

    python tools/audit/flag_lists.py --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smarts_data as sd  # noqa: E402
from catalog_to_table import load_catalog  # noqa: E402
from catalog_audit import BASELINE_13  # noqa: E402

CORRUPT_THRESH = 0.1


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    out = Path(args.outdir)
    df = load_catalog(root)
    with np.load(Path(root, sd.EXTRACT_DIR, "catalog", "catalog_derived.npz")) as z:
        der = {k: z[k] for k in z.files}

    fo = pd.read_csv(out / "flux_outliers.csv")
    bad = fo[fo.max_abs_dev > CORRUPT_THRESH].copy()
    bundle = np.clip(np.ceil(df.PERIOD.values[bad.row.values]).astype(int), 1, 180)
    bad["member"] = [f"{b:03d}/{f}" for b, f in zip(bundle, bad.file)]
    bad["reason"] = [f"corrupt galaxy-noise orbits {o}: max|flux-1|={m:.3g}"
                     for o, m in zip(bad.orbits_hit, bad.max_abs_dev)]
    bad["action"] = ("flag: load_smarts_fits drops these orbits; exclude if using raw flux "
                     "or the WAVELET HDU; same galaxy => all files sharing it are affected")
    bad[["row", "member", "reason", "action", "max_abs_dev"]].sort_values("row").to_csv(
        out / "exclusion_candidates.csv", index=False)

    p, p_hi = df.PERIOD.values, der["P_HI"]
    rules = [
        ("P_eq > B/2 (B = 357.04 d)", p > BASELINE_13 / 2,
         "fewer than 2 cycles in the 13-sector baseline: unmeasurable; keep only as decline/negative examples"),
        ("P_eq > B/3", p > BASELINE_13 / 3,
         "beyond the current few-peak comb-fit reach (~B/3)"),
        ("active band P_hi > B/2", p_hi > BASELINE_13 / 2,
         "part of the spot-period band is unmeasurable even when P_eq is in reach"),
        ("active band disjoint from P_eq +/-10%", der["BAND_DISJOINT"],
         "P_eq is never a spot period; P_eq label is not directly observable (diffrot outcome only)"),
        ("active band extends beyond P_eq +/-10%", der["BAND_EXTENDS"],
         "measured period can legitimately differ from P_eq by >REL_TOL"),
        ("amplitude proxy < 0.05", der["AMP_PROXY"] < 0.05,
         "low activity and/or near pole-on: likely undetectable (proxy only; calibrate in phase 2)"),
        ("OVERLAP > CYCLE", df.OVERLAP.values > df.CYCLE.values,
         "valid but outside the documented range: overlapping cycles, weak cycle modulation"),
    ]
    pd.DataFrame([dict(rule=r, n=int(m.sum()), frac=float(m.mean()), reason=why)
                  for r, m, why in rules]).to_csv(out / "flag_rules.csv", index=False)
    print(f"{len(bad)} exclusion candidates of {len(fo)} scanned files")
    print(pd.read_csv(out / "flag_rules.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
