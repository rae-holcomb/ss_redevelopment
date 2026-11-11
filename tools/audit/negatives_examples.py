#!/usr/bin/env python3
"""T27: save light curve, ACF and Lomb-Scargle for a few non-rotator false positives (for the report).

Picks ``--n`` answered negatives spread over reported-period bins (largest-count bins first,
seeded), reloads each exactly as the pipeline did (``load_smarts_fits`` with the manifest's
sectors), and stores time/flux, the Stage 2 ACF and an LS periodogram in one ``.npz`` plus a
CSV of the picks. Small output (a few MB), so it lives in the report's ``data/``.

Usage::

    python tools/audit/negatives_examples.py --neg-dev <drive>/extracted/t27_negatives/dev_medium
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(HERE))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import coverage as cov  # noqa: E402
import evaluation as ev  # noqa: E402
import smarts_data as sd  # noqa: E402

OUTDIR = ROOT / "reports" / "2026-11-10_t27-nonrotator-negatives" / "data"


def pick_examples(stars: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` answered negatives, one per reported-period bin in turn (one draw per source)."""
    a = stars[stars.final_outcome != ev.NONE].copy()
    a["rbin"] = a.final_period.map(ev.period_bin)
    a = a.sample(frac=1, random_state=seed).drop_duplicates("noise_tic")
    order = a.rbin.value_counts().index
    picks, i = [], 0
    while len(picks) < n and i < 10:
        for b in order:
            g = a[(a.rbin == b) & ~a.star_id.isin([p.star_id for p in picks])]
            if len(g) > i and len(picks) < n:
                picks.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picks)


def main(argv=None):
    """CLI entry point."""
    from astropy.timeseries import LombScargle
    from acf_utils import compute_acf
    from preprocessing import load_smarts_fits
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--neg-dev", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)
    root = sd.require_drive(sd.data_root(args.data_root))
    stars = pd.read_csv(Path(args.neg_dev) / "stars.csv.gz", low_memory=False)
    man = pd.read_csv(ROOT / "datasets" / "negatives_v0.csv", keep_default_na=False).set_index("star_id")
    picks = pick_examples(stars, args.n, args.seed)
    arrays = {}
    for k, r in enumerate(picks.itertuples()):
        m = man.loc[r.star_id]
        kw = {"sectors": list(cov.parse_pattern(m.sectors))} if m.sectors else {}
        lc = load_smarts_fits(root / m.path, **kw)
        t, f = lc.time, lc.flux
        lags, acf = compute_acf(t, f)
        ok = np.isfinite(f)
        fr = np.geomspace(1 / 60, 1 / 0.1, 4000)
        pw = LombScargle(t[ok], f[ok]).power(fr)
        arrays.update({f"t{k}": t.astype(np.float32), f"f{k}": f.astype(np.float32),
                       f"lag{k}": lags.astype(np.float32), f"acf{k}": acf.astype(np.float32),
                       f"freq{k}": fr.astype(np.float32), f"pow{k}": pw.astype(np.float32)})
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outdir / "fp_examples.npz", **arrays)
    cols = ["star_id", "noise_tic", "source_class", "assigned_coverage_class", "n_sectors", "final_period",
            "decline_kind"]
    picks[cols].to_csv(args.outdir / "fp_examples.csv", index=False)
    print(picks[cols].to_string())


if __name__ == "__main__":
    main()
