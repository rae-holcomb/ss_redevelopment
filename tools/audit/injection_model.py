"""T20 phase 2 (a): was the spot signal injected additively or multiplicatively into the noise light curve?

Uses TICs whose noise light curve has corrupt orbits (|g - 1| up to ~10^2):
there the two models make opposite predictions for a member file f = g (+ or x) s.

- additive  f = g + (s - 1):  f - g  is the same size in corrupt and clean orbits,
  while f / g - 1 shrinks or grows with 1 / g;
- multiplicative f = g * s:   f / g - 1 is the same size everywhere,
  while f - g scales with g.

For each check member (strong injected signal) of a corrupt TIC, with the
stack estimate g (``galaxy_stack.py``), this measures the robust rms of both
residuals inside corrupt cadences (|g - 1| > ``--dev``) and in clean ones, and
reports ``ratio = rms_corrupt / rms_clean`` for each model: the right model
gives ~1.

Usage::

    $PY tools/audit/injection_model.py --out reports/<folder>/data/injection_model.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galaxy_stack as gs  # noqa: E402
import smarts_data as sd  # noqa: E402


def residual_ratios(f: np.ndarray, g: np.ndarray, dev: float) -> dict:
    """rms(corrupt)/rms(clean) of f - g (additive) and f / g - 1 (multiplicative); robust rms."""
    bad = np.abs(g - 1) > dev
    clean = (np.abs(g - 1) < 0.05) & np.isfinite(f)
    out = dict(n_corrupt=int(bad.sum()), g_abs_dev_median_corrupt=float(np.median(np.abs(g[bad] - 1))))
    for name, r in (("add", f - g), ("mul", f / g - 1)):
        rc, rb = gs.robust_scatter(r[clean]), gs.robust_scatter(r[bad])
        out[f"{name}_rms_clean_ppm"], out[f"{name}_rms_corrupt_ppm"] = rc * 1e6, rb * 1e6
        out[f"{name}_ratio"] = rb / rc if rc > 0 else np.nan
    return out


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dev", type=float, default=0.3, help="|g - 1| threshold for 'corrupt' cadences")
    ap.add_argument("--min-corrupt", type=int, default=200, help="min corrupt cadences per TIC")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    sdir = Path(root, sd.EXTRACT_DIR, gs.STACK_DIR)
    S = np.load(sdir / "stacks.npy", mmap_mode="r")
    idx = pd.read_csv(sdir / "stack_index.csv")
    sel = pd.read_csv(sdir / "selection.csv")
    g_all = {r.galaxy: np.asarray(S[i, 2], np.float64) for i, r in idx.iterrows()}
    corrupt = [g for g, s in g_all.items() if (np.abs(s - 1) > args.dev).sum() >= args.min_corrupt]
    chk = sel[sel.galaxy.isin(corrupt) & (sel.role == "check")]
    rows, F = gs.load_selected(sdir / "flux", chk.row.values)
    gal = chk.set_index("row").galaxy
    recs = []
    for r, f in zip(rows, F.astype(np.float64)):
        recs.append(dict(row=int(r), galaxy=int(gal[r]), **residual_ratios(f, g_all[gal[r]], args.dev)))
    df = pd.DataFrame(recs).merge(sel[["row", "E", "E_rank_frac"]], on="row")
    df.to_csv(args.out, index=False)
    strong = df[df.E_rank_frac > 0.9]
    print(f"{len(corrupt)} corrupt TICs, {len(df)} check members ({len(strong)} top-decile E)")
    print("median ratio  additive: %.3f   multiplicative: %.3f   (top-decile E: %.3f / %.3f)" % (
        df.add_ratio.median(), df.mul_ratio.median(), strong.add_ratio.median(), strong.mul_ratio.median()))


if __name__ == "__main__":
    main()
