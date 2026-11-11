#!/usr/bin/env python3
"""T27: is a T20 A+B noise stack a fair non-rotator? Two checks on the negative sources.

1. **Injected-signal imprint.** If the median over a source's stacking members kept a
   trace of their injected rotation, the stack's Lomb-Scargle top period would match
   members' injected PERIOD more often than chance. Per source: the fraction of its A+B
   members with |PERIOD - P1| / P1 < 2% (P1 = LS top peak of the ``all`` stack, from
   T20's ``ls_peaks.csv``). Null: the same fraction with P1 taken from a *different*
   random source (``--n-perm`` permutations), which keeps both period distributions.
2. **White-noise level.** Point-to-point scatter (1.4826 MAD of first differences / sqrt 2)
   of the stack vs its single member files (raw flux, ``flux/NNN_flux.npy``) for a random
   ``--n-p2p`` sources, plus T20's per-member residual p2p (member - stack) for all.

Outputs (``--outdir``, default ``reports/2026-11-10_t27-nonrotator-negatives/data``):
``fairness_sources.csv`` (per source), ``fairness_summary.json``.

Usage::

    python tools/audit/negatives_fairness.py            # ~2 min, needs the drive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import smarts_data as sd  # noqa: E402

MATCH_TOL = 0.02
OUTDIR = ROOT / "reports" / "2026-11-10_t27-nonrotator-negatives" / "data"


def p2p_ppm(x: np.ndarray) -> float:
    """Point-to-point (white) scatter in ppm: 1.4826 MAD of first differences / sqrt(2)."""
    d = np.diff(np.asarray(x, float))
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) * 1e6)


def match_fraction(p1: float, periods: np.ndarray, tol: float = MATCH_TOL) -> float:
    """Fraction of ``periods`` within ``tol`` (relative) of ``p1``; NaN if p1 is missing."""
    if not np.isfinite(p1) or len(periods) == 0:
        return np.nan
    return float(np.mean(np.abs(periods - p1) / p1 < tol))


def imprint(src: pd.DataFrame, members: dict, n_perm: int, seed: int) -> tuple:
    """Observed per-source match fractions and the permutation null (mean over sources per perm)."""
    obs = np.array([match_fraction(p, members[g]) for g, p in zip(src.galaxy, src.P1)])
    rng = np.random.default_rng(seed)
    p1 = src.P1.to_numpy()
    gal = src.galaxy.to_numpy()
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(src))
        # a derangement is not needed in expectation; drop self-pairs explicitly
        fr = [match_fraction(p1[j], members[g]) for g, j, i in zip(gal, perm, range(len(src))) if j != i]
        null.append(np.nanmean(fr))
    return obs, np.array(null)


def main(argv=None):
    """CLI entry point: run both checks on datasets/negatives_v0_sources.csv."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--sources", default=str(ROOT / "datasets" / "negatives_v0_sources.csv"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-p2p", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args(argv)
    root = sd.require_drive(sd.data_root(args.data_root))
    sdir = root / "extracted" / "galaxy_stack"
    src = pd.read_csv(args.sources)
    ls = pd.read_csv(sdir / "ls_peaks.csv")
    ls = ls[ls["stack"] == "all"].set_index("galaxy")
    src["P1"] = src.galaxy.map(ls.P1)
    src["pow1"] = src.galaxy.map(ls.pow1)
    src["amp1_ppm"] = src.galaxy.map(ls.amp1_ppm)
    src["stack_rms_ppm"] = src.galaxy.map(ls.rms_ppm)
    sel = pd.read_csv(sdir / "selection.csv")
    sel = sel[sel.galaxy.isin(src.galaxy) & sel.role.isin(["A", "B"])]
    from catalog_to_table import load_catalog
    cat = load_catalog(root)
    sel["PERIOD"] = sel.row.map(pd.Series(cat["PERIOD"].to_numpy(), index=np.arange(len(cat))))
    members = {g: d.PERIOD.to_numpy(float) for g, d in sel.groupby("galaxy")}
    obs, null = imprint(src, members, args.n_perm, args.seed)
    src["imprint_match_frac"] = obs
    src["n_members"] = src.galaxy.map(sel.groupby("galaxy").size())

    res = pd.read_csv(sdir / "residuals.csv")
    res = res[res.galaxy.isin(src.galaxy) & res.role.isin(["A", "B"])]
    src["member_resid_p2p_ppm_median"] = src.galaxy.map(res.groupby("galaxy").res_p2p_ppm.median())
    src["member_resid_rms_ppm_median"] = src.galaxy.map(res.groupby("galaxy").res_rms_ppm.median())

    stacks = np.load(sdir / "stacks.npy", mmap_mode="r")
    idx = pd.read_csv(sdir / "stack_index.csv")
    gpos = pd.Series(np.arange(len(idx)), index=idx.galaxy)
    src["stack_p2p_ppm"] = [p2p_ppm(stacks[gpos[g], 2]) for g in src.galaxy]
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(src.galaxy.to_numpy(), size=min(args.n_p2p, len(src)), replace=False)
    from galaxy_stack import load_selected
    want = sel[sel.galaxy.isin(sample)]
    rows, F = load_selected(sdir / "flux", want.row.to_numpy())
    rgal = pd.Series(want.galaxy.to_numpy(), index=want.row.to_numpy())[rows].to_numpy()
    mp2p = pd.Series([p2p_ppm(F[i]) for i in range(len(rows))]).groupby(rgal).median()
    src["member_p2p_ppm_median"] = src.galaxy.map(mp2p)
    src["p2p_ratio_stack_over_member"] = src.stack_p2p_ppm / src.member_p2p_ppm_median

    m = src.member_p2p_ppm_median.notna()
    summary = dict(
        n_sources=int(len(src)), n_sources_with_P1=int(src.P1.notna().sum()), match_tol=MATCH_TOL,
        imprint_mean_match_frac=float(np.nanmean(obs)),
        imprint_null_mean=float(null.mean()), imprint_null_p95=float(np.percentile(null, 95)),
        imprint_null_p05=float(np.percentile(null, 5)),
        imprint_obs_over_null=float(np.nanmean(obs) / null.mean()),
        imprint_perm_p_value=float((np.sum(null >= np.nanmean(obs)) + 1) / (len(null) + 1)),
        imprint_by_set={k: float(np.nanmean(v.imprint_match_frac)) for k, v in src.groupby("eval_set")},
        n_p2p_sources=int(m.sum()),
        p2p_ratio_median=float(src.p2p_ratio_stack_over_member[m].median()),
        p2p_ratio_p05=float(src.p2p_ratio_stack_over_member[m].quantile(0.05)),
        p2p_ratio_p95=float(src.p2p_ratio_stack_over_member[m].quantile(0.95)),
        stack_p2p_ppm_median=float(src.stack_p2p_ppm.median()),
        member_resid_p2p_ppm_median=float(src.member_resid_p2p_ppm_median.median()),
        member_resid_rms_ppm_median=float(src.member_resid_rms_ppm_median.median()))
    args.outdir.mkdir(parents=True, exist_ok=True)
    with sd.atomic_path(args.outdir / "fairness_sources.csv") as tmp:
        src.to_csv(tmp, index=False)
    np.save(args.outdir / "fairness_imprint_null.npy", null)
    with sd.atomic_path(args.outdir / "fairness_summary.json") as tmp:
        tmp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
