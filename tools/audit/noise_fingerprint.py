"""Detect SMARTS files that share the same TESS-galaxy noise background (leakage check).

SMARTS-TESS adds one of ~2,000 real galaxy light curves (~1,000 train,
~1,000 val/test per Claytor+2022) to each of the 1M simulations, but neither
the catalog nor the FITS headers record which galaxy. Files that share a
galaxy have identical high-frequency noise. Fingerprint = first difference
of the raw flux (removes the slowly varying star signal for P >> 30 min),
with linearly interpolated gap runs and constant end-fills zeroed (those are
deterministic and shared by every galaxy with the same gap template, which
would fake a match), then standardized. The Pearson correlation of two
fingerprints is ~0 +/- 1/sqrt(17520) ~ 0.008 for different galaxies and
large (-> 1 at low stellar amplitude) for the same galaxy.

Outputs (``--outdir``):

- ``fingerprint_pairs.csv``: every pair with corr > ``--report-thresh``
- ``fingerprint_maxcorr.csv``: per-file maximum correlation and partner
- ``fingerprint_summary.json``: pair counts above thresholds and a
  collision-rate estimate of the number of distinct galaxies
  (N ~ C(n, 2) / n_pairs, valid when each file draws a galaxy uniformly).

The full corpus would need an all-pairs (or template-library) match: a batch job
for Rae, not this script as written (O(n^2) memory).

Usage::

    python tools/audit/noise_fingerprint.py --inputs DIR_OR_GLOB ... --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lc_spotcheck import interp_runs  # noqa: E402


def fingerprint(path: Path) -> np.ndarray:
    """Standardized first-difference flux with interpolated runs and end-fills zeroed."""
    with fits.open(path) as h:
        f = np.asarray(h["LIGHTCURVE"].data["flux"], dtype=np.float64)
    d = np.diff(f)
    mask = np.ones_like(d, bool)
    for s, e in interp_runs(f, min_run=3):
        mask[max(s - 1, 0):e + 1] = False
    mask &= d != 0  # constant (mean-filled) stretches
    x = np.where(mask, d, 0.0)
    good = x[mask]
    # Robust standardization so a few outliers don't dominate the correlation.
    med = np.median(good)
    scale = 1.4826 * np.median(np.abs(good - med))
    x = np.where(mask, np.clip((d - med) / scale, -8, 8), 0.0)
    return x / np.linalg.norm(x)


def collect_paths(inputs: list, stride: int = 1) -> list:
    """Expand directories / globs into a sorted, de-duplicated list of FITS paths (every ``stride``-th)."""
    paths = []
    for s in inputs:
        p = Path(s)
        paths += sorted(p.glob("smarts-tess-v1.0-*.fits")) if p.is_dir() else \
            [Path(x) for x in sorted(glob.glob(s))]
    seen, out = set(), []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out[::stride]


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", required=True, help="directories or globs of FITS files")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth file of the sorted list")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--report-thresh", type=float, default=0.1)
    ap.add_argument("--split-analysis", action="store_true",
                    help="also test whether SIM_ID partitions the galaxy pool (needs the drive)")
    args = ap.parse_args()
    paths = collect_paths(args.inputs, args.stride)
    n = len(paths)
    print(f"fingerprinting {n} files", flush=True)
    X = np.stack([fingerprint(p) for p in paths]).astype(np.float32)
    C = X @ X.T
    np.fill_diagonal(C, -np.inf)
    rows = [int(p.name[-11:-5]) for p in paths]
    iu = np.triu_indices(n, 1)
    vals = C[iu]
    keep = vals > args.report_thresh
    pairs = pd.DataFrame(dict(row_a=np.array(rows)[iu[0][keep]], row_b=np.array(rows)[iu[1][keep]],
                              corr=vals[keep])).sort_values("corr", ascending=False)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "fingerprint_pairs.csv", index=False)
    j = np.argmax(C, axis=1)
    pd.DataFrame(dict(row=rows, max_corr=C[np.arange(n), j], partner=np.array(rows)[j])).to_csv(
        out / "fingerprint_maxcorr.csv", index=False)
    hist_edges = np.linspace(-0.1, 1.0, 111)
    h, _ = np.histogram(vals, hist_edges)
    pd.DataFrame(dict(lo=hist_edges[:-1], hi=hist_edges[1:], n_pairs=h)).to_csv(
        out / "fingerprint_corr_hist.csv", index=False)
    summ = dict(n_files=n, n_pairs_total=comb(n, 2),
                null_sigma_expected=float(1 / np.sqrt(X.shape[1])),
                null_sigma_observed_robust=float(1.4826 * np.median(np.abs(vals - np.median(vals)))),
                pairs_above={str(t): int((vals > t).sum()) for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)})
    for t in (0.2, 0.3, 0.5):
        k = summ["pairs_above"][str(t)]
        summ[f"n_galaxies_est_thresh_{t}"] = (comb(n, 2) / k) if k else None
    if args.split_analysis:
        summ["split"] = split_analysis(pairs, np.array(rows), out)
    (out / "fingerprint_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def split_analysis(pairs: pd.DataFrame, rows: np.ndarray, out: Path, match_thresh: float = 0.8) -> dict:
    """Test whether the catalog 'Simulation Number' (SIM_ID) partitions the galaxy pool.

    For candidate cuts, counts same-galaxy pairs (corr > ``match_thresh``) whose
    two SIM_IDs fall on opposite sides, versus the number expected if galaxies
    were assigned independently of SIM_ID. Then estimates the galaxy-pool size
    on each side of SIM_ID = 800,000 from the collision rate, with a Poisson
    95% interval. Writes ``fingerprint_split.csv`` (per-cut counts) and returns
    the per-side estimates. Needs the catalog on the drive.
    """
    from scipy import stats
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from catalog_to_table import load_catalog
    sim = load_catalog().SIM_ID.values
    s = sim[rows]
    hi = pairs[pairs["corr"] > match_thresh]
    sa, sb = sim[hi.row_a.values], sim[hi.row_b.values]
    recs = []
    for cut in range(500_000, 1_000_001, 10_000):
        f = np.mean(s < cut)
        recs.append(dict(cut=cut, cross_pairs=int(np.sum((sa < cut) != (sb < cut))),
                         expected_if_random=float(len(hi) * 2 * f * (1 - f))))
    pd.DataFrame(recs).to_csv(out / "fingerprint_split.csv", index=False)
    res = {"match_thresh": match_thresh, "n_match_pairs": int(len(hi))}
    for name, side, both in (("sim_lt_800k", s < 800_000, (sa < 800_000) & (sb < 800_000)),
                             ("sim_ge_800k", s >= 800_000, (sa >= 800_000) & (sb >= 800_000))):
        n, k = int(side.sum()), int(both.sum())
        lo, up = stats.chi2.ppf(0.025, 2 * k) / 2, stats.chi2.ppf(0.975, 2 * k + 2) / 2
        res[name] = dict(n_files=n, n_pairs=k, n_galaxies_est=comb(n, 2) / k if k else None,
                         ci95=[comb(n, 2) / up, comb(n, 2) / lo if k else None])
    idx = {r: i for i, r in enumerate(rows)}
    g = coo_matrix((np.ones(len(hi)), ([idx[r] for r in hi.row_a], [idx[r] for r in hi.row_b])),
                   shape=(len(rows), len(rows)))
    nc, lab = connected_components(g, directed=False)
    res["n_noise_groups"] = int(nc)
    res["group_size_counts"] = {int(k): int(v) for k, v in enumerate(np.bincount(np.bincount(lab))) if v}
    pd.DataFrame(dict(row=rows, sim_id=s, noise_group=lab)).to_csv(out / "fingerprint_groups.csv",
                                                                    index=False)
    return res


if __name__ == "__main__":
    main()
