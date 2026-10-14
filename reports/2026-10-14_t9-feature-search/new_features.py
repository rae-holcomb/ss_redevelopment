#!/usr/bin/env python3
"""
new_features.py -- T9 pass 2: add prototype/new features to the pass-1 table.

Regenerates the same synthetic suite (deterministic), recomputes each
star's ACF (cheap), and evaluates the new features at every candidate's
FITTED period -- no refitting. Truth is only used afterwards, for the
`outcome` labels that pass 1 already attached.

New features (implemented in claude_files/ml_features.py, via
_alias_discrimination_features): acf_height_at_P, acf_half_lag_contrast,
acf_odd_even_contrast, fold_half_corr, fold_half_corr_at_2P.
Prototype-only (evaluated here, NOT added to ml_features -- see report):
    rvar_p2p_snr -- r_var / point-to-point white-noise estimate (star level)
    star_g_acf   -- highest ACF value beyond the ACF's first local minimum

Also writes a per-candidate timing for the new features.

Usage (repo root):
    /opt/anaconda3/envs/ss_local/bin/python reports/2026-10-14_t9-feature-search/new_features.py \
        --n 200 --seed 0 --indir runs/t9_feature_table
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))

from acf_utils import compute_acf  # noqa: E402
from ml_features import _alias_discrimination_features, _r_var  # noqa: E402
from synthetic import make_suite  # noqa: E402


def _p2p_noise(time: np.ndarray, flux: np.ndarray) -> float:
    """Robust white-noise estimate from point-to-point differences:
    1.4826 * median(|f[i+1] - f[i]|) / sqrt(2), using only time-adjacent
    valid pairs (pairs spanning a gap are excluded, as in _von_neumann_eta).
    Returns NaN with fewer than 10 usable pairs.
    """
    dt = np.median(np.diff(time))
    valid = np.isfinite(flux)
    good_pair = valid[:-1] & valid[1:] & (np.diff(time) <= 1.5 * dt)
    if good_pair.sum() < 10:
        return float("nan")
    d = np.diff(flux)[good_pair]
    return float(1.4826 * np.median(np.abs(d)) / np.sqrt(2.0))


def _rvar_p2p_snr(time: np.ndarray, flux: np.ndarray, r_var: float) -> float:
    """Variability amplitude over white noise: r_var / _p2p_noise. A direct,
    dimensionless signal-to-noise proxy for "is there enough modulation to
    measure a period at all" (decline prediction). NaN if the noise
    estimate is zero or undefined.
    """
    noise = _p2p_noise(time, flux)
    if not np.isfinite(noise) or noise <= 0 or not np.isfinite(r_var):
        return float("nan")
    return float(r_var / noise)


def star_g_acf(acf: np.ndarray) -> float:
    """Highest ACF value after the ACF's first local minimum (a star-level
    G_ACF analogue, Breton et al. 2021). NaN if the ACF never turns up."""
    d = np.diff(acf)
    up = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if len(up) == 0:
        return float("nan")
    return float(np.max(acf[up[0] + 1:]))


def main(argv=None):
    """Add pass-2 features to candidates_pass1.pkl -> candidates.pkl."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--indir", default="runs/t9_feature_table")
    args = p.parse_args(argv)
    indir = Path(args.indir)
    table = pd.read_pickle(indir / "candidates_pass1.pkl")
    suite = {lc.star_id: lc for lc in make_suite(args.n, seed=args.seed)}

    new_rows = []
    t_total, n_cand = 0.0, 0
    for sid, g in table.groupby("star_id", sort=False):
        lc = suite[sid]
        lags, acf = compute_acf(lc.time, lc.flux)
        t0 = time.perf_counter()
        snr = _rvar_p2p_snr(lc.time, lc.flux, _r_var(lc.time, lc.flux))
        gacf = star_g_acf(acf)
        t_total += time.perf_counter() - t0
        for idx, r in g.iterrows():
            P = r.get("fitted_P", np.nan)
            t0 = time.perf_counter()
            feats = (_alias_discrimination_features(lc.time, lc.flux, lags, acf, P)
                     if np.isfinite(P) else {})
            t_total += time.perf_counter() - t0
            n_cand += int(np.isfinite(P))
            feats["rvar_p2p_snr"] = snr
            feats["star_g_acf"] = gacf
            feats["_idx"] = idx
            new_rows.append(feats)
    add = pd.DataFrame(new_rows).set_index("_idx")
    out = table.join(add)
    out.to_pickle(indir / "candidates.pkl")
    print(f"wrote {indir / 'candidates.pkl'}: {out.shape}")
    print(f"new-feature cost: {t_total:.2f} s total, {1e3 * t_total / max(n_cand, 1):.2f} ms/candidate, "
          f"{t_total / table['star_id'].nunique():.3f} s/star")


if __name__ == "__main__":
    main()
