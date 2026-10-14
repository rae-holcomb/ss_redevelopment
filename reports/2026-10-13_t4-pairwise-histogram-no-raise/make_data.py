"""Build the small data files this report's notebook reads.

Copies the pairwise_histogram rows and final outcomes from the before/after
scorecard runs, and regenerates ACFs for a few stars where the old code
raised, so the notebook can show *why* they have <2 ACF peaks.

Run from the repo root:
    /opt/anaconda3/envs/ss_local/bin/python reports/2026-10-13_t4-pairwise-histogram-no-raise/make_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "claude_files"))
from acf_utils import compute_acf  # noqa: E402
from guesses import _acf_peak_candidates  # noqa: E402
from synthetic import make_suite  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main():
    """Write pairwise_before_after.csv, final_before_after.csv and example_acfs.npz."""
    OUT.mkdir(exist_ok=True)
    cols = ["star_id", "true_period", "period_bin", "outcome", "rank", "n_candidates", "error", "seconds"]
    pm = {k: pd.read_csv(REPO / f"runs/t4_{k}/per_method.csv") for k in ("before", "after")}
    pw = pd.merge(*(pm[k].query("method == 'pairwise_histogram'")[cols] for k in ("before", "after")),
                  on=["star_id", "true_period", "period_bin"], suffixes=("_before", "_after"))
    ps = {k: pd.read_csv(REPO / f"runs/t4_{k}/per_star.csv") for k in ("before", "after")}
    fcols = ["star_id", "true_period", "period_bin", "final_period", "final_outcome", "n_pooled", "seconds", "meta_snr", "meta_inclination_deg", "meta_spot_lifetime_days"]
    fin = pd.merge(ps["before"][fcols], ps["after"][fcols].drop(columns=["true_period", "period_bin", "meta_snr", "meta_inclination_deg", "meta_spot_lifetime_days"]),
                   on="star_id", suffixes=("_before", "_after"))
    pw = pw.merge(fin[["star_id", "meta_snr"]], on="star_id")

    # Regenerate ACFs (same suite the scorecard used) and count the ACF peaks
    # guess_pairwise_histogram would see with its default settings.
    lcs = {lc.star_id: lc for lc in make_suite(200, seed=0)}
    n_peaks, examples = [], {}
    raised = set(pw.loc[pw.error_before.notna(), "star_id"])
    for sid, lc in lcs.items():
        lags, acf = compute_acf(lc.time, lc.flux)
        dt = np.median(np.diff(lags))
        prom = 5.0 * np.std(np.diff(acf, 2))
        pl, ph, _ = _acf_peak_candidates(lags, acf, min_lag=3 * dt, prominence=prom)
        n_peaks.append(dict(star_id=sid, n_acf_peaks=len(pl), prominence=prom, dt=dt))
        if sid in raised:
            examples[sid] = dict(lags=lags, acf=acf, peak_lags=pl, prominence=prom, true_period=lc.true_period)
    pw = pw.merge(pd.DataFrame(n_peaks), on="star_id")
    pw.to_csv(OUT / "pairwise_before_after.csv", index=False)
    fin.to_csv(OUT / "final_before_after.csv", index=False)

    # one example per period bin among stars that raised
    pick = pw[pw.error_before.notna()].sort_values("star_id").groupby("period_bin").head(1).star_id.tolist()
    np.savez_compressed(OUT / "example_acfs.npz", **{
        f"{sid}__{k}": np.asarray(v) for sid in pick for k, v in examples[sid].items()})
    print("examples:", pick)


if __name__ == "__main__":
    main()
