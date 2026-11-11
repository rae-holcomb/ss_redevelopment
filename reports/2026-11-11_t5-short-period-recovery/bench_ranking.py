"""T5: interleaved A/B timing of the Stage-1 ranking change (old rank_by="power" vs new "hybrid").

For N dev_medium stars (every 1000/N-th row, full-year), runs the 3 pooled generators + comb fit
with each ranking, alternating A/B/A/B per repeat in one process, so CPU contention from other
jobs hits both arms equally. Reports per-star medians of generation and fit seconds.

Usage: python reports/2026-11-11_t5-short-period-recovery/bench_ranking.py [N] [repeats]
Writes data/bench_ranking.csv.
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "claude_files")]
import determinism  # noqa: E402

determinism.pin_blas_threads()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import guesses as G  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402


def one(t, f, lags, acf, counts, lags1, acf1, rank_by):
    """(generation seconds, fit seconds, n fitted) for the 3-method pool under one ranking."""
    t0 = time.perf_counter()
    pooled = []
    for fn, kw in ((G.guess_pairwise_histogram, {}), (G.guess_lombscargle, {"rank_by": rank_by}),
                   (G.guess_acf_fft, {"rank_by": rank_by})):
        pooled += G.two_view_guesses(fn, t, f, (lags1, acf1), (lags, acf), n_guesses=10, **kw)
    t1 = time.perf_counter()
    res = fit_rotation_period(lags, acf, pooled, acf_counts=counts)
    return t1 - t0, time.perf_counter() - t1, res.n_candidates_tried


def main(n=40, repeats=2):
    """Run the benchmark and save data/bench_ranking.csv."""
    warnings.simplefilter("ignore")
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv")
    rows = []
    for _, r in man.iloc[:: max(1, len(man) // n)].head(n).iterrows():
        pre = load_smarts_fits(ROOT / "data/smarts_dev" / r.path)
        t, f = pre.time, pre.flux
        lags, acf, counts = compute_acf(t, f, return_counts=True)
        lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
        for rep in range(repeats):
            for arm in (("power", "hybrid") if rep % 2 == 0 else ("hybrid", "power")):
                g, fi, nt = one(t, f, lags, acf, counts, lags1, acf1, arm)
                rows.append(dict(star_id=r.star_id, eval_pbin=r.eval_pbin, rank_by=arm, rep=rep,
                                 gen_seconds=g, fit_seconds=fi, n_tried=nt))
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "data/bench_ranking.csv", index=False)
    per = df.groupby(["star_id", "rank_by"])[["gen_seconds", "fit_seconds", "n_tried"]].median().unstack()
    print(per.median().round(4))
    tot = (per["gen_seconds"] + per["fit_seconds"])
    print("median total power %.3f s, hybrid %.3f s, median paired ratio %.3f" % (
        tot["power"].median(), tot["hybrid"].median(), (tot["hybrid"] / tot["power"]).median()))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
