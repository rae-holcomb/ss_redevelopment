#!/usr/bin/env python3
"""Per-change microbenchmarks (T12): old vs new implementation of each optimized function, same inputs.

Usage: python reports/2026-11-12_t12-speed-up/microbench.py   (writes data/microbench.csv; needs data/smarts_dev)
The "old" implementations are the pre-T12 reference copies kept in tests/test_t12_speedups.py.
"""
import sys
import timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("claude_files", "benchmarks", "tests"):
    sys.path.insert(0, str(ROOT / sub))
import determinism  # noqa: E402
determinism.pin_blas_threads()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import acf_utils  # noqa: E402
import comb_fit as cf  # noqa: E402
import ml_features as mf  # noqa: E402
import scorecard as sc  # noqa: E402
import test_t12_speedups as ref  # noqa: E402


def best_of(fn, number, repeat=7):
    """Median over `repeat` timeit runs of seconds per call."""
    return float(np.median(timeit.repeat(fn, number=number, repeat=repeat)) / number)


def main():
    """Time old vs new on two dev_medium stars (one short-P full-year, one long-P assigned)."""
    args = type("A", (), dict(synthetic=None, manifest=ROOT / "datasets/dev_medium.csv", data_root="data/smarts_dev",
                              ignore_manifest_sectors=True, coverage="full"))()
    tasks = {t["star_id"]: t for t in sc.build_tasks(args)}
    rows = []
    for sid, P in (("smarts-tess-v1.0-001810", 0.4238), ("smarts-tess-v1.0-151481", 27.3422)):
        _, t, f, *_ = sc._load(tasks[sid])
        lags, acf, counts = acf_utils.compute_acf(t, f, return_counts=True)
        w = np.sqrt(counts / counts[0])
        min_lag = 3 * np.median(np.diff(lags))
        t0s = np.linspace(min_lag, min_lag + P, 200, endpoint=False)
        old = lambda: [cf.comb_score(lags, acf, P, x, n_max=5) for x in t0s]  # noqa: E731
        new = lambda: cf._comb_scores_over_t0(lags, acf, P, t0s, 5)  # noqa: E731
        rows.append(dict(star=sid, change="2 grid_search_t0 (200 phases)", old=best_of(old, 5), new=best_of(new, 5)))
        wins = cf._build_windows(lags, P, 0.9 * P, 8, max_points_per_window=100)
        st = cf._comb_model_setup(wins, lags, acf, P, 0.9 * P, weights=w)
        r_old, j_old = ref._loop_residual_jacobian(st, wins, lags, acf, w)
        x = st["x0"]
        rows.append(dict(star=sid, change="3 comb residual", old=best_of(lambda: r_old(x), 200),
                         new=best_of(lambda: st["residual"](x), 200)))
        rows.append(dict(star=sid, change="3 comb jacobian", old=best_of(lambda: j_old(x), 200),
                         new=best_of(lambda: st["jacobian"](x), 200)))

        def acf_old():
            ref._uncached_full_acf(np.asarray(f, float))
        rows.append(dict(star=sid, change="4 compute_acf, repeat call (cache hit)", old=best_of(acf_old, 5),
                         new=best_of(lambda: acf_utils.compute_acf(t, f, max_lag_frac=1 / 3), 5)))
        rows.append(dict(star=sid, change="5 _s_ph", old=best_of(lambda: ref._s_ph_loop(t, f, P), 5),
                         new=best_of(lambda: mf._s_ph(t, f, P), 5)))
    df = pd.DataFrame(rows)
    df["speedup"] = df.old / df.new
    out = Path(__file__).resolve().parent / "data" / "microbench.csv"
    df.to_csv(out, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
