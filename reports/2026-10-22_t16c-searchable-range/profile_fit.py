"""T16c runtime profile: Stage 2 (comb fit) with and without the R/2 skip, on
identical inputs, for a few dev_medium stars at their assigned coverage.

Usage: python profile_fit.py [n_stars_per_class]  (writes data/profile_fit.csv)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import determinism  # noqa: E402
determinism.pin_blas_threads()
import cProfile, pstats, io, time, warnings  # noqa: E402,E401
import numpy as np, pandas as pd  # noqa: E402,E401
import coverage as cov  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from comb_fit import fit_rotation_period  # noqa: E402
from guesses import two_view_guesses  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402
import scorecard as sc  # noqa: E402

warnings.simplefilter("ignore")


def pooled_inputs(path, sectors):
    """Load one star at its assigned sectors and build the scorecard's Stage 1 pool and Stage 2 ACF."""
    pre = load_smarts_fits(path, sectors=sectors)
    t, f = pre.time, pre.flux
    lags, acf, counts = compute_acf(t, f, return_counts=True)
    lags1, acf1 = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
    pooled = []
    for name, fn in sc.discover_guess_functions().items():
        try:
            fl = sc._gap_filled(f) if name == "wavelet" else f
            pooled += two_view_guesses(fn, t, fl, (lags1, acf1), (lags, acf),
                                       n_guesses=sc.N_GUESSES.get(name, sc.N_GUESSES_DEFAULT))
        except Exception:  # noqa: BLE001
            pass
    return t, f, lags, acf, counts, pooled


def main(n_per=3, reps=3):
    """Time fit_rotation_period(max_period=None) vs 'auto' per star; profile one star."""
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv")
    rows = []
    prof_done = False
    for cls in ["single", "consecutive-2", "gapped-2", "consecutive-3-5", "full"]:
        sub = man[man.coverage_class == cls].head(n_per)
        for _, r in sub.iterrows():
            t, f, lags, acf, counts, pooled = pooled_inputs(
                ROOT / "data/smarts_dev" / r.path, cov.parse_pattern(r.sectors))
            out = {}
            for mode in (None, "auto"):
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    res = fit_rotation_period(lags, acf, pooled, acf_counts=counts, max_period=mode)
                    ts.append(time.perf_counter() - t0)
                out[str(mode)] = (min(ts), res)
            if not prof_done and cls == "single":
                pr = cProfile.Profile(); pr.enable()
                fit_rotation_period(lags, acf, pooled, acf_counts=counts, max_period=None)
                pr.disable(); s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
                (Path(__file__).parent / "data").mkdir(exist_ok=True)
                (Path(__file__).parent / "data/profile_single_star.txt").write_text(s.getvalue())
                prof_done = True
            a, b = out["None"][1], out["auto"][1]
            rows.append(dict(star_id=r.star_id, coverage_class=cls, true_period=r.true_period,
                             n_pooled=len(pooled), n_fit_before=a.n_candidates_tried,
                             n_fit_after=b.n_candidates_tried, n_beyond=len(b.beyond_reach_candidates),
                             reach=b.searchable_range["reach"],
                             fit_s_before=out["None"][0], fit_s_after=out["auto"][0],
                             P_before=a.best_fit.P if a.success else np.nan,
                             P_after=b.best_fit.P if b.success else np.nan))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).parent / "data/profile_fit.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
