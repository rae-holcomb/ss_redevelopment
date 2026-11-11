"""T5 phase 1: per-star Stage-1 table, old (raw-power) vs new (hybrid) peak ranking.

Reads the peak dump written by dump_peaks.py (every acf_fft / lombscargle peak, both
ACF views, both coverage modes) and, for every dev_medium star and mode, records:
  - rank_<m>: rank of the first success peak (correct or diffrot) in raw-power order
    (acf_fft: short B/3 view), and n_peaks_<m>;
  - old_<m> / new_<m>: does the method's two-view top-10 contain a success, with
    rank_by="power" (pre-T5) vs "hybrid" (guesses._select_peaks, the shipped code);
  - chance_old_<m> / chance_new_<m>: same lists scored against a random other star's
    truth from the same period bin and mode (shuffled-truth chance level, seed 0);
  - pairwise (unchanged by T5) from the T10 dev_medium tables, and snr_rot from T26.

Usage:
    python reports/2026-11-11_t5-short-period-recovery/phase1_tables.py <peak_dump_dir>
Writes data/phase1_stage1.csv.
"""
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
import evaluation as ev  # noqa: E402
from guesses import _select_peaks  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
MAIN = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment")  # T10 tables live in the main checkout
T26 = OUT / "t26_snr_rot.csv"  # copied from branch claude/t26-detectability-7f7523 (see README)
SUC = set(ev.SUCCESS_OUTCOMES)


def hit(periods, truth_row) -> bool:
    """True if any period is a success for the star described by truth_row."""
    pr = ev.active_band_period_range(truth_row.true_period, truth_row.shear, truth_row.minlat, truth_row.maxlat)
    return any(ev.classify_period(p, truth_row.true_period, ev.REL_TOL, pr) in SUC for p in periods)


def select(d, rank_by, n=10) -> np.ndarray:
    """P0 of the n peaks chosen from a dumped peak dict under rank_by."""
    if d is None:
        return np.array([])
    idx = _select_peaks(1.0 / d["P0"], d["power"], np.arange(len(d["P0"])), n, rank_by=rank_by)
    return d["P0"][idx]


def two_view(rec, rank_by) -> np.ndarray:
    """acf_fft two-view list: short view top-n plus long-view picks beyond half the short lag range."""
    s = select(rec["acf_fft_short"], rank_by)
    lo = select(rec["acf_fft_long"], rank_by)
    return np.concatenate([s, lo[lo > 0.5 * rec["lag1"]]])


def first_success_rank(d, truth_row):
    """1-based rank (raw-power order) of the first success peak, or NaN."""
    if d is None:
        return np.nan
    order = np.argsort(-d["power"])
    pr = ev.active_band_period_range(truth_row.true_period, truth_row.shear, truth_row.minlat, truth_row.maxlat)
    for k, i in enumerate(order, start=1):
        if ev.classify_period(d["P0"][i], truth_row.true_period, ev.REL_TOL, pr) in SUC:
            return k
    return np.nan


def main(dump_dir: str) -> None:
    """Build data/phase1_stage1.csv from the peak dump."""
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv").set_index("star_id")
    t26 = pd.read_csv(T26).set_index("star_id")
    pw = {m: pd.read_csv(MAIN / f"runs/t10_dev_medium_{d}/stars.csv.gz").set_index("star_id")
          .stage1_pairwise_histogram_outcome.isin(SUC) for m, d in (("fy", "fullyear"), ("asg", "assigned"))}
    rows, lists = [], []
    for fn in sorted(glob.glob(f"{dump_dir}/*.pkl")):
        rec = pickle.load(open(fn, "rb"))
        sid, mode = rec["sid"], rec["mode"]
        r = man.loc[sid]
        L = {("acf_fft", "old"): two_view(rec, "power"), ("acf_fft", "new"): two_view(rec, "hybrid"),
             ("lombscargle", "old"): select(rec["lombscargle"], "power"),
             ("lombscargle", "new"): select(rec["lombscargle"], "hybrid")}
        row = dict(star_id=sid, mode=mode, eval_pbin=r.eval_pbin, true_period=r.true_period,
                   snr_rot=t26.loc[sid, f"snr_rot_{mode}"], n_points=rec["N"], baseline=rec["B"],
                   stage1_lag_range=rec["lag1"], pairwise=bool(pw[mode].get(sid, False)))
        for m, key in (("acf_fft", "acf_fft_short"), ("lombscargle", "lombscargle")):
            row[f"rank_{m}"] = first_success_rank(rec[key], r)
            row[f"n_peaks_{m}"] = 0 if rec[key] is None else len(rec[key]["P0"])
            for v in ("old", "new"):
                row[f"{v}_{m}"] = hit(L[(m, v)], r)
        rows.append(row)
        lists.append(L)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for col in [f"chance_{v}_{m}" for v in ("old", "new") for m in ("acf_fft", "lombscargle")]:
        df[col] = False
    for (b, mode), g in df.groupby(["eval_pbin", "mode"]):
        perm = rng.permutation(g.index.to_numpy())
        for i, j in zip(g.index, perm):
            other = man.loc[df.at[j, "star_id"]]
            for m in ("acf_fft", "lombscargle"):
                for v in ("old", "new"):
                    df.at[i, f"chance_{v}_{m}"] = hit(lists[i][(m, v)], other)
    df.to_csv(OUT / "phase1_stage1.csv", index=False)
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main(sys.argv[1])
