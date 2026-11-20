"""Summaries of the Stage-2 screen (runs/t32_clean/s2/*.csv) against the clean _v7 baselines.

Writes data/screen_summary.csv (per variant x mode: success / alias / wrong / none,
gains and losses vs _v7, none->wrong, median fit seconds) and
data/screen_by_bin.csv (outcome fractions per period bin).
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S2 = ROOT / "runs/t32_clean/s2"
S = {"correct": "succ", "diffrot": "succ", "half": "alias", "double": "alias", "wrong": "wrong", "none": "none"}
BASE = {"fy": "fullyear", "as": "assigned"}
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]


def baseline(mode):
    """Clean _v7 per-star table for one mode."""
    return pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_clean_{BASE[mode]}_v7_tables/per_star.csv")


def main():
    """Build the two summary tables."""
    rows, brows = [], []
    for f in sorted(S2.glob("*_fy.csv")) + sorted(S2.glob("*_as.csv")):
        v, mode = f.stem.rsplit("_", 1)
        d = pd.read_csv(f)
        b = baseline(mode)[["star_id", "final_outcome"]].rename(columns={"final_outcome": "v7"})
        m = d.merge(b, on="star_id")
        o, ob = m.final_outcome.map(S), m.v7.map(S)
        n = len(m)
        rows.append(dict(variant=v, mode=mode, n=n, success=(o == "succ").mean(), alias=(o == "alias").mean(),
                         wrong=(o == "wrong").mean(), none=(o == "none").mean(),
                         gained=int(((o == "succ") & (ob != "succ")).sum()),
                         lost=int(((o != "succ") & (ob == "succ")).sum()),
                         none2wrong=int(((ob == "none") & (o == "wrong")).sum()),
                         wrong2none=int(((ob == "wrong") & (o == "none")).sum()),
                         fit_s_median=d.fit_seconds.median(), fit_s_mean=d.fit_seconds.mean()))
        for pb in BINS:
            x = o[m.period_bin == pb]
            brows.append(dict(variant=v, mode=mode, period_bin=pb, n=len(x), success=(x == "succ").mean(),
                              alias=(x == "alias").mean(), wrong=(x == "wrong").mean()))
    for mode in BASE:
        b = baseline(mode)
        o = b.final_outcome.map(S)
        rows.append(dict(variant="_v7", mode=mode, n=len(b), success=(o == "succ").mean(),
                         alias=(o == "alias").mean(), wrong=(o == "wrong").mean(), none=(o == "none").mean(),
                         gained=0, lost=0, none2wrong=0, wrong2none=0, fit_s_median=b.fit_seconds.median(),
                         fit_s_mean=b.fit_seconds.mean()))
        for pb in BINS:
            x = o[b.period_bin == pb]
            brows.append(dict(variant="_v7", mode=mode, period_bin=pb, n=len(x), success=(x == "succ").mean(),
                              alias=(x == "alias").mean(), wrong=(x == "wrong").mean()))
    (HERE / "data").mkdir(exist_ok=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(HERE / "data/screen_summary.csv", index=False)
    pd.DataFrame(brows).round(4).to_csv(HERE / "data/screen_by_bin.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.sort_values(["mode", "success"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
