"""P44 step 2: the mean_height_snr cutoff sweep on dev_medium_b_clean (tuning set), both modes.

Reads runs/p44/sweep_B_{mode}.pkl (p44lib.py). Checks that the old definition (local_tol_trim) at 1.0
reproduces the dev_medium_b_clean _v8 per-star outcomes exactly, then tabulates success / wrong / alias /
none per (variant, cutoff) and applies the selection rule (Rae, task brief): the cutoff whose wrong rate
matches B's _v8 wrong rate (wrong <= _v8 wrong) with the highest success; ties -> lower wrong, then higher
cutoff. Full-year decides if the modes disagree.

Writes data/sweep_B.csv, data/cands_B_{mode}.csv.gz, data/chosen_cutoff.json.
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
S = {"correct", "diffrot"}
A = {"half", "double"}


def load(mode: str, tag: str = "B"):
    """Sweep records for one mode."""
    return pickle.load(open(ROOT / f"runs/p44/sweep_{tag}_{mode}.pkl", "rb"))


def sweep_table(recs, mode: str) -> pd.DataFrame:
    """success/wrong/alias/none per (variant, cutoff)."""
    rows = []
    keys = sorted(recs[0]["sweep"])
    N = len(recs)
    for v, cut in keys:
        out = [r["sweep"][(v, cut)][1] for r in recs]
        s = sum(o in S for o in out)
        w = sum(o == "wrong" for o in out)
        a = sum(o in A for o in out)
        rows.append(dict(mode=mode, variant=v, cutoff=cut, n=N, success=s / N, wrong=w / N, alias=a / N,
                         none=sum(o == "none" for o in out) / N, n_success=s, n_wrong=w,
                         precision=s / max(N - sum(o == "none" for o in out), 1)))
    return pd.DataFrame(rows)


def choose(tab: pd.DataFrame, target_wrong: float) -> pd.Series:
    """Highest-success cutoff of the new definition with wrong <= target (ties: lower wrong, higher cutoff)."""
    t = tab[(tab.variant == "local_tol_trim_l0x") & (tab.wrong <= target_wrong + 1e-12)]
    return t.sort_values(["success", "wrong", "cutoff"], ascending=[False, True, False]).iloc[0]


if __name__ == "__main__":
    tabs, chosen = [], {}
    for mode in ("fullyear", "assigned"):
        recs = load(mode)
        base = pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_b_clean_{mode}_v8_tables/per_star.csv",
                           index_col="star_id")
        mine = pd.Series({r["star_id"]: r["sweep"][("local_tol_trim", 1.0)][1] for r in recs})
        bad = (mine != base.final_outcome.reindex(mine.index))
        assert len(mine) == len(base) and not bad.any(), f"{mode}: harness differs from _v8 on {bad.sum()} stars"
        print(f"{mode}: harness reproduces _v8 per-star outcomes on all {len(mine)} stars")
        tab = sweep_table(recs, mode)
        tabs.append(tab)
        v8 = json.load(open(ROOT / f"benchmarks/baselines/dev_medium_b_clean_{mode}_v8.json"))["final"]["overall"]
        c = choose(tab, v8["wrong"])
        chosen[mode] = dict(target_wrong=v8["wrong"], v8_success=v8["success"], cutoff=float(c.cutoff),
                            success=float(c.success), wrong=float(c.wrong), alias=float(c.alias))
        print(mode, chosen[mode])
        rows = []
        for r in recs:
            for v, cl in r["cands"].items():
                for c_ in cl:
                    rows.append(dict(star_id=r["star_id"], true_period=r["true_period"], period_bin=r["period_bin"],
                                     variant=v, **c_))
        pd.DataFrame(rows).to_csv(DATA / f"cands_B_{mode}.csv.gz", index=False)
    pd.concat(tabs).to_csv(DATA / "sweep_B.csv", index=False)
    chosen["chosen"] = chosen["fullyear"]["cutoff"]
    chosen["modes_agree"] = chosen["fullyear"]["cutoff"] == chosen["assigned"]["cutoff"]
    json.dump(chosen, open(DATA / "chosen_cutoff.json", "w"), indent=1)
    print(json.dumps(chosen, indent=1))
