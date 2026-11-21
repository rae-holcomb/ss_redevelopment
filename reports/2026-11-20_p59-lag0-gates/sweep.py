"""P59: threshold sweep on dev_medium_b_clean (tuning set), both modes; the selection rule.

Reads runs/p59/{coarse,fine}_B_{mode}.pkl (p59lib.py). Checks that local_tol_trim_l0x at (4, 0.8, 1.02)
reproduces the dev_medium_b_clean _v9 per-star outcomes, tabulates success / wrong per combination, and
applies the rule: highest success with wrong <= B's _v9 full-year wrong; ties -> lower wrong, then the
combination closest to the current thresholds. Full-year decides.

Writes data/sweep_B.csv and data/chosen.json; `python sweep.py fine` also proposes nothing (refinement grid
is chosen in the report from the coarse result).
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
S, A = {"correct", "diffrot"}, {"half", "double"}
NEW, OLD = "local_tol_trim_l0x2", "local_tol_trim_l0x"
RECORD = (4, 0.8, 1.02)


def table(recs, mode, stage) -> pd.DataFrame:
    """success / wrong / alias / none per (variant, mp, fp, snr)."""
    N = len(recs)
    rows = []
    for key in sorted(recs[0]["sweep"]):
        out = [r["sweep"][key] for r in recs]
        s = sum(o in S for o in out)
        w = sum(o == "wrong" for o in out)
        rows.append(dict(mode=mode, stage=stage, variant=key[0], mp=key[1], fp=key[2], snr=key[3], n=N,
                         n_success=s, n_wrong=w, n_alias=sum(o in A for o in out),
                         success=s / N, wrong=w / N, alias=sum(o in A for o in out) / N,
                         none=sum(o == "none" for o in out) / N))
    return pd.DataFrame(rows)


def choose(tab, target, variant=NEW) -> pd.Series:
    """Rule's pick among `variant` rows with wrong <= target."""
    t = tab[(tab.variant == variant) & (tab.wrong <= target + 1e-12)].copy()
    t["dist"] = (t.mp - RECORD[0]).abs() + (t.fp - RECORD[1]).abs() * 5 + (t.snr - RECORD[2]).abs() * 10
    return t.sort_values(["success", "wrong", "dist"], ascending=[False, True, True]).iloc[0]


if __name__ == "__main__":
    tabs = []
    for mode in ("fullyear", "assigned"):
        base = pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_b_clean_{mode}_v9_tables/per_star.csv",
                           index_col="star_id")
        for stage in ("coarse", "fine"):
            pk = ROOT / f"runs/p59/{stage}_B_{mode}.pkl"
            if not pk.exists():
                continue
            recs = pickle.load(open(pk, "rb"))
            if (OLD,) + RECORD in recs[0]["sweep"]:
                mine = pd.Series({r["star_id"]: r["sweep"][(OLD,) + RECORD] for r in recs})
                bad = mine != base.final_outcome.reindex(mine.index)
                assert len(mine) == len(base) and not bad.any(), f"{stage} {mode}: differs from _v9 on {bad.sum()}"
                print(f"{stage} {mode}: harness reproduces _v9 on all {len(mine)} stars")
            tabs.append(table(recs, mode, stage))
            if stage == "coarse":
                rows = [dict(star_id=r["star_id"], true_period=r["true_period"], period_bin=r["period_bin"],
                             variant=v, **c) for r in recs for v, cl in r["cands"].items() for c in cl]
                pd.DataFrame(rows).to_csv(DATA / f"cands_B_{mode}.csv.gz", index=False)
    tab = pd.concat(tabs).drop_duplicates(["mode", "variant", "mp", "fp", "snr"], keep="last")
    tab.to_csv(DATA / "sweep_B.csv", index=False)
    chosen = {}
    for mode in ("fullyear", "assigned"):
        v9 = json.load(open(ROOT / f"benchmarks/baselines/dev_medium_b_clean_{mode}_v9.json"))["final"]["overall"]
        t = tab[tab["mode"] == mode]
        c = choose(t, v9["wrong"])
        o = choose(t, v9["wrong"], OLD)
        chosen[mode] = dict(target_wrong=v9["wrong"], v9_success=v9["success"],
                            mp=int(c.mp), fp=float(c.fp), snr=float(c.snr), success=float(c.success),
                            wrong=float(c.wrong),
                            old_best=dict(mp=int(o.mp), fp=float(o.fp), snr=float(o.snr),
                                          success=float(o.success), wrong=float(o.wrong)),
                            n_combos={v: int((t.variant == v).sum()) for v in (OLD, NEW)})
    chosen["chosen"] = {k: chosen["fullyear"][k] for k in ("mp", "fp", "snr")}
    chosen["modes_agree"] = all(chosen["fullyear"][k] == chosen["assigned"][k] for k in ("mp", "fp", "snr"))
    json.dump(chosen, open(DATA / "chosen.json", "w"), indent=1)
    print(json.dumps(chosen, indent=1))
