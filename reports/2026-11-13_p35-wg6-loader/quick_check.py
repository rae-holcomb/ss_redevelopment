"""Quick check (<=20 stars, 2 workers): run the standard scorecard per-star path on a handful of WG6 dev stars.

Run from the repo root under nice:  nice -n 15 python reports/2026-11-13_p35-wg6-loader/quick_check.py
Uses scorecard.build_wg6_tasks (the same tasks `--manifest wg6r_*_v0_improved_s1` builds) and scorecard.score_one_star,
so it exercises the real dataset-string path on 10 FL rotators (distinct templates, WFmag 17) + 10 contaminants.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402

determinism.pin_blas_threads()
sys.path.insert(0, str(ROOT / "benchmarks"))
import os  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402
import scorecard as sc  # noqa: E402
from wg6 import resampled_root  # noqa: E402

def main():
    if not resampled_root().is_dir():
        sys.exit("plug in the Data_Vault drive")
    fl = [t for t in sc.build_wg6_tasks("wg6r_fl_v0_improved_s1") if t["meta"]["wfmag"] == 17]
    seen, fl10 = set(), []
    for t in fl:
        if t["meta"]["fl_template"] not in seen:
            seen.add(t["meta"]["fl_template"]); fl10.append(t)
    co = sc.build_wg6_tasks("wg6r_contam_v0_improved_s1")
    co10, per = [], {}
    for t in co:
        c = t["meta"]["cls"]
        if per.get(c, 0) < 2 and c in ("ECL", "ELL", "T2CEP", "RRLYR", "DSCT"):
            per[c] = per.get(c, 0) + 1; co10.append(t)
    tasks = fl10[:10] + co10[:10]
    worker = partial(sc.score_one_star, pipeline_methods=sc.DEFAULT_PIPELINE_METHODS, rel_tol=sc.ev.REL_TOL)
    with ProcessPoolExecutor(2, initializer=determinism.init_worker) as ex:
        recs = list(ex.map(worker, tasks))
    rows = [dict(star_id=r["star_id"], cls=t["meta"]["cls"], is_eb=t["meta"]["is_eb"], P_true=t["meta"]["P_true"],
                 P_score=t["true_period"], P_found=r.get("final_period"), outcome=r["final_outcome"], seconds=r["seconds"])
            for t, r in zip(tasks, recs)]
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).parent / "data/quick_check.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
