"""T32 step 1: per-peak vertex residuals for local's gains, local's none->wrong stars, and shared-correct stars.

For each star, the pooled candidates are fit with "local" and with "lsq"; the
candidate of interest is local's selected answer (seed P0), and the lsq fit of
the same seed is taken from the lsq run. Writes data/step1_peaks.csv.

Usage: nice -n 15 python reports/2026-11-13_t32-local-drift/step1_peaks.py [n_shared_per_mode]
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t32lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T31 = L.ROOT / "reports/2026-11-13_t31-comb-fit-speed/data/changed_stars_v_local__dec.csv"
T31RUNS = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/t31-comb-fit-speed-e7e5bb/runs/t31/runs")


def groups(n_shared: int = 30, seed: int = 0) -> pd.DataFrame:
    """(star_id, mode, group) rows: gain_vertex, none_to_wrong, shared_correct."""
    d = pd.read_csv(T31)
    a = d[(d.change == "gained") & d.decline_gates_main.fillna("").str.contains("frac_vertex")]
    b = d[(d.final_outcome_main == "none") & (d.final_outcome_var == "wrong")]
    rows = [dict(star_id=r.star_id, mode=r["mode"], group="gain_vertex") for _, r in a.iterrows()]
    rows += [dict(star_id=r.star_id, mode=r["mode"], group="none_to_wrong") for _, r in b.iterrows()]
    rng = np.random.default_rng(seed)
    for mode, tag in (("fullyear", "dmfy"), ("assigned", "dmas")):
        m = pd.read_csv(T31RUNS / f"A_{tag}/per_star.csv")
        v = pd.read_csv(T31RUNS / f"v_local__dec_{tag}/per_star.csv")
        j = m.merge(v, on="star_id", suffixes=("_m", "_v"))
        ok = j[(j.final_outcome_m == "correct") & (j.final_outcome_v == "correct")].star_id.tolist()
        for s in rng.choice(ok, n_shared, replace=False):
            rows.append(dict(star_id=s, mode=mode, group="shared_correct"))
    return pd.DataFrame(rows)


def one(args):
    """Per-peak rows (local and lsq fits of local's selected seed) for one (task, group, mode)."""
    task, group, mode = args
    warnings.simplefilter("ignore")
    st = L.load_star(task)
    pooled = L.pooled_guesses(st)
    rl = L.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant="local")
    if not rl.success:
        return []
    sel = rl.candidates[0]
    rq = L.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant="lsq")
    rows = L.peak_records(st, sel, "local")
    lsq_c = [c for c in rq.candidates if abs(c.period - sel.period) < 1e-12]
    if lsq_c:
        rows += L.peak_records(st, lsq_c[0], "lsq")
    oc = L.ev.classify_period(sel.fit.P, st["true_p"], L.ev.REL_TOL, st["prange"])
    for r in rows:
        r.update(star_id=st["star_id"], mode=mode, group=group, true_period=st["true_p"], outcome_local=oc,
                 lsq_failed=";".join(lsq_c[0].failed_gates) if lsq_c else "")
    return rows


def main():
    """Build the groups, run them at 3 workers, write data/step1_peaks.csv."""
    n_shared = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    g = groups(n_shared)
    jobs = []
    for mode in ("fullyear", "assigned"):
        by_id = {t["star_id"]: t for t in L.make_tasks(mode)}
        for _, r in g[g["mode"] == mode].iterrows():
            jobs.append((by_id[r.star_id], r.group, mode))
    with ProcessPoolExecutor(3, initializer=L.determinism.init_worker) as ex:
        out = [row for rows in ex.map(one, jobs) for row in rows]
    (HERE / "data").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(HERE / "data/step1_peaks.csv", index=False)
    print(len(out), "rows")


if __name__ == "__main__":
    main()
