#!/usr/bin/env python3
"""Collect T12 measurement outputs (scratch run directories) into small CSVs under data/.

Usage: python reports/2026-11-12_t12-speed-up/collect_data.py SCRATCH_DIR

SCRATCH_DIR holds the runs made for this report:
    runs_before/{dsfy,dsas,syn,dmfy,dmas}   scorecards on the unchanged code (9063c95) = v6 baselines
    runs_after/{syn,dmfy,dmas,dsfy,dsas}    scorecards on the optimized code
    runs_before2/dsfy                       second before-code run (run-to-run noise check)
    runs_before_t, runs_after_t/{dmfy,dmas} back-to-back before/after dev_medium runs used for timing
    runs_s1, runs_s2                         dev_small after change 1, after changes 1-3 (step checks)
    ab/t_{before,after}_{fy,as}_r{0,1,2}.csv interleaved serial per-stage timings (benchmarks/profile_pipeline.py)
    ab/prof_{before,after}_{fy,as}.txt      cProfile text
    workers/w{2,4,6,8}_r{0,1}               dev_small full-year scorecards at several worker counts
    ft/{before,after}_{fy,as}               dev_small feature tables (tools/build_feature_tables.py)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"


def rows(run: Path) -> pd.DataFrame:
    """Per-star timing + descriptors from a scorecard run (rows.jsonl joined to per_star.csv)."""
    recs = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line.strip()]
    df = pd.DataFrame([dict(star_id=r["star_id"], seconds=r["seconds"], fit_seconds=r.get("fit_seconds"),
                            n_pooled=r.get("n_pooled"),
                            **{f"m_{k}": v.get("seconds") for k, v in r.get("methods", {}).items()})
                       for r in recs])
    ps = pd.read_csv(run / "per_star.csv")[["star_id", "true_period", "period_bin", "coverage_class",
                                            "n_sectors", "final_outcome"]]
    return df.merge(ps, on="star_id", how="left")


def main(scratch: Path):
    """Write every data/ table used by the notebook."""
    DATA.mkdir(exist_ok=True)
    # 1. per-stage serial timings
    parts = []
    for f in sorted((scratch / "ab").glob("t_*_r*.csv")):
        tag, mode, rep = f.stem.split("_")[1:]
        d = pd.read_csv(f)
        d["code"], d["mode"], d["rep"] = tag, {"fy": "fullyear", "as": "assigned"}[mode], int(rep[1:])
        parts.append(d)
    pd.concat(parts).to_csv(DATA / "stage_times.csv", index=False)
    for f in (scratch / "ab").glob("prof_*.txt"):
        shutil.copy(f, DATA / f.name)
    # 2. per-star runtime, before vs after, dev_medium 4-worker scorecards
    parts = []
    for mode, key in (("fullyear", "dmfy"), ("assigned", "dmas")):
        # timing pair run back-to-back on the idle machine (03:04-03:35); the first before runs (01:25-02:12)
        # ran while the machine was ~1.9x slower (see README), so they are not used for timing
        b, a = rows(scratch / "runs_before_t" / key), rows(scratch / "runs_after_t" / key)
        m = b.merge(a[["star_id", "seconds", "fit_seconds"] + [c for c in a if c.startswith("m_")]],
                    on="star_id", suffixes=("_before", "_after"))
        m["mode"] = mode
        parts.append(m)
    pd.concat(parts).to_csv(DATA / "per_star_runtime.csv", index=False)
    # 3. run wall times
    wall = []
    for tag in ("runs_before", "runs_after", "runs_before2", "runs_before_t", "runs_after_t", "runs_s1", "runs_s2"):
        for run in sorted((scratch / tag).glob("*/metrics.json")):
            c = json.loads(run.read_text())["config"]
            wall.append(dict(code=tag, run=run.parent.name, dataset=c["dataset"], wall_seconds=c["wall_seconds"],
                             git_commit=c["git_commit"]))
    pd.DataFrame(wall).to_csv(DATA / "run_wall_seconds.csv", index=False)
    # 4. worker scaling
    wk = []
    for run in sorted((scratch / "workers").glob("w*_r*/metrics.json")):
        w, r = run.parent.name.split("_")
        c = json.loads(run.read_text())["config"]
        n = sum(1 for line in (run.parent / "rows.jsonl").read_text().splitlines() if line.strip())
        wk.append(dict(n_workers=int(w[1:]), rep=int(r[1:]), wall_seconds=c["wall_seconds"], n_stars=n))
    pd.DataFrame(wk).to_csv(DATA / "worker_scaling.csv", index=False)
    # 5. bit-identity checks
    cmp_tool = ROOT / "tools" / "compare_runs_exact.py"
    checks = [("dev_small full-year, after change 1", "runs_before/dsfy", "runs_s1/dsfy"),
              ("dev_small assigned, after change 1", "runs_before/dsas", "runs_s1/dsas"),
              ("dev_small full-year, after changes 1-3", "runs_before/dsfy", "runs_s2/dsfy"),
              ("dev_small assigned, after changes 1-3", "runs_before/dsas", "runs_s2/dsas"),
              ("dev_small full-year, final", "runs_before/dsfy", "runs_after/dsfy"),
              ("dev_small assigned, final", "runs_before/dsas", "runs_after/dsas"),
              ("synth200, final", "runs_before/syn", "runs_after/syn"),
              ("dev_medium full-year, final", "runs_before/dmfy", "runs_after/dmfy"),
              ("dev_medium assigned, final", "runs_before/dmas", "runs_after/dmas"),
              ("dev_medium full-year, final (timing pair)", "runs_before/dmfy", "runs_after_t/dmfy"),
              ("dev_medium assigned, final (timing pair)", "runs_before/dmas", "runs_after_t/dmas"),
              ("dev_medium full-year, before vs before rerun", "runs_before/dmfy", "runs_before_t/dmfy"),
              ("dev_medium assigned, before vs before rerun", "runs_before/dmas", "runs_before_t/dmas"),
              ("dev_small full-year, before vs before rerun", "runs_before/dsfy", "runs_before2/dsfy"),
              ("feature tables dev_small full-year, final", "ft/before_fy", "ft/after_fy"),
              ("feature tables dev_small assigned, final", "ft/before_as", "ft/after_as")]
    out = []
    for label, a, b in checks:
        if not (scratch / a).exists() or not (scratch / b).exists():
            continue
        js = DATA / f"compare_{Path(b).parent.name}_{Path(b).name}.json"
        subprocess.run([sys.executable, str(cmp_tool), str(scratch / a), str(scratch / b), "--json", str(js)],
                       capture_output=True)
        s = json.loads(js.read_text())
        out.append(dict(check=label, n_stars=s["n_stars"], values_compared=s["values_compared"],
                        n_differ=s["n_diffs"], runtime_fields_skipped=s["runtime_keys_skipped"],
                        diffs="; ".join(s["diffs"][:10])))
    pd.DataFrame(out).to_csv(DATA / "identity_checks.csv", index=False)
    print(pd.DataFrame(out)[["check", "values_compared", "n_differ"]].to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
