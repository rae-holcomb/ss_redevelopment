#!/usr/bin/env python3
"""
make_data.py -- collect the T16b instrumented runs into two compact tables
under data/ that the notebook (and gate_lib's offline replays) read:

    data/stars.csv.gz       one row per (dataset, star): truth, coverage
                            descriptors, final outcome, decline gates, best
                            success candidate's failed gates
    data/methods.csv.gz     one row per (dataset, star, Stage 1 method): outcome
                            of its best-listed candidate (scorecard per_method.csv)
    data/candidates.csv.gz  one row per (dataset, star, fitted candidate):
                            fitted period, outcome, redchi, every gate's
                            value / threshold / pass / enabled

Datasets: synthetic n=200 seed 0 under coverage full / consecutive-2 /
gapped-2 / single (coverage seed 0), dev_medium with assigned coverage, and
the pb_scan noise set (noise-1s, noise-2s, noise-13s). Held-out validation:
synthetic seed 1 under single / gapped-2 (s1-*) and pb_scan noise with seed
offset 1500 (v-noise-*).

Usage
-----
    python reports/2026-10-21_t16b-gate-diagnosis/make_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_lib as gl  # noqa: E402

STAR_COLS = ["dataset", "star_id", "true_period", "period_lo", "period_hi", "period_bin", "coverage_class",
             "n_sectors", "sectors", "total_baseline_days", "longest_contiguous_days", "largest_gap_days",
             "duty_cycle", "acf_lag_max", "final_period", "final_outcome", "decline_kind", "decline_gates",
             "n_fit_candidates", "n_success_candidates", "best_success_P", "best_success_failed_gates",
             "meta_amplitude", "meta_white_noise", "meta_red_noise_sigma", "meta_shear"]
CAND_COLS = (["dataset", "star_id", "cand_rank", "P0", "P_fit", "method", "contributing_methods", "redchi",
              "outcome", "n_teeth_available", "n_peaks_required", "few_peaks", "passed_base_gates",
              "passed_gates", "selected", "failed_gates", "acf_pair_frac_at_P", "half_lag_contrast"]
             + [f"{g}_{k}" for g in gl.GATES for k in ("value", "threshold", "pass", "enabled")])


def main():
    """Build data/stars.csv.gz and data/candidates.csv.gz from runs/."""
    stars, cands, meths = [], [], []
    runs = dict(gl.SYNTH) | {"dev_medium": gl.DEV_MEDIUM} | gl.VALID
    for label, name in runs.items():
        ps, pc = gl.load_run(name)
        pm = pd.read_csv(gl.RUNS / name / "per_method.csv", usecols=["star_id", "period_bin", "method", "outcome"])
        meths.append(pm.assign(dataset=label))
        ps["dataset"], pc["dataset"] = label, label
        stars.append(ps.reindex(columns=STAR_COLS))
        cands.append(pc[pc.P_fit.notna()].reindex(columns=CAND_COLS))
    nz = pd.concat([pd.read_csv(gl.RUNS / d / "noise_candidates.csv", float_precision="round_trip").assign(pre=pre)
                    for pre, d in (("noise-", gl.NOISE), ("v-noise-", gl.NOISE_VALID))], ignore_index=True)
    nz["dataset"] = nz.pre + nz.n_sectors.astype(str) + "s"
    nz["star_id"] = nz.dataset + "-" + nz.seed.astype(str) + nz.red.map({True: "-red", False: "-white"})
    ns = nz.groupby(["dataset", "star_id"]).agg(n_sectors=("n_sectors", "first"), acf_lag_max=("acf_lag_max", "first"),
                                                pipeline_success=("pipeline_success", "first"),
                                                decline_gates=("decline_gates", "first")).reset_index()
    ns["final_outcome"] = ns.pipeline_success.map({True: "noise", False: "none"})
    ns["coverage_class"] = ns.dataset.str.extract(r"(\d+)s$")[0].map({"1": "single", "2": "consecutive-2", "13": "full"})
    stars.append(ns.reindex(columns=STAR_COLS))
    cands.append(nz[nz.P_fit.notna()].reindex(columns=CAND_COLS))
    (HERE / "data").mkdir(exist_ok=True)
    s, c = pd.concat(stars, ignore_index=True), pd.concat(cands, ignore_index=True)
    s.to_csv(HERE / "data/stars.csv.gz", index=False)
    c.to_csv(HERE / "data/candidates.csv.gz", index=False)
    pd.concat(meths, ignore_index=True).to_csv(HERE / "data/methods.csv.gz", index=False)
    print(s.groupby("dataset").size(), len(c))


if __name__ == "__main__":
    main()
