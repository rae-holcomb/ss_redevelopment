"""Where every dev rotator ends up at the v1 pick (per-bin p = 0.86): one category per star and table.

Feeds P65-P68 in docs/TASKS_PROPOSED.md. Reads the run's dev_top1_*.csv, the dev tables' stars.csv.gz
(noise_tic, noise class) and the drive's T20 noise_source_flags (P_ref; optional, skipped if the drive is
not mounted). Writes data/loss_categories.csv and prints the summary tables.

    python reports/2026-11-21_ranker-v1/loss_categories.py --run <main>/runs/ranker_v1/nohalf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_RUNS = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/runs"
FLAGS = "/Volumes/Data_Vault/smarts_data/extracted/galaxy_ids/noise_source_flags.csv.gz"
TABLES = {"dev_medium": "p31_l0x_dev_medium", "dev_medium_b": "p31_l0x_dev_medium_b"}
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
SUC, ALI = {"correct", "diffrot"}, {"half", "double"}


def same_period(p, q, tol=0.05):
    """p within tol of q, q/2 or 2q (relative)."""
    r = np.asarray(p, float) / np.asarray(q, float)
    return np.isfinite(r) & ((abs(r - 1) < tol) | (abs(r - 0.5) < tol / 2) | (abs(r - 2) < 2 * tol))


def categorize(d: pd.DataFrame, thr: dict) -> pd.Series:
    """One loss category per star from its top-1 row and the per-bin thresholds."""
    ans = d.score_v1 >= d.top_bin.map(thr).astype(float)
    ok, al = d.top_outcome.isin(SUC), d.top_outcome.isin(ALI)
    return pd.Series(np.select(
        [ans & ok, ans & al, ans & ~ok & ~al, ~ans & ok,
         ~ans & ~ok & (d.max_relevance == 2), ~ans & ~ok & (d.max_relevance == 1),
         ~ans & (d.max_relevance == 0) & (d.searchable == 1)],
        ["answered_success", "answered_alias", "answered_wrong", "declined_top1_right",
         "declined_correct_cand_not_top1", "declined_alias_cand_only", "no_correct_cand_searchable"],
        "no_correct_cand_unsearchable"), index=d.index)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    thr = json.load(open(os.path.join(args.run, "thresholds.json")))["rules"]["head_perbin_p0.86"]
    flags = (pd.read_csv(FLAGS, usecols=["tic", "P_ref"]).drop_duplicates("tic").set_index("tic").P_ref
             if os.path.exists(FLAGS) else None)
    parts = []
    for s, tab in TABLES.items():
        for m in ("fullyear", "assigned"):
            d = pd.read_csv(os.path.join(args.run, f"dev_top1_{s}_{m}.csv"))
            d = d[d.is_rotator].copy()
            st = pd.read_csv(os.path.join(MAIN_RUNS, f"{tab}_{m}", "stars.csv.gz"),
                             usecols=["star_id", "noise_tic", "noise_classification"])
            d = d.merge(st, on="star_id")
            d["category"] = categorize(d, thr)
            d["set"], d["mode"] = s, m
            parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    if flags is not None:
        o["noise_P_ref"] = o.noise_tic.map(flags)
        o["top1_is_noise_period"] = same_period(o.top_period, o.noise_P_ref)
    cols = ["set", "mode", "star_id", "period_bin", "true_period", "category", "top_period", "top_outcome",
            "top_method", "score_v1", "max_relevance", "searchable", "band_width_class", "noise_tic",
            "noise_classification"] + (["noise_P_ref", "top1_is_noise_period"] if flags is not None else [])
    o[cols].to_csv(os.path.join(HERE, "data", "loss_categories.csv"), index=False)
    pd.set_option("display.width", 200)
    print(pd.crosstab([o.set, o["mode"]], o.category, margins=True))
    print(pd.crosstab([o.category, o["mode"]], o.period_bin).reindex(columns=BINS, fill_value=0))
    if flags is not None:
        print(o.groupby("category").top1_is_noise_period.mean().round(3))


if __name__ == "__main__":
    main()
