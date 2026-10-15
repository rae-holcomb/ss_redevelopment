#!/usr/bin/env python3
"""ablation.py -- T18 gate ablation by offline replay (arbitration.py) on the
diag_cache.py caches. Thresholds were chosen on the DEV caches (synth200
seed 0, pb_scan seed offsets 500 and 1500); the VALIDATION caches (synth200
seed 1, pb_scan evaluation seeds = offset 0) are only scored here, never used
for choosing.

Writes data/ablation.csv (one row per rule x dataset) and data/ablation_bins.csv
(synth success by period bin). Run from this folder, after
`python extra_feats.py <cache dirs>`.
"""
from pathlib import Path

import pandas as pd

import arbitration as A

RUNS = Path("../../runs")
DEV = {"synth_s0": "t18_cache_s0", "pb_dev": "t18_cache_pbdev", "pb_dev_noise2": "t18_cache_pbdev2"}
VAL = {"synth_s1": "t18_cache_s1", "pb_eval": "t18_cache_pbeval"}

HL = {"hlc_allpos": 1, "hlc_or": 0.1}           # half-lag consistency gate (as implemented)
FEW = {"few_acfh_min": 0.4}                      # few-peak ACF-height guard (as implemented)
REC = {**HL, **FEW}                              # recommended = new default
RULES = {
    "base (pre-T18)": {},
    "+ half-lag gate only": HL,
    "+ few-peak guard only": FEW,
    "RECOMMENDED: half-lag gate + few-peak guard": REC,
    "rec, half-lag 'all terms > 0' only (no 0.1 escape)": {"hlc_allpos": 1, **FEW},
    "rec, few-peak guard 0.3": {**HL, "few_acfh_min": 0.3},
    "rec, few-peak guard 0.5": {**HL, "few_acfh_min": 0.5},
    "rec, few-peak guard = PDM theta <= 0.7": {**HL, "few_pdm_max": 0.7},
    "rec, factor-2 guard off": {**REC, "guard": "none"},
    "rec, factor-2 ratio 0.6": {**REC, "guard_thr": 0.6},
    "rec, factor-2 ratio 0.8": {**REC, "guard_thr": 0.8},
    "rec, factor-2 by fold_half_corr_at_2P < 0.5": {**REC, "guard": "fold", "fold_thr": 0.5},
    "rec, select max contrast": {**REC, "select": "hlc"},
    "rec, select redchi / n_methods": {**REC, "select": "redchi_over_methods"},
    "rec, require >= 2 methods": {**REC, "min_methods": 2},
    "rec, vertex gate 0.5": {**REC, "vertex_min": 0.5},
    "rec, height-SNR gate 0.7": {**REC, "snr_min": 0.7},
    "rec, height-SNR gate 0.5": {**REC, "snr_min": 0.5},
    "raw contrast gate hlc >= 0.02": {"hlc_min": 0.02, **FEW},
    "contrast t-stat >= 3": {"hlc_t_min": 3, **FEW},
    "B: half-lag PREFERRED, else base gates": {**REC, "prefer": 1},
    "C: base gates, select max contrast": {"select": "hlc", **FEW},
    "D: half-lag preferred, else base gates + contrast > 0": {**REC, "prefer": 1, "fallback_hlc_min": 0.0},
}


def main():
    """Replay every rule on every cache; write the two CSVs and print a compact view."""
    rows, bins = [], []
    for split, sets in (("dev", DEV), ("val", VAL)):
        for name, d in sets.items():
            for rule_name, rule in RULES.items():
                df = A.replay(str(RUNS / d), rule)
                rows.append(dict(split=split, dataset=name, rule=rule_name, **A.summarize(df)))
                if name.startswith("synth"):
                    sig = df[df.kind != "noise"]
                    for b, g in sig.groupby("period_bin"):
                        bins.append(dict(split=split, dataset=name, rule=rule_name, bin=b, n=len(g),
                                         success=g.outcome.isin(["correct", "diffrot"]).mean()))
    out = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    out.to_csv("data/ablation.csv", index=False)
    pd.DataFrame(bins).to_csv("data/ablation_bins.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for ds in list(DEV) + list(VAL):
        print("\n==", ds)
        cols = ["rule", "success", "correct", "diffrot", "alias", "wrong", "none"] + \
               (["fp_white", "fp_red", "n_noise_each"] if ds.startswith("pb") else [])
        print(out[out.dataset == ds][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
