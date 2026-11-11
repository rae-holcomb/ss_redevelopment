"""T5 phase 2: flatten tools/t5_pool_compare.py shards into per-star, per-config tables.

Usage:
    python reports/2026-11-11_t5-short-period-recovery/phase2_tables.py runs/t5_pool
Reads <dir>/{rot_fy,rot_asg,neg}/*.pkl and writes data/phase2_{rot,neg}.csv.gz with, per star and config:
final period and outcome, whether any fitted candidate (after the short-period cap and dedup) is a
success, the number of fitted candidates and of wrong fitted candidates, the number of fitted
candidates that the added method contributed to, fit seconds and the added generator's seconds.
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

OUT = Path(__file__).resolve().parent / "data"
ADDED = {"base+highpass": ("acf_fft_highpass", "acf_fft_hp"), "base+acf_short": ("acf_fft_short", "acf_fft_short"),
         "base+ls_short": ("lombscargle_short", "lombscargle_short")}
SUC = set(ev.SUCCESS_OUTCOMES)


def rows_for(rec: dict, truth=None) -> list:
    """Per-config rows for one shard; truth = (P, (P_lo, P_hi)) for rotators, None for non-rotators."""
    out = []
    for cname, c in rec["configs"].items():
        added, tag = ADDED.get(cname, (None, None))
        fitted = [x for x in c["fitted"] if x["P"] is not None]
        if truth is None:
            oc = ["wrong"] * len(fitted)
            final = "declined" if c["final_period"] is None else "answered"
        else:
            P, pr = truth
            oc = [ev.classify_period(x["P"], P, ev.REL_TOL, pr) for x in fitted]
            final = ev.classify_period(c["final_period"], P, ev.REL_TOL, pr)
        by_added = [x for x in fitted if tag and tag in x["methods"]]
        out.append(dict(
            star_id=rec["star_id"], config=cname, final_period=c["final_period"], final_outcome=final,
            fitted_success=any(o in SUC for o in oc), n_fitted=len(fitted),
            n_fitted_wrong=sum(o not in SUC for o in oc),
            n_fitted_from_added=len(by_added),
            n_fitted_wrong_from_added=sum(o not in SUC for x, o in zip(fitted, oc) if tag and tag in x["methods"]),
            n_pooled=c["n_pooled"], n_tried=c["n_tried"], fit_seconds=c["fit_seconds"],
            added_gen_seconds=rec["gen_seconds"].get(added, 0.0) if added else 0.0,
            base_gen_seconds=sum(rec["gen_seconds"].get(m, 0.0) for m in ("pairwise_histogram", "lombscargle", "acf_fft")),
            load_seconds=rec["load_seconds"]))
    if truth is not None:  # raw (pre-cap) Stage-1 hit per generator, same for every config row
        P, pr = truth
        raw = {k: any(ev.classify_period(p, P, ev.REL_TOL, pr) in SUC for p, _ in v) for k, v in rec["raw"].items()}
        for row in out:
            row.update({f"raw_{k}": v for k, v in raw.items()})
    return out


def main(run_dir: str) -> None:
    """Write data/phase2_rot.csv and data/phase2_neg.csv."""
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv").set_index("star_id")
    rot = []
    for mode in ("fy", "asg"):
        for fn in sorted(glob.glob(f"{run_dir}/rot_{mode}/*.pkl")):
            rec = pickle.load(open(fn, "rb"))
            if rec["error"]:
                continue
            r = man.loc[rec["star_id"]]
            pr = ev.active_band_period_range(r.true_period, r.shear, r.minlat, r.maxlat)
            for row in rows_for(rec, (r.true_period, pr)):
                rot.append(dict(row, mode=mode, eval_pbin=r.eval_pbin))
    if rot:
        pd.DataFrame(rot).to_csv(OUT / "phase2_rot.csv.gz", index=False)
    neg_man = pd.read_csv(ROOT / "datasets/negatives_v0.csv").set_index("star_id")
    neg = []
    for fn in sorted(glob.glob(f"{run_dir}/neg/*.pkl")):
        rec = pickle.load(open(fn, "rb"))
        if rec["error"]:
            continue
        m = neg_man.loc[rec["star_id"]]
        for row in rows_for(rec):
            neg.append(dict(row, coverage_class=m.coverage_class, source_class=m.source_class,
                            full=(rec["star_id"].endswith("-full"))))
    if neg:
        pd.DataFrame(neg).to_csv(OUT / "phase2_neg.csv.gz", index=False)
    print(f"rotator rows {len(rot)}, negative rows {len(neg)}")


if __name__ == "__main__":
    main(sys.argv[1])
