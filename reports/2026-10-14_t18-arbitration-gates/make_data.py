#!/usr/bin/env python3
"""make_data.py -- extract the small tables the notebook plots from the (gitignored)
runs/t18_cache_* diagnostic caches into data/. Run from this folder after
diag_cache.py, extra_feats.py and diagnose.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
RUNS = ROOT / "runs"
CACHES = {"synth_s0": "t18_cache_s0", "synth_s1": "t18_cache_s1", "pb_dev": "t18_cache_pbdev",
          "pb_dev_noise2": "t18_cache_pbdev2", "pb_eval": "t18_cache_pbeval"}
COLS = ["star_id", "P", "few_peaks", "passed", "label", "acf_half_lag_contrast", "hlc_min_term", "hlc_t",
        "acf_height_at_P", "mean_hsnr", "mean_h", "pdm_theta", "fold_half_corr", "n_methods"]


def main():
    """Write data/selected_base.csv, data/few_peak_cands.csv and data/example_params.json."""
    sel, few = [], []
    for name, d in CACHES.items():
        s = pd.read_csv(RUNS / d / "stars.csv")
        c = pd.read_csv(RUNS / d / "cands_x.csv")
        meta = s[["star_id", "kind", "true_period", "period_bin", "final_outcome"] +
                 [k for k in ("snr", "red", "n_sectors") if k in s.columns]]
        x = c[c.selected][COLS].merge(meta, on="star_id")
        sel.append(x.assign(dataset=name))
        f = c[c.few_peaks & c.passed][COLS].merge(meta, on="star_id")
        few.append(f.assign(dataset=name))
    pd.concat(sel).to_csv("data/selected_base.csv", index=False)
    pd.concat(few).to_csv("data/few_peak_cands.csv", index=False)

    # frozen parameters of the example stars (independent of later make_suite changes)
    from synthetic import make_suite
    suite = make_suite(180, seed=0)
    ex = {}
    for i in (64, 175, 144):
        p = {k: v for k, v in suite[i].params.items() if k not in ("spots", "period_range", "snr")}
        p["star_id"] = suite[i].star_id
        ex[suite[i].star_id] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in p.items()}
    for ns, seed in ((1, 901501), (2, 902501)):
        ex[f"pb-noise-red-{ns}s-{seed}"] = dict(period=1.0, amplitude=0.0, white_noise=0.001, n_sectors=ns, seed=seed,
                                                red_noise_sigma=0.001, red_noise_tau_days=2.0,
                                                star_id=f"pb-noise-red-{ns}s-{seed}")
    Path("data/example_params.json").write_text(json.dumps(ex, indent=1))


if __name__ == "__main__":
    main()
