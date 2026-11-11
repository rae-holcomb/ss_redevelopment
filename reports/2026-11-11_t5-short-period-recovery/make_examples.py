"""T5 phase 1: save light curve, ACF and spectra (+ old/new top-10 guesses) for the diagnosis example stars.

Usage (local dev files, ~30 s):
    python reports/2026-11-11_t5-short-period-recovery/make_examples.py
Writes data/examples.npz; the notebook draws fig_diag_*.png from it.
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "claude_files")]
import determinism  # noqa: E402

determinism.pin_blas_threads()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import guesses as G  # noqa: E402
from acf_utils import STAGE1_MAX_LAG_FRAC, compute_acf  # noqa: E402
from preprocessing import load_smarts_fits  # noqa: E402

EXAMPLES = ["smarts-tess-v1.0-002390", "smarts-tess-v1.0-001208", "smarts-tess-v1.0-000718",
            "smarts-tess-v1.0-002919"]


def main() -> None:
    """Compute and save the per-star, per-mode arrays for the diagnosis figures."""
    warnings.simplefilter("ignore")
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv").set_index("star_id")
    out = {}
    for sid in EXAMPLES:
        r = man.loc[sid]
        prange = ev.active_band_period_range(r.true_period, r.shear, r.minlat, r.maxlat)
        out[f"{sid}|meta"] = np.array([r.true_period, prange[0], prange[1]])
        for mode in ("asg", "fy"):
            kw = {} if mode == "fy" else {"sectors": [int(s) for s in str(r.sectors).split("-")]}
            pre = load_smarts_fits(ROOT / "data/smarts_dev" / r.path, **kw)
            t, f = pre.time, pre.flux
            lags, acf = compute_acf(t, f, max_lag_frac=STAGE1_MAX_LAG_FRAC)
            k = f"{sid}|{mode}"
            out[k + "|t"], out[k + "|f"] = t.astype(np.float32), f.astype(np.float32)
            out[k + "|lags"], out[k + "|acf"] = lags.astype(np.float32), acf.astype(np.float32)
            for m, fn in (("acf_fft", G.guess_acf_fft), ("lombscargle", G.guess_lombscargle)):
                old = fn(t, f, lags, acf, n_guesses=10, rank_by="power")
                new = fn(t, f, lags, acf, n_guesses=10, rank_by="hybrid")
                if m == "acf_fft":
                    x, y = 1 / old[0].info["fft_freq"], old[0].info["fft_power"]
                else:
                    x, y = old[0].info["periodogram"]
                out[k + f"|{m}|P"], out[k + f"|{m}|pow"] = x.astype(np.float32), y.astype(np.float32)
                out[k + f"|{m}|old"] = np.array([g.P0 for g in old])
                out[k + f"|{m}|new"] = np.array([g.P0 for g in new])
    np.savez_compressed(Path(__file__).resolve().parent / "data/examples.npz", **out)
    print("saved", len(out), "arrays")


if __name__ == "__main__":
    main()
