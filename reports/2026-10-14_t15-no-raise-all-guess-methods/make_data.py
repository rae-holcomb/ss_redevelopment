"""Stage-1 wavelet data for the T15 report.

Runs guess_wavelet (the only method with a nonzero baseline error rate) on
synthetic:n=200,seed=0 using the guesses.py found in --code-dir, exactly as
benchmarks/scorecard.py calls it (gap-filled flux, n_guesses=10), and
records per star whether it raised, the error text, and the candidate
count. For the stars that raised it also saves the GWPS and the first
iteration's min_peak_snr threshold, so the notebook can show why.

Usage (from the repo root):
    python <report>/make_data.py --code-dir <dir> --out <report>/data/wavelet_<tag>.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def gwps_and_threshold(guesses, t, f, min_peak_snr=3.0):
    """Return (periods, gwps, threshold) using guess_wavelet's own grid and
    the threshold rule of its first peak-extraction iteration."""
    dt = np.median(np.diff(t))
    periods = np.geomspace(4 * dt, (t[-1] - t[0]) / 2, 200)
    scales = guesses._period_to_morlet_scale(periods, dt, 6.0)
    power = np.abs(guesses._cwt_morlet(f - np.mean(f), scales, 6.0)) ** 2
    gwps = power.mean(axis=1)
    med = np.median(gwps)
    mad = np.median(np.abs(gwps - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(gwps)
    return periods, gwps, med + min_peak_snr * sigma


def main():
    """Run guess_wavelet over the synthetic suite and save a per-star CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.code_dir) / "claude_files"))
    import guesses
    from acf_utils import compute_acf
    from synthetic import make_suite
    warnings.simplefilter("ignore")

    rows, spectra = [], {}
    for lc in make_suite(200, seed=0):
        t, f = lc.time, lc.flux
        lags, acf = compute_acf(t, f)
        idx = np.arange(len(f))
        ok = np.isfinite(f)
        fg = np.interp(idx, idx[ok], f[ok])
        row = dict(star_id=lc.star_id, true_period=lc.true_period,
                   snr=lc.params.get("snr"), inclination=lc.params.get("inclination_deg"))
        try:
            out = guesses.guess_wavelet(t, fg, lags, acf, n_guesses=10)
            row.update(error=None, n_candidates=len(out))
        except Exception as exc:  # noqa: BLE001
            row.update(error=f"{type(exc).__name__}: {exc}", n_candidates=0)
            per, g, thr = gwps_and_threshold(guesses, t, fg)
            spectra[lc.star_id] = dict(periods=per.tolist(), gwps=g.tolist(), threshold=float(thr),
                                       true_period=lc.true_period)
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if spectra:
        Path(args.out).with_suffix(".spectra.json").write_text(json.dumps(spectra))
    print(f"wrote {args.out}: {sum(r['error'] is not None for r in rows)} errors")


if __name__ == "__main__":
    main()
