"""Step-0 diagnosis: which windows does the MAD outlier rejection drop under local vs lsq, and why."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p43lib as P  # noqa: E402
import numpy as np  # noqa: E402

for sid, mode in [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]:
    st, pooled = P.star_and_guesses("smarts-tess-v1.0-" + sid, mode)
    res = P.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant="local_tol")
    r = [r for r in res.candidates if r.fit is not None and P.classify(r.fit.P, st) in ("correct", "diffrot")][0]
    c = st["counts"]
    w = np.sqrt(np.clip(c, 0, None) / max(c[0], 1.0))  # fit_rotation_period's pair-count weights
    print(f"#### {sid} {mode} P0={r.period:.3f} t0={r.t0:.3f} acf_std={np.std(st['acf']):.3f}")
    for name in ("local", "lsq"):
        for rej in (3, 0):
            fit = P.C._fit_single_candidate(st["lags"], st["acf"], r.period, r.t0, n_peaks=8, max_reject_iters=rej,
                                            min_peaks_required=2, min_keep_after_trim=r.n_peaks_required,
                                            weights=w, comb_spec=P.C.COMB_VARIANTS[name])
            d = P.C.assess_rotation_candidate(fit, st["acf"], min_peaks=r.n_peaks_required)
            s = " ".join(f"{n}:{pk['height']:+.2f}/{pk['residual_rms']:.3f}" for n, pk in sorted(fit.per_peak.items()))
            print(f"  {name:5s} rej={rej} P={fit.P:.3f} drop={fit.n_peaks_dropped} trim={fit.n_peaks_trimmed}"
                  f" snr={np.nanmean(d['height_snr']):.2f} | n:h/rms {s}")
