"""Per-window residual RMS of one candidate under lsq vs local, before any rejection (mechanism probe)."""
import pickle, sys, warnings
import numpy as np
sys.path.insert(0, 'reports/2026-11-16_p43-clean-rerun'); import p43lib as L
C = L.C
warnings.simplefilter('ignore')
mode, ids = sys.argv[1], sys.argv[2].split(',')
tasks = {t['star_id']: t for t in L.make_tasks(mode)}
recs = {r['star_id']: r for r in pickle.load(open(f'runs/p43_clean/harness/changed_{mode}.pkl', 'rb'))}
for sid in ids:
    sid = [k for k in recs if k.endswith(sid)][0]
    st = L.load_star(tasks[sid]); w = np.sqrt(np.clip(st['counts'], 0, None) / max(st['counts'][0], 1))
    c = [c for c in recs[sid]['variants']['lsq']['candidates'] if c['passed'] and c['outcome'] in ('correct', 'diffrot')][0]
    print(f"\n{sid[-6:]} P0={c['P0']:.3f} t0={c['t0']:.3f}")
    for v in sys.argv[3].split(",") if len(sys.argv) > 3 else ("lsq", "local"):
        f = C._fit_single_candidate(st['lags'], st['acf'], c['P0'], c['t0'], weights=w, max_reject_iters=0,
                                    trim_decayed_tail=False, min_peaks_required=2, comb_spec=C.COMB_VARIANTS[v])
        rms = np.array([f.per_peak[n]['residual_rms'] for n in sorted(f.per_peak)])
        med = np.median(rms); s = 1.4826 * np.median(np.abs(rms - med))
        print(f"  {v:6s} P={f.P:.3f} cut={med+4*s:.4f} " + ' '.join(f"n{n}:h{f.per_peak[n]['height']:+.2f}/r{f.per_peak[n]['residual_rms']:.3f}" for n in sorted(f.per_peak)))
