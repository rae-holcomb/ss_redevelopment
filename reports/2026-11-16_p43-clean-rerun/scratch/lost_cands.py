"""For each lost star: the success candidates under local_tol vs lsq, failed gates and per-tooth heights."""
import pickle, sys
import numpy as np, pandas as pd
S = {'correct', 'diffrot'}
mode = sys.argv[1]
recs = {r['star_id']: r for r in pickle.load(open(f'runs/p43_clean/harness/changed_{mode}.pkl', 'rb'))}
for sid, r in recs.items():
    if not (r['variants']['local_screen2']['final_outcome'] in S and r['variants']['local_tol']['final_outcome'] not in S):
        continue
    print(f"\n=== {sid[-6:]} Ptrue={r['true_period']:.3f} local_tol={r['variants']['local_tol']['final_outcome']}")
    for v in ('lsq', 'local_tol'):
        for c in r['variants'][v]['candidates']:
            if c['outcome'] not in S and not c['passed']:
                continue
            t = [x for x in c['teeth']]
            hs = ' '.join(f"{x['height']:+.2f}{'' if np.isfinite(x['free_vertex']) else '*'}{'T' if x['tol_ok'] else ''}" for x in t)
            print(f"  {v:9s} P={c['P']:.3f} {c['outcome']:7s} pass={c['passed']:d} rchi={c['redchi']:.3g} snr={c.get('v_mean_height_snr',np.nan):.2f} "
                  f"fpos={c.get('v_frac_positive_heights',np.nan):.2f} ftol={c.get('v_frac_vertex_tol',np.nan):.2f} hl={c.get('v_half_lag_consistency',np.nan):+.3f} "
                  f"trim={c['n_trimmed']} drop={c['n_dropped']} req={c['n_required']} fail=[{c['failed']}]\n      h: {hs}")
