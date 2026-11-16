"""Outcomes of each variant on the stars local_tol changed vs _v7 (harness), and repro checks."""
import pickle, sys
import pandas as pd
sys.path.insert(0, 'reports/2026-11-16_p43-clean-rerun'); import p43lib as L
S = {'correct', 'diffrot'}
V = sys.argv[1].split(',') if len(sys.argv) > 1 else ['local_screen2', 'lsq', 'local', 'local_tol', 'local_tol_pos', 'local_tol_trim', 'local_tol_trim_pos']
for mode, tag in [('fullyear', 'dmfy'), ('assigned', 'dmas')]:
    recs = pickle.load(open(f'runs/p43_clean/harness/changed_{mode}.pkl', 'rb'))
    b = pd.read_csv(f'benchmarks/baselines/dev_medium_clean_{mode}_v7_tables/per_star.csv').set_index('star_id')
    n = pd.read_csv(f'runs/p43_clean/sc/local_tol_{tag}/per_star.csv').set_index('star_id')
    T = pd.DataFrame({v: L.per_star_frame(recs, v).set_index('star_id').final_outcome for v in V if v in recs[0]['variants']})
    T['v7'] = b.final_outcome.reindex(T.index)
    if 'local_screen2' in T:
        print(mode, 'repro v7:', (T.local_screen2 == T.v7).mean(), 'repro local_tol:', (T.local_tol == n.final_outcome.reindex(T.index)).mean())
    for v in T.columns:
        s = T[v].isin(S); print(f'  {v:20s} succ {s.sum():3d} wrong {(T[v]=="wrong").sum():3d} none {(T[v]=="none").sum():3d}')
    cols = [c for c in T.columns if c not in ('lsq', 'local')]
    print(T[T.v7.isin(S) & ~T.local_tol.isin(S)][cols].to_string())
