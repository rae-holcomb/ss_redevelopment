"""Screening table: every variant on all dev_medium_clean stars (harness), both modes, vs local_tol and _v7.

Writes data/screen_{mode}.csv (per-star outcome per variant) and data/screen_summary.csv.
The harness's local_screen2 column must equal the _v7 per-star table and local_tol the scorecard (checked).
"""
import pickle, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import p43lib as L
S = {'correct', 'diffrot'}
rows = []
for mode, tag in [('fullyear', 'dmfy'), ('assigned', 'dmas')]:
    recs = pickle.load(open(L.ROOT / f'runs/p43_clean/harness/all_{mode}.pkl', 'rb'))
    V = list(recs[0]['variants'])
    T = pd.DataFrame({v: L.per_star_frame(recs, v).set_index('star_id').final_outcome for v in V})
    b = pd.read_csv(L.ROOT / f'benchmarks/baselines/dev_medium_clean_{mode}_v7_tables/per_star.csv').set_index('star_id')
    n = pd.read_csv(L.ROOT / f'runs/p43_clean/sc/local_tol_{tag}/per_star.csv').set_index('star_id')
    assert (T.local_screen2 == b.final_outcome.reindex(T.index)).all(), 'harness does not reproduce _v7'
    assert (T.local_tol == n.final_outcome.reindex(T.index)).all(), 'harness does not reproduce local_tol scorecard'
    T['period_bin'] = b.period_bin.reindex(T.index)
    T.to_csv(HERE / f'data/screen_{mode}.csv')
    N = len(T)
    for v in V:
        s, s_tol, s_v7 = T[v].isin(S), T.local_tol.isin(S), T.local_screen2.isin(S)
        ans = T[v] != 'none'
        rows.append(dict(mode=mode, variant=v, success=s.mean(), wrong=(T[v] == 'wrong').mean(),
                         alias=T[v].isin(['half', 'double']).mean(), none=(T[v] == 'none').mean(),
                         precision=s.sum() / max(ans.sum(), 1),
                         gained_vs_tol=int((s & ~s_tol).sum()), lost_vs_tol=int((~s & s_tol).sum()),
                         newwrong_vs_tol=int(((T[v] == 'wrong') & (T.local_tol != 'wrong')).sum()),
                         lost_v7_recovered=int((s & s_v7 & ~s_tol).sum()), lost_v7_total=int((s_v7 & ~s_tol).sum())))
out = pd.DataFrame(rows)
out.to_csv(HERE / 'data/screen_summary.csv', index=False)
pd.set_option('display.width', 200)
print(out.round(3).to_string(index=False))
