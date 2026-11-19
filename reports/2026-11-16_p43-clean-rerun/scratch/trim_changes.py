"""What local_tol_trim changes vs local_tol (all stars): gains, losses, new wrongs, with v7 outcome."""
import pandas as pd
for mode in ('fullyear', 'assigned'):
    T = pd.read_csv(f'reports/2026-11-16_p43-clean-rerun/data/screen_{mode}.csv', index_col=0)
    d = T[T.local_tol != T.local_tol_trim][['period_bin', 'local_screen2', 'local_tol', 'local_tol_trim', 'local_tol_cs', 'local_tol_cs_trim']]
    print(mode); print(d.to_string())
