"""Recompute the multiplicative-injection evidence (phase-2 table) from injection_model.csv."""
import pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
d = pd.read_csv(f"{A}/reports/2026-10-21_t20-noise-sources-phase2/data/injection_model.csv")
for thr in (1, 3, 10):
    s = d[d.g_abs_dev_median_corrupt > thr]
    print(f">{thr}: files {len(s)} TICs {s.galaxy.nunique()} add {s.add_ratio.median():.2f} mul {s.mul_ratio.median():.2f}")
s = d[d.g_abs_dev_median_corrupt < 1]
print("<1 (most of sample):", len(s), "add", round(s.add_ratio.median(), 2), "mul", round(s.mul_ratio.median(), 2))
