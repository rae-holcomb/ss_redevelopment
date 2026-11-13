# T32 progress log

- 13:15 setup: data symlinked; T31 runs reused from the t31 worktree (`runs/t31/runs/`), v7 runs from main `runs/t31_merge/`.
- 13:25 step 1 (`step1_peaks.py`, 117 star-modes, no scorecard): figure `fig_step1_residual_vs_n.png`.
  Gains = linear fan-out of tracked peaks from frozen n*P0 windows (P0 off by ~3%); none->wrong = offset from n=0
  plus random scatter. RMS residual about the fitted comb: gains 0.053, wrong 0.085, shared-correct 0.030.
- 13:50 five switches + 3 combinations in `COMB_VARIANTS`; tests `tests/test_t32_local_drift.py` (16).
- 14:00 dev_small candidate study (`cand_stats.py`): 30-40% of correct candidates' teeth have no in-window free
  vertex under frozen windows (tracking: ~10%); tol-fraction stats don't separate, scatter does (0.017 vs 0.05).
  Thresholds fixed from dev_small: scatter <= 0.04, min_free 0.5, min_frac_tol 0.6 (untracked) / 0.8 (tracked).
- full pytest 500 passed (niced).
