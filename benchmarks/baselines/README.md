# Benchmark baselines

**Use the `_t5` files for new work** (current pipeline code, since T5's hybrid peak ranking merged 2026-11-11). The `_v5`
files describe the pre-T5 pipeline and stay for before/after comparisons. Scorecard v6 (T26) adds only the `detectable`
keys; no v6 baseline is committed yet (the speed-up pass's "before" run is meant to supply it), so goals on those keys
report n/a. The other files are kept because a goal, test or tool cites them.

| File | What | Env / commit |
|---|---|---|
| `dev_medium_fullyear_t5.json` + `_t5_tables/` | **Current judging baseline** (T5, 2026-11-11): same dataset string as `_v5`, on the T5 pipeline (hybrid peak ranking in `lombscargle`/`acf_fft`). Success 0.251, wrong 0.088, <1 d union recall 0.698. Scorecard v5 keys (no `detectable`). | `ss_openblas`, run as `99c46ef-dirty` = pipeline code of `15c88d8` (checked: runs postdate the last code edit) |
| `dev_medium_assigned_t5.json` + `_t5_tables/` | Same, assigned coverage. Success 0.147, wrong 0.055, <1 d union recall 0.779. | same |
| `synth200_s0_t5.json` + `_t5_tables/` | `synthetic:n=200,seed=0` on the T5 pipeline. Success 0.755, wrong 0. | same |
| `dev_medium_fullyear_v5.json` + `_v5_tables/` | Pre-T5 judging baseline (T24/T25/P22b, 2026-11-10): `datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=ignored`. Every v4 key is bit-identical to v4; adds `final.noise.*` (wrong split into `wrong_noise_period`/`wrong_other`), `final.quiet_noise`, `final.quiet_noise_strict`, `final.periodic_noise`; per-star `noise_*` columns. `per_candidate.csv` (`beyond_reach` rows) isn't committed (10 MB). | `ss_openblas`, claude/t24-t25-scorecard-v5-84eca3 |
| `dev_medium_assigned_v5.json` + `_v5_tables/` | Same, assigned coverage (`sectors=manifest`). | same |
| `dev_small_assigned_v5.json`, `synth200_s0_v5.json` | v5 re-runs of the quick-check sets (synth200 has no noise columns, so only the v4 keys). | same |
| `dev_small_fullyear.json` + `_tables/` | `...dev_small.csv,data_root=data/smarts_dev,sectors=ignored` (v3 keys only; no v4/v5 re-run exists). Success 0.260, wrong 0.040, declined 0.680. | `ss_openblas`, 4c2a994, 2026-10-15 (T2) |
| `dev_medium_fullyear.json` + `_tables/` | v3 baseline for dev_medium full-year (success 0.248, wrong 0.088, declined 0.650). Kept because goals G9–G12 quote it and `tools/audit/galaxy_phase2_analysis.py` reads its `per_star.csv`. | same |
| `synth200_s0.json` + `synth200_s0_tables/` | v2/v3 baseline for `synthetic:n=200,seed=0` (success 0.730, wrong 0, declined 0.270). Cited by `goals.json`, `check_goals.py`'s usage line, and `tests/test_coverage.py` (exact per-star equivalence). | `ss_openblas`, main @ bd9bed2, 2026-10-15 |
| `synth200_s0_v1_scorecard1.json` | Pre-diffrot baseline (scorecard v1, older synthetic population). Not comparable. Kept only because `goals.json` names it. | 2026-10-13 |

Refreshing a baseline is Rae's call. Record why in docs/PROJECT_CONTEXT.md's changelog.

**Version notes.** v3 keys are unchanged and comparable across v3–v6. `check_goals.py` reports a goal on a key the
baseline lacks (v4: `final.by_band_width.*`, `final.by_coverage_class.*`, `final.(un)searchable.*`; v5: `final.noise.*`,
`final.quiet_noise*`; v6: `detectable`) as n/a. The noise join comes from `benchmarks/noise_sources_dev.csv`
(dev_small + dev_medium only; `tools/build_noise_table.py`), so `synthetic:` runs and the holdout have no noise columns.

**Pruned 2026-11-11 (P30)**, recoverable from git at commit `d3f73e6`: `synth200_s0_pre_acf` (+tables, before the ACF
long-period merge, success 0.670), `synth200_s0_pre_t18` (+tables, before T18's gates, success 0.725 / wrong 0.060),
`synth200_s0_accelerate` (post-T4 run under Apple Accelerate), `synth200_s0_v4`, `dev_small_assigned` (+tables) and
`_v4`, `dev_medium_assigned` (+tables) and `_v4` (+tables), `dev_medium_fullyear_v4` (+tables). Each was a history-only
copy or was superseded by a `_v5` file on the same dataset string.
