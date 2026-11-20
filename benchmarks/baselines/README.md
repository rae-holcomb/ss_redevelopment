# Benchmark baselines

**Judge on the `_v8` files `dev_medium_clean_{fullyear,assigned}_v8.json`** (Rae, 2026-11-19: default comb fit
`local_tol_trim`). Fresh scorecard runs of `datasets/dev_medium_clean.csv` on main after the decayed-tail merges
(`ss_openblas`, 3 workers, code of `a111147` + the default flip). Full-year success 0.265, wrong 0.024; assigned
0.153 / 0.030; `synth200_s0_v8` 0.755 / 0.005. Every per-star value is bit-identical to the P43 clean rerun's
`local_tol_trim` runs (only `meta_path` differs: worktree vs main). Against `_v7`: dev_medium both modes PASS;
synth200 flags `final.by_bin.>30d.success` 1.000 -> 0.875 (one star, synth-s0-00099; P45). `per_candidate.csv` not
committed (size). `tests/test_coverage.py` now checks exact equivalence against `synth200_s0_v8_tables`.

The `_v7` notes below are history (default `local_screen2`).

**Previously (2026-11-16):** noise-clean files `dev_medium_clean_{fullyear,assigned}_v7.json`. They are the `_v7`
tables restricted to `datasets/dev_medium_clean.csv` (874 stars: the 126 `noise_exclude` stars dropped) and recomputed
with `tools/rescore_subset.py` (the same `scorecard.compute_metrics`; unrestricted, it reproduces `_v7` exactly: 0 of
1396 / 1819 values differ). Full-year success 0.254, wrong 0.038; assigned 0.146 / 0.029. `config.rescored_from` names
the source; `config.wall_seconds` is dropped. The full-set `_v7` files below are history.

**Use the `_v7` files for new work** (2026-11-13, T31 merge). They are the current pipeline: reproducible wavelet
(P4), default comb fit `local_screen2`, default CWT `lean` (Rae's decisions). Against `_v6w` (same code before the
default flip), every metrics.json value is identical (1394 / 1817 / 724 values, 0 differ). All final answers and
per-method results are identical too, except one full-year star (009530: P 1.82055 -> 1.82125, still correct). Per-star
diagnostics of *declined* candidates (`decline_gates`, `best_success_*`, `n_success_candidates`) differ for ~170-300 stars:
candidates failing > 2 gates under the cheap `local` screen keep the local fit's record instead of an lsq refit.
`_v6w` replaced `_v6` (Rae, 2026-11-13); `_v6` and older stay as history.

Earlier note: every v5 key of
the `_v6` files is bit-identical to the `_t5` files on the same dataset string (checked with
`tools/compare_scorecard_keys.py`: 1151 / 1439 / 724 values, 0 differ), so `_v6` = `_t5` + the v6 `detectable` keys
(`final.detectable.*`, `final.undetectable.*`, `final.detectable_searchable.*`; per-star `detectability`/`detectable`).
The `_t5` files stay as history; the `_v5` files describe the pre-T5 pipeline. The other files are kept because a goal,
test or tool cites them.

| File | What | Env / commit |
|---|---|---|
| `dev_medium_clean_fullyear_v8.json` + `_tables/` | **Current judging baseline** (2026-11-19): `datasets/dev_medium_clean.csv,data_root=data/smarts_dev,sectors=ignored`, `config.comb_variant=local_tol_trim`. Success 0.265, wrong 0.024. | `ss_openblas`, 3 workers, main after `a111147` |
| `dev_medium_clean_assigned_v8.json` + `_tables/` | Same, assigned coverage. Success 0.153, wrong 0.030. | same |
| `profile15_fullyear_v8.json` | RT2 runtime baseline (2026-11-20): `profile_pipeline.py --star-ids-file benchmarks/profile_stars.txt --ignore-manifest-sectors --features --repeats 3`, `local_tol_trim`. Mean 0.739 s/star. | `ss_openblas`, serial, `ab859e4` |
| `synth200_s0_v8.json` + `_tables/` | `synthetic:n=200,seed=0`, same code. Success 0.755, wrong 0.005. Used by `tests/test_coverage.py`. | same |
| `dev_medium_clean_fullyear_v7.json` + `_tables/` | Previous judging baseline (`local_screen2`): `datasets/dev_medium_clean.csv,data_root=data/smarts_dev,sectors=ignored`, `_v7` restricted to the clean stars. Success 0.254, wrong 0.038. | from `_v7` via `tools/rescore_subset.py` |
| `dev_medium_clean_assigned_v7.json` + `_tables/` | Same, assigned coverage (`sectors=manifest`). Success 0.146, wrong 0.029. | same |
| `dev_medium_fullyear_v7.json` + `_v7_tables/` | Previous judging baseline, all 1,000 stars (T31 merge, 2026-11-13): `datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=ignored`, scorecard v6 keys, `config.comb_variant=local_screen2`, `config.cwt_variant=lean`. Success 0.251, wrong 0.088. `per_candidate.csv` not committed (size). | `ss_openblas`, 3 workers, `5d14f3f` |
| `dev_medium_assigned_v7.json` + `_v7_tables/` | Same, assigned coverage (`sectors=manifest`). Success 0.147, wrong 0.055. | same |
| `synth200_s0_v7.json` + `_v7_tables/` | `synthetic:n=200,seed=0` on the same code. Success 0.755, wrong 0. | same |
| `dev_medium_fullyear_v6.json` + `_v6_tables/` | Previous judging baseline (T12 before run, 2026-11-12): `datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=ignored`, scorecard v6. Success 0.251, wrong 0.088 (same as `_t5`). `per_candidate.csv` not committed (size). | `ss_openblas`, 4 workers, pipeline code of `9063c95` (= T5 pipeline, unchanged since) |
| `dev_medium_assigned_v6.json` + `_v6_tables/` | Same, assigned coverage (`sectors=manifest`). Success 0.147, wrong 0.055. | same |
| `synth200_s0_v6.json` + `_v6_tables/` | `synthetic:n=200,seed=0`, scorecard v6 (no detectability table rows for synthetic, so the v6 groups are empty). Success 0.755, wrong 0. | same |
| `dev_medium_fullyear_v6w.json` + `_v6w_tables/` | Adopted by Rae as the `_v6` replacement (T31 Part A, 2026-11-13; superseded by `_v7` the same day): same dataset string as `_v6`, on code with the reproducible wavelet (P4). Identical to `_v6` in every metrics.json value; per star only `methods.wavelet.top1` differs (last ~10 digits, 884 stars). A rerun is bit-identical (0 of 1,007,054 values differ). | `ss_openblas`, 3 workers, `32531d8` |
| `dev_medium_assigned_v6w.json` + `_v6w_tables/` | Same, assigned coverage (818 stars differ in `methods.wavelet.top1` only; rerun 0 of 863,445 differ). | same |
| `dev_medium_fullyear_t5.json` + `_t5_tables/` | Previous judging baseline (T5, 2026-11-11): same dataset string as `_v5`, on the T5 pipeline (hybrid peak ranking in `lombscargle`/`acf_fft`). Success 0.251, wrong 0.088, <1 d union recall 0.698. Scorecard v5 keys (no `detectable`). | `ss_openblas`, run as `99c46ef-dirty` = pipeline code of `15c88d8` (checked: runs postdate the last code edit) |
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
`final.quiet_noise*`; v6: `detectable`) as n/a; against a `_v6` baseline the v6 keys are guarded too. The noise join comes from `benchmarks/noise_sources_dev.csv`
(dev_small + dev_medium only; `tools/build_noise_table.py`), so `synthetic:` runs and the holdout have no noise columns.

**Pruned 2026-11-11 (P30)**, recoverable from git at commit `d3f73e6`: `synth200_s0_pre_acf` (+tables, before the ACF
long-period merge, success 0.670), `synth200_s0_pre_t18` (+tables, before T18's gates, success 0.725 / wrong 0.060),
`synth200_s0_accelerate` (post-T4 run under Apple Accelerate), `synth200_s0_v4`, `dev_small_assigned` (+tables) and
`_v4`, `dev_medium_assigned` (+tables) and `_v4` (+tables), `dev_medium_fullyear_v4` (+tables). Each was a history-only
copy or was superseded by a `_v5` file on the same dataset string.
