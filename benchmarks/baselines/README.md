# Benchmark baselines

| File | What | Env / commit |
|---|---|---|
| `synth200_s0.json` | **Current baseline** for `synthetic:n=200,seed=0` (scorecard v2 dataset string; also valid for v3 `--coverage full`). Success 0.730, wrong 0, declined 0.270. | `ss_openblas`, main @ bd9bed2 (T18 merged; T16a verified bit-identical on default output), 2026-10-15 |
| `dev_small_assigned.json` + `_tables/` | `datasets/dev_small.csv,data_root=data/smarts_dev,sectors=manifest` (assigned coverage). Success 0.160, wrong 0.040, declined 0.780. | `ss_openblas`, 4c2a994, 2026-10-15 (T2) |
| `dev_small_fullyear.json` + `_tables/` | `...dev_small.csv,data_root=data/smarts_dev,sectors=ignored` (same stars, full year). Success 0.260, wrong 0.040, declined 0.680. | same |
| `dev_medium_assigned.json` + `_tables/` | `datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=manifest`. Success 0.148, wrong 0.054, declined 0.789. | same |
| `dev_medium_fullyear.json` + `_tables/` | `...dev_medium.csv,data_root=data/smarts_dev,sectors=ignored`. Success 0.248, wrong 0.088, declined 0.650. | same |
| `dev_medium_fullyear_v4.json` + `_v4_tables/` | **Scorecard v4 baseline** (T22/T23, 2026-10-22), same dataset string as `dev_medium_fullyear.json`. Every v3 key is bit-identical to the v3 file; adds `final.by_band_width`, `final.by_coverage_class`, `final.searchable.*`, `final.unsearchable.*` (and per-star `acf_reach`, `P_over_reach`, `searchable`, `band_width*`, `flux_amp` in the tables). Use this one for new work. | `ss_openblas`, T22/T23 branch, 2026-10-22 |
| `dev_medium_assigned_v4.json` + `_v4_tables/` | Same, for assigned coverage (the by-coverage-class metrics are only informative here). | same |
| `dev_small_assigned_v4.json`, `synth200_s0_v4.json` | v4 re-runs of the quick-check sets (v3 keys identical to their v3 files). | same |
| `dev_medium_fullyear_v5.json` + `_v5_tables/` | **Scorecard v5 baseline** (T24/T25/P22b, 2026-11-10), same dataset string as the v4 file. Every v4 key is bit-identical; adds `final.noise.*` (wrong split into `wrong_noise_period`/`wrong_other`), `final.quiet_noise`, `final.quiet_noise_strict`, `final.periodic_noise`; per-star `noise_*` columns; `per_candidate.csv` `beyond_reach` rows (not committed here; 10 MB). Use this one for new work. | `ss_openblas`, claude/t24-t25-scorecard-v5-84eca3 |
| `dev_medium_assigned_v5.json` + `_v5_tables/` | Same, assigned coverage. | same |
| `dev_small_assigned_v5.json`, `synth200_s0_v5.json` | v5 re-runs of the quick-check sets (synth200 has no noise columns, so only the v4 keys). | same |
| `synth200_s0_tables/` | `per_star.csv` / `per_method.csv` of that same run, for **exact** output-equivalence checks (e.g. T12, `tests/test_coverage.py`). | same |
| `synth200_s0_pre_t18.json` + `_tables/` | Before T18's gates (success 0.725, wrong 0.060). History. | `ss_openblas`, main @ df7699d |
| `synth200_s0_pre_acf.json` + `_tables/` | Baseline before the ACF long-period merge (success 0.670). History. | `ss_openblas`, main @ 761315c |
| `synth200_s0_accelerate.json` | Previous baseline (post-T4), same population and outcomes, but run under Apple Accelerate (not bit-reproducible). History only. | `ss_local`, 2026-10-13 |
| `synth200_s0_v1_scorecard1.json` | Pre-diffrot baseline (scorecard v1, older synthetic population). Not comparable. History only. | 2026-10-13 |

Refreshing a baseline is Rae's call. Record why in docs/PROJECT_CONTEXT.md's changelog.

**Scorecard v4 note (T22/T23).** Baselines without `_v4` were written by scorecard v2/v3 and lack the v4 keys
(`final.by_band_width.*`, `final.by_coverage_class.*`, `final.searchable.*`, `final.unsearchable.*`). `check_goals.py`
reports a goal/guard on such a key as n/a against an old baseline, so the new groups are *reported, not guarded*, until a
`_v4` baseline is passed. The v3 keys are unchanged and still comparable across versions. The `_v4` files supersede the
non-`_v4` dev_medium ones as the current baselines; the old ones stay for history.

**Scorecard v5 note (T24/T25).** `_v5` files supersede `_v4`. `check_goals.py` reports goals on the new `final.noise.*` / `final.quiet_noise*` keys as n/a against a `_v4` or older baseline. The noise join comes from `benchmarks/noise_sources_dev.csv` (dev_small + dev_medium only; built by `tools/build_noise_table.py`), so `synthetic:` runs and the holdout have no noise columns. `_v4_tables` pre-date the 0.7-baseline reach-cap fix, so their candidate-count columns differ from current runs (metrics and outcomes do not).
