# Benchmark baselines

| File | What | Env / commit |
|---|---|---|
| `synth200_s0.json` | **Current baseline** for `synthetic:n=200,seed=0` (scorecard v2 dataset string; also valid for v3 `--coverage full`). Success 0.730, wrong 0, declined 0.270. | `ss_openblas`, main @ bd9bed2 (T18 merged; T16a verified bit-identical on default output), 2026-10-15 |
| `dev_small_assigned.json` + `_tables/` | `datasets/dev_small.csv,data_root=data/smarts_dev,sectors=manifest` (assigned coverage). Success 0.160, wrong 0.040, declined 0.780. | `ss_openblas`, 4c2a994, 2026-10-15 (T2) |
| `dev_small_fullyear.json` + `_tables/` | `...dev_small.csv,data_root=data/smarts_dev,sectors=ignored` (same stars, full year). Success 0.260, wrong 0.040, declined 0.680. | same |
| `dev_medium_assigned.json` + `_tables/` | `datasets/dev_medium.csv,data_root=data/smarts_dev,sectors=manifest`. Success 0.148, wrong 0.054, declined 0.789. | same |
| `dev_medium_fullyear.json` + `_tables/` | `...dev_medium.csv,data_root=data/smarts_dev,sectors=ignored`. Success 0.248, wrong 0.088, declined 0.650. | same |
| `synth200_s0_tables/` | `per_star.csv` / `per_method.csv` of that same run, for **exact** output-equivalence checks (e.g. T12, `tests/test_coverage.py`). | same |
| `synth200_s0_pre_t18.json` + `_tables/` | Before T18's gates (success 0.725, wrong 0.060). History. | `ss_openblas`, main @ df7699d |
| `synth200_s0_pre_acf.json` + `_tables/` | Baseline before the ACF long-period merge (success 0.670). History. | `ss_openblas`, main @ 761315c |
| `synth200_s0_accelerate.json` | Previous baseline (post-T4), same population and outcomes, but run under Apple Accelerate (not bit-reproducible). History only. | `ss_local`, 2026-10-13 |
| `synth200_s0_v1_scorecard1.json` | Pre-diffrot baseline (scorecard v1, older synthetic population). Not comparable. History only. | 2026-10-13 |

Refreshing a baseline is Rae's call. Record why in docs/PROJECT_CONTEXT.md's changelog.
