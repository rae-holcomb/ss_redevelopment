# Benchmark baselines

| File | What | Env / commit |
|---|---|---|
| `synth200_s0.json` | **Current baseline** for `synthetic:n=200,seed=0` (scorecard v2). `check_goals.py --baseline` default. | `ss_openblas` (OpenBLAS, bit-reproducible), main @ 761315c (T14+T15+T9 merged), 2026-10-14 |
| `synth200_s0_tables/` | `per_star.csv` / `per_method.csv` of that same run, for **exact** output-equivalence checks (e.g. T12) via `tools/compare_runs.py`. | same |
| `synth200_s0_accelerate.json` | Previous baseline (post-T4), same population and outcomes, but run under Apple Accelerate (not bit-reproducible). History only. | `ss_local`, 2026-10-13 |
| `synth200_s0_v1_scorecard1.json` | Pre-diffrot baseline (scorecard v1, older synthetic population). Not comparable. History only. | 2026-10-13 |

Refreshing a baseline is Rae's call. Record why in docs/PROJECT_CONTEXT.md's changelog.
