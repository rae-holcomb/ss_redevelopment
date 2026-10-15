# T16a: patchy-coverage infrastructure (coverage library, synthetic sectors=, scorecard --coverage)

**Date:** 2026-10-14 · **Author:** method-improver agent (T16a session) · **Branch:** `claude/t16a-coverage-infra-fe5be3` · **Status:** draft

## TL;DR
- **New coverage library:** `datasets/coverage/cycle1_patterns.csv` lists the 121 real Cycle-1 sector patterns with star counts (sum = 3,438). By class: 74.2% `single`, 15.9% `consecutive-2`, 1.4% `gapped-2`, 4.7% `3-5`, 2.7% `6-12`, 1.1% `full`. It's built by `tools/build_coverage_library.py`, and a test checks it against a fresh build.
- **Synthetic `sectors=` option:** `synthetic.make_lightcurve(sectors=[...])` / `make_suite(sectors=...)` give the same time grid and NaN structure as `load_smarts_fits(sectors=...)` (exact match in tests). The star itself is unchanged; only coverage differs.
- **Scorecard `--coverage` option:** `--coverage {full, real-weighted, single, consecutive-2, gapped-2, 3-5, 6-12, pattern:…}` with `--coverage-seed`, for synthetic and FITS manifests. It adds seven coverage descriptor columns to `per_star.csv`, and `SCORECARD_VERSION` is now 3. Metric definitions are unchanged.
- **The default is unchanged:** `--coverage full` on `synthetic:n=200,seed=0` reproduces the committed baseline for 200/200 final periods and outcomes, bit for bit, and `check_goals` passes with every delta +0.000. The only differences are in wavelet on 3 stars (top-1 ≤1.5e-10 relative, and candidate counts): the known P4 nondeterminism, not this change (see "What didn't work").
- **Demo (illustration for T16b, not a conclusion):** on the same 100 stars, success falls from 0.69 (full) to 0.28 (single), 0.30 (gapped-2) and 0.33 (real-weighted). The loss is mostly confident wrong answers, not declines: wrong is 0.08 → 0.36 / 0.44 / 0.37. All 36 single-sector wrong answers are 0.13–0.42 d periods that passed every gate.

## Question / goal
This is task **T16a** in `docs/TASKS.md`, the infrastructure for T16b–f. It makes no method changes and touches no goal in `goals.json`. Scope:
- a frequency-weighted real coverage library;
- a synthetic `sectors=` option that mirrors `load_smarts_fits(sectors=...)`;
- scorecard dataset options and per-star coverage descriptors;
- tests;
- a small demo.

I read discussion items D1 and D2. Nothing here pre-empts them: there are no searched-range outputs, and SMARTS can't represent multi-year gaps.

## What I did

**1. What `target_df_shortform.csv` contains.** 8,156 rows, one per unique TIC, with columns:
- unnamed index;
- `TIC`;
- `ec_long`, `ec_lat` (ecliptic coordinates, deg);
- `num_s` (number of sectors);
- `sec` (string repr of the sorted list of every observed sector, 1–96).

`num_s != len(sec)` for 37 rows (0.45%). I use `sec` throughout. 3,438 stars have at least one sector in 1–13. All of them are southern (ecliptic latitude −89° to −6°), as expected for Cycle 1.

**2. Coverage library.** The new module `claude_files/coverage.py` holds the pattern parsing, `coverage_class`, `build_library`, `sample_patterns` and `coverage_descriptors`. `tools/build_coverage_library.py` writes `datasets/coverage/cycle1_patterns.csv`, which has a `#` provenance line with the input's sha256. Provenance and biases are in `datasets/coverage/README.md`. The patterns are exactly the 121 in `sector_configs.txt` (tested); the file is now superseded but kept.

**3. Synthetic `sectors=`.** The star is generated on the full `n_sectors` grid with the same RNG draws, then reduced to the kept sectors with `load_smarts_fits` conventions:
- sector s = orbits 2s−1, 2s;
- unobserved and gap cadences become NaN;
- the grid is trimmed to run from the first to the last kept valid cadence, with absolute times kept (no shift to 0);
- flux is re-centered on the kept data.

`sectors=None` takes the old code path untouched.

One small asymmetry: an explicit all-13 pattern also trims the trailing orbit-gap cadences that the legacy grid keeps. That's why `--coverage full` maps to `sectors=None`.

**4. Scorecard.**
- New flags: `--coverage`, `--coverage-seed` (default 0), `--coverage-weighting uniform|frequency` (for class specs; default uniform, as specified) and `--coverage-library`.
- Patterns are drawn once per star, in star order, from `default_rng(coverage_seed)`, independently of the suite seed. The same stars can therefore be re-run under a different coverage draw.
- FITS manifests pass the pattern as `load_smarts_fits(sectors=...)`. A manifest may instead carry a per-row `sectors` column ("1-2-13"), which is what T16f needs for frozen dev sets.
- For FITS, the `sectors` descriptor is what survived flagged-sector removal (`pre.sectors_used`), and the requested pattern goes in `meta_sectors_requested`.
- The dataset string is unchanged for `full`, so existing baselines and goals still apply. Otherwise it gains `,coverage=<spec>[,coverage_weighting=frequency],coverage_seed=<s>`. `config.coverage` records the spec, seed, weighting and library path.

**5. Commands** (all with `--n-workers 2`):
```
scorecard.py --synthetic 200 --seed 0 --coverage full                       -> synthetic:n=200,seed=0
scorecard.py --synthetic 100 --seed 0 --coverage single      --coverage-seed 0 -> synthetic:n=100,seed=0,coverage=single,coverage_seed=0
scorecard.py --synthetic 100 --seed 0 --coverage gapped-2    --coverage-seed 0 -> ...,coverage=gapped-2,coverage_seed=0
scorecard.py --synthetic 100 --seed 0 --coverage real-weighted --coverage-seed 0 -> ...,coverage=real-weighted,coverage_seed=0
```
The 100-star suite is the first 100 stars of the 200-star suite, so `full` for those 100 is taken from the n=200 run.

### Coverage classes

| class | definition | Cycle-1 stars | patterns | note |
|---|---|---|---|---|
| `single` | 1 sector | 2552 (74.2%) | 13 | |
| `consecutive-2` | sectors s, s+1 | 547 (15.9%) | 12 | |
| `gapped-2` | 2 non-adjacent sectors | 49 (1.4%) | 10 | 21 of the 49 are `1-13`: adjacent on the sky but 11 months apart, the largest possible gap |
| `3-5` | 3–5 sectors | 160 (4.7%) | 37 | 127 of the 160 are consecutive runs |
| `6-12` | 6–12 sectors | 92 (2.7%) | 48 | mostly near-full years missing 1–2 sectors (48 of 92 have 11–12 sectors) |
| `full` | all 13 | 38 (1.1%) | 1 | |

I kept the six requested classes. The data does suggest one refinement, which I didn't make, because it's Rae's call and T16b can do it with the `consecutive` column: split `3-5` into consecutive (127) and gapped (33).

### Per-star descriptor columns (scorecard v3)
- `n_sectors`, `sectors` ("1-2-13"), `coverage_class`;
- `total_baseline_days` (first to last finite cadence);
- `longest_contiguous_days` (longest stretch with no gap > 5 d, so orbit gaps don't split a stretch but a missing sector does);
- `largest_gap_days`;
- `duty_cycle` (finite cadences / cadences spanned by the baseline).

## Results

### Library
![coverage library](figures/fig1_coverage_library.png)
Single sectors dominate (74%). Beyond consecutive pairs, the long tail of 88 patterns covers just 301 stars, so `real-weighted` almost never draws multi-sector patterns. Per-class runs are needed to study those.

### Default bit-identity
`tools/compare_runs.py runs/t16a_full200 benchmarks/baselines/synth200_s0_tables`:
```
200 stars in common
final_period differs: 0 stars
final_outcome differs: 0 stars
  acf_fft / acf_fft_highpass / acf_fft_short / lombscargle / lombscargle_short / pairwise_histogram: 0 / 0
  wavelet                top1 differs:   2   n_candidates differs:   3
```
The wavelet differences are the P4 residue (see "What didn't work"):
- top-1 differs on 00038 by 1.5e-10 relative and on 00058 by 1.9e-11, with the same outcome;
- candidate counts differ on 00038, 00058 and 00158.

Wavelet isn't pooled into the fit.

`check_goals.py runs/t16a_full200/metrics.json --baseline benchmarks/baselines/synth200_s0.json`:
```
   base     new   delta  limit  metric
  0.725   0.725  +0.000  0.020  final.overall.success
  0.689   0.689  +0.000  0.060  final.by_bin.1-3d.success
  0.767   0.767  +0.000  0.060  final.by_bin.10-30d.success
  0.861   0.861  +0.000  0.060  final.by_bin.3-10d.success
  0.647   0.647  +0.000  0.060  final.by_bin.<1d.success
  0.750   0.750  +0.000  0.060  final.by_bin.>30d.success
  0.889   0.889  +0.000  0.060  union_recall.by_bin.1-3d
  0.977   0.977  +0.000  0.060  union_recall.by_bin.10-30d
  0.972   0.972  +0.000  0.060  union_recall.by_bin.3-10d
  0.882   0.882  +0.000  0.060  union_recall.by_bin.<1d
  1.000   1.000  +0.000  0.060  union_recall.by_bin.>30d
OVERALL: PASS
```
Tests pin bit-identity in three ways:
- `test_default_suite_bit_identical_to_pre_t16a`: sha256 of `make_suite(5, seed=0)` time and flux, recorded from main 589cb8d's `synthetic.py`;
- `test_default_suite_matches_baseline_population`: per-star parameters against the baseline CSV;
- `test_default_scorecard_rows_bit_identical_to_baseline` (slow): 3 stars end to end, exact final periods.

### Demo: outcome vs coverage (same 100 stars)

| run | n | success | correct | diffrot | alias | wrong | none |
|---|---|---|---|---|---|---|---|
| full (first 100 of n=200) | 100 | 0.690 | 0.440 | 0.250 | 0.020 | 0.080 | 0.210 |
| single | 100 | 0.280 | 0.150 | 0.130 | 0.040 | 0.360 | 0.320 |
| gapped-2 | 100 | 0.300 | 0.180 | 0.120 | 0.020 | 0.440 | 0.240 |
| real-weighted | 100 | 0.330 | 0.190 | 0.140 | 0.040 | 0.370 | 0.260 |

![outcome vs coverage](figures/fig2_outcome_vs_coverage.png)
Look at the red: patchy coverage mostly turns stars into confident wrong answers, not declines. The right panel is the real-weighted run split by sampled class. Its n is tiny outside `single` (69), so it only shows the infrastructure working.

![success by bin](figures/fig3_success_by_bin_and_coverage.png)
On one or two sectors, success at 10–30 d and >30 d drops to 0. That's expected, since one sector is about 27 d. Short periods lose much less, but still lose.

![outcome vs P/span](figures/fig4_outcome_vs_p_over_span.png)
`wrong` climbs with P / longest contiguous span: 0.15 (≤0.03), 0.33 (0.03–0.1), 0.41 (0.1–1/3), 0.65 (1/3–1), 0.81 (>1). Stars whose period can't fit in the data get a confident short answer instead of a decline. This plot is the axis for T16c's `max_period(coverage)`.

### Failure mode: short confident wrong answers
In the single-sector run, all 36 wrong answers are 0.13–0.42 d (median 0.21 d). They aren't only low-SNR stars: by SNR bin, wrong is 8 (≤2), 18 (2–5) and 10 (5–20).

![failure examples](figures/fig5_single_sector_wrong_examples.png)
One star per SNR bin. The ACF has no peak at the chosen lag, only noise texture on a slowly decaying envelope. The comb still passed the gates at 0.15–0.39 d. This looks like open issues 10/15 (low-SNR short-lag answers; the lag-0 tooth acting as a noise filter), strongly amplified by short baselines.

A quick side check, not in the notebook: pure noise on one sector (12 seeds, white 0.002 plus red 0.0005) gave 2/12 confident answers near 0.16 d.

### Runtime
Median s/star at 2 workers: 3.51 (full), 0.62 (single), 0.90 (gapped-2), 1.00 (real-weighted).

## What didn't work
- **Wavelet counts looked like a regression, but aren't.** The full run's wavelet output differed from the baseline on stars 00038, 00058 and 00158 (counts; top-1 by ≤1.5e-10 relative), which at first looked like a side effect of the extra descriptor computation. Pristine main 589cb8d, run in-process, gives 6 candidates on 00058 every time, including under heap perturbation. This branch gave 8 once in-process (while the background scorecard was loading the CPU), then 6 and 6 on reruns, and 7 in the pool run. So the same code produces different counts across invocations: this is P4's wavelet nondeterminism. P4 currently says top-1 differs by ~2e-11. Here it reached 1.5e-10 (00038), and the candidate *count* can change too (6/7/8 on 00058). Wavelet isn't pooled into the fit, so final answers are unaffected.
- **An end-to-end test through a fake SMARTS FITS can't be exact.** `split_lightcurve` re-detects gaps heuristically: on a synthetic star it flagged one orbit and shifted gap edges by a few cadences. The exact equivalence test therefore runs against `recombine_segments` + `to_even_cadence`, the deterministic part of `load_smarts_fits`. The FITS round-trip test uses tolerances: sectors equal, start within 0.1 d, NaN mismatch below 1%.
- **In-process scorecard tests need `threadpool_limits(1)`.** Without it, star 00002's final period differed from the baseline in the 14th digit, because pytest has already imported numpy, so BLAS isn't pinned.

## Open questions for Rae
1. **`target_df_shortform.csv` provenance.** How were these 8,156 stars selected (magnitude cut? a rotation catalog's cross-match? CVZ-heavy?) The library inherits that selection. Known biases: Cycle 1 only, southern sky only, and stars observed only after Cycle 1 dropped (4,718 of 8,156). Should we build a Cycle-2 (northern) library too? Or a library from a better-defined target list, e.g. all 2-min targets or a magnitude-limited FFI sample?
   **Answered (Rae, 2026-10-15):** the few thousand brightest TESS stars, with sector coverage
   from `tess-point` at each sky position, meant as a guide to typical coverage patterns. So the
   library is geometric FFI coverage at bright-star positions, which suits SMARTS (FFI). See
   `datasets/coverage/README.md`. The Cycle-2 / northern-library question remains open.
2. **Class boundaries.** Keep the six classes, or split `3-5` into consecutive (127 stars) and gapped (33)? And should `1-13` (21 of 49 `gapped-2` stars, an 11-month gap) be its own case? It behaves very differently from `5-7`.
3. **Within-class sampling.** It's uniform over patterns as specified, so `gapped-2` draws `1-13` 10% of the time instead of 43%. `--coverage-weighting frequency` is available. Which should T16b and T16f use by default?
4. **Approve the new `per_star.csv` columns** (the task asked for your OK): `n_sectors, sectors, coverage_class, total_baseline_days, longest_contiguous_days, largest_gap_days, duty_cycle`, plus `meta_sectors_requested` for FITS. The scorecard version is bumped to 3. `GAP_SPLIT_DAYS = 5` defines "contiguous". Is that right for real SMARTS gaps, where some Cycle-1 interruptions are 2–4 d?
5. **The headline failure mode** (36% confident wrong on one sector, all ~0.2 d). It belongs to T16b/T16d, but it also bears directly on T18's gates. I left a note for the T18 session in the shared notes folder. T18's gate changes should be checked with `--coverage single` as well as on full-year synth200.
6. **Proposed per-coverage-class metrics (T16f).** Add `final.by_coverage_class.<class>.{...outcome fractions...}` and `union_recall.by_coverage_class.<class>` alongside `by_bin`, computed the same way from the new `coverage_class` column. Possibly also `final.by_coverage_class_and_bin`. Regression guards per class would need goals.json entries. Not implemented here; that's T16f.

## Reproduce
```bash
PY=/opt/anaconda3/envs/ss_openblas/bin/python
$PY tools/build_coverage_library.py
$PY benchmarks/scorecard.py --synthetic 200 --seed 0 --coverage full --n-workers 2 --outdir runs/t16a_full200
for c in single gapped-2 real-weighted; do $PY benchmarks/scorecard.py --synthetic 100 --seed 0 --coverage $c --coverage-seed 0 --n-workers 2 --outdir runs/t16a_$c; done
$PY tools/compare_runs.py runs/t16a_full200 benchmarks/baselines/synth200_s0_tables
$PY benchmarks/check_goals.py runs/t16a_full200/metrics.json --baseline benchmarks/baselines/synth200_s0.json
# copy runs/t16a_*/{per_star.csv,metrics.json} into data/<run>/ (already committed), then:
$PY tools/run_notebook.py reports/2026-10-14_t16a-coverage-infrastructure/notebook.ipynb
$PY -m pytest -q tests/test_coverage.py
```
