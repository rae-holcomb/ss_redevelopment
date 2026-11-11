# Changelog archive (before 2026-10-22)

Moved verbatim from `docs/PROJECT_CONTEXT.md`'s changelog on 2026-11-11 (P30), to cut the start-up reading for every
session. Newer entries stay in `PROJECT_CONTEXT.md`. Section and open-issue numbers mentioned here refer to
`PROJECT_CONTEXT.md`; issue numbers are stable.

- **(project start through the module split)**: everything in Sections
  1-4 above reflects the cumulative state as of the `comb_fit.py` /
  `guesses.py` split and the subsequent `_method_spectrum_features`
  extension + pyflakes cleanup. This document was created at that point,
  retroactively summarizing that whole arc rather than being built up
  entry-by-entry — so don't expect a detailed turn-by-turn history above
  this line.
- **2026-09-01 — TESS window-function / phase-coverage figures**
  (`window_function/`): recreated Fig. 4 of Rodel et al. 2024
  (doi:10.1093/mnras/stae474) with current pointings, using the real FFI
  (QLP + TESS-SPOC) window function of TIC 167814656, 45 sectors
  (S1-13, 27-39, 61-69, 87-90, 93-98), 7.45 yr baseline. Coverage is
  computed exactly by unioning phase arcs, not by binning; validated
  against a brute-force phase grid. Three findings worth carrying
  forward into any multi-sector work here: (1) mean per-sector duty
  cycle improved 0.919 (Cycle 1) to 0.967 (Cycles 5-8), so modern window
  functions are less lossy than the paper's Cycle-1 example; (2) TESS
  sectors are **no longer 27 d** — Cycle 8 (S97, S98) uses ~55-57 d
  pointings, so any code assuming a 27.4 d sector length is now wrong;
  (3) at long periods the dominant effect is per-cycle *clumping*, not
  intra-sector gaps — at fixed on-sky time, 45 sectors cover 0.98 of
  phase at P=730 d if back-to-back but only 0.74 as actually observed,
  and coverage is strongly non-monotonic in period. Relevant to open
  issue 6 (multi-sector support) and to the long-period end of the
  rotation search generally. See `window_function/README.md`.
- **2026-10-13: Agent/TDD infrastructure and synthetic baseline.**
  `REL_TOL` changed 0.15 → 0.10 project-wide and moved into the new
  `evaluation.py`, with P/2 and 2P as separate alias outcomes (Rae's decision).
  Added `synthetic.py`, a pytest suite (`tests/`, 64 tests, ~5 s),
  `benchmarks/scorecard.py` + `check_goals.py` + `goals.json`, the report
  contract (`write-report` skill, `tools/new_report.py`), six agent roles in
  `.claude/agents/`, `docs/TASKS.md` and `docs/AGENT_WORKFLOW.md`. Open issue 8
  is now a strict-xfail test and active goal G5. Baseline on
  `synthetic:n=200,seed=0`: 67% correct / 3.5% alias / 13% wrong / 16.5%
  declined. See `reports/2026-10-13_phase0-synthetic-baseline/` for this and for
  new open issues 10–12. Python env is `ss_local` (base anaconda is 3.8 without
  lmfit).
- **2026-10-13: SMARTS generation details** (Claytor et al. 2022,
  arXiv:2104.14566, Table 1; MAST HLSP page): 1M butterpy simulations;
  P_eq **uniform** in 0.1–180 d (so only 16.6% have P < 30 d, the
  pipeline's ~1-yr-baseline limit, and 5.5% have P < 10 d; dev sets must
  stratify on period, not sample at random); activity 0.1–10x solar
  log-uniform; spot decay timescale (1–10) x P_eq; latitudes lat_min 0–40,
  lat_max lat_min+5 to 80 deg; shear as in open issue 11; inclination
  uniform in sin^2 i; noise = ~2000 southern-CVZ galaxy FFI light curves
  (1000 train / 1000 test), 13 sectors, 30-min cadence, gaps linearly
  interpolated. A per-simulation input-parameter catalog exists on MAST
  (`_cat.tar.gz`). Env check: `codeastro` also passes the test suite and
  gives identical periods, but is about 40% slower per star than `ss_local` and has an
  incompatible numpy/scipy pair; staying on `ss_local` pending Rae's call.
- **2026-10-13: Differential-rotation-aware scoring.** Rae decided that a
  measured spot-latitude period is a success: we want whatever physical period
  is measurable. New `evaluation.DIFFROT` outcome and `success` metric
  (correct + diffrot), using band range P_eq/(1 - shear sin^2 lat) over
  [MINLAT, MAXLAT] (butterpy sign: shear > 0 solar-like; the paper's Eq. 1 sign
  is wrong, verified in butterpy's 2021 code). `synthetic.make_suite` now draws
  shear, latitude band and inclination from the SMARTS distributions. Scorecard
  v2 reads `DIFFROT`/`MINLAT`/`MAXLAT` from FITS headers. Full generation
  details are in the new `docs/SMARTS_REFERENCE.md` (cross-checked against
  10,000 real headers in results_products/batch1). Future feature F1
  (diffrot detection) is logged in TASKS.md.
  New baseline (`synthetic:n=200,seed=0`, scorecard v2): 67% success =
  46% correct + 21% diffrot; 3.5% alias, 5.5% wrong, 24% declined. Declines are
  now the main loss: high-SNR and solar-like-shear stars get rejected by the
  gates (draft goal G7). See `reports/2026-10-13_synthetic-baseline-v2-diffrot/`.
- **2026-10-13: Open issue 8 resolved (T4).** `guess_pairwise_histogram` now
  returns `[]` instead of raising when it finds fewer than 2 ACF peaks. The same
  applies to an empty period window, fewer than 3 lags, and zero-width histogram
  bins. On synthetic n=200 seed 0, its error rate went 0.275 → 0 and G5 passes.
  Final outcomes are identical, because callers already caught the exception.
  The stars that raised were low-SNR, nearly pole-on, or had short spot
  lifetimes. Issue 8 is kept as a placeholder so the numbering stays stable.
  Follow-ups P2 (other methods still raise) and P3 (prominence calibration) are
  in TASKS.md. See `reports/2026-10-13_t4-pairwise-histogram-no-raise/`.
- **2026-10-13: T4 merged; baseline updated.** `benchmarks/baselines/synth200_s0.json` is now T4's post-fix run (final outcomes identical to the v2 baseline; only pairwise_histogram's error rate went 0.275 → 0). G5 marked met. T4 report open questions 1–3 became tasks T13 (prominence recalibration), T14 (bit-reproducible scorecards) and T15 (no guess_* method raises).
- **2026-10-14: T14, bit-reproducible scorecards.** The run-to-run final-period noise (up to 9.8e-3
  relative, a few % of stars) came from Apple Accelerate's `ddot`, whose last bits depend on the memory
  alignment of its input (np.dot / np.linalg.norm on 1-D float64). scipy `least_squares` (trf) uses
  those norms every iteration, so the comb fit varied even within one process. Thread pinning alone does
  not fix it. New env **`ss_openblas`** (python 3.12, numpy 1.26.4 + scipy 1.16.0 on OpenBLAS;
  `environment.yml`, built via pip with a hand-picked scipy wheel) is now the project env. `ss_local`
  is kept but isn't reproducible. New `claude_files/determinism.py` pins BLAS to 1 thread per process
  in the scorecard and batch runners, and their default worker count is now one per physical
  P-core (4 on the i5) instead of 10.
  On synthetic n=200 seed 0: 21/21 run pairs have identical final periods (before: 16–20/200 differ
  per pair). Outcomes and check_goals are unchanged, and runtime shows no measurable change.
  The backend switch shifted final periods once, by ≤9e-5 relative, with no outcome changes.
  New test `tests/test_determinism.py` (slow; XFAILs on non-OpenBLAS). New `tools/compare_runs.py`.
  New open issue 13, proposals P4/P5. See `reports/2026-10-14_t14-bit-reproducible-scorecards/`.

- **2026-10-14: T15 (merged): no guess_* method raises.** Every degenerate exit in `guess_lombscargle`, `guess_acf_fft`, `guess_acf_fft_highpass` and `guess_wavelet` (plus the `_short` wrappers) now returns `[]`. That covers too few points, NaN/constant input, an empty, inverted, zero or NaN period window, too few FFT bins, and no GWPS peak above SNR. The new helper `_valid_period_window` checks the window. LS on constant flux and acf_fft on a flat or NaN ACF also return `[]` instead of meaningless candidates. A missing astropy still raises ImportError. 53 new introspection-parametrized tests. On synthetic n=200 seed 0, every method's error_rate is 0 (wavelet 0.035 → 0), and final outcomes and Stage 1 are identical. Follow-up P4 (wavelet SNR threshold calibration). See `reports/2026-10-14_t15-no-raise-all-guess-methods/`.
- **2026-10-14: T9 feature search (alias / spurious separation).** Five new
  per-candidate features in `ml_features.py` (FEATURE_DOCUMENTATION Section G):
  `acf_height_at_P`, `acf_half_lag_contrast`, `acf_odd_even_contrast`,
  `fold_half_corr`, `fold_half_corr_at_2P` (~1.7 ms/candidate). The feature
  schema gained 5 columns; existing definitions are unchanged. On synthetic n=200 seed 0,
  `fold_half_corr` is the best P-vs-2P separator (pooled AUC 0.06) and
  `acf_half_lag_contrast` the best overall (0.83). Open issue 12 is mostly gating:
  30/36 lost stars had their success candidate fail the gates. Literature table,
  decline analysis and proposals (P7, P8; numbered P4/P5 in the report) in `reports/2026-10-14_t9-feature-search/`.

- **2026-10-14: Baseline refreshed under `ss_openblas`; G8; Rae's decisions on T9/T15.** New `benchmarks/baselines/synth200_s0.json` (+ `synth200_s0_tables/` per-star CSVs for exact equivalence checks; see `benchmarks/baselines/README.md`). Outcomes are identical to the previous baseline for all 200 stars; 125 periods shifted ≤5.1e-5 relative (one-time Accelerate→OpenBLAS shift). New active goal **G8** (every method's error_rate = 0) with wildcard-goal support in `check_goals.py` (+ `tests/test_check_goals.py`). Decisions recorded in TASKS.md: T15 behaviors kept; T9's alias gate folded into P1 (validate on another seed); diffrot-label check required in T10; CNN-σ_P comparison in T11; new small task T17 (`s_ph_5P`); Lu+2020 = Astraea (AJ 160, 168), Lu+2022 = ZTF periods (AJ 164, 251), added to the T9 literature table and to T16d/D1.
- **2026-10-13: Long-period ACF limit lifted (P up to ~B/3).** `compute_acf` now reaches
  ~0.7 B (pair-coverage cut is the real limit) and can return per-lag pair counts. The comb fit's
  peak gate is adaptive (clamp(teeth in range, 2, 4)), and relaxed answers carry
  `few_peaks`/`n_teeth_available`, as Rae requested. Also added: a factor-of-2 guard (half-tooth height
  ratio instead of redchi for ~2x pairs); the lag-0 tooth skipped for unsaturated combs; decayed trailing
  teeth trimmed; no MAD rejection below the required count; sqrt(pair-count) residual weights (measured
  neutral); window decimation to <= 100 points (speed); two-view Stage 1 (guessers keep the B/3 ACF,
  plus out-of-reach long candidates from the 0.7 B ACF, `STAGE1_MAX_LAG_FRAC`); and edge-clipped
  windows judged against their nominal bound. Result: P/B scan correct at 0.10-0.33 went 0.00 -> 0.94;
  synth200 success 0.670 -> 0.725 with no regression guard tripped. Costs: rare red-noise long
  few-peak false positives (1/60 eval), and runtime +60% on 1-2 sectors / +30% on synth200. New benchmark `benchmarks/pb_scan.py`. See `reports/2026-10-13_acf-long-period-limit/`.
- **2026-10-14: ACF long-period branch merged; baseline refreshed.** Post-merge `ss_openblas` run on synth200 reproduces the branch's numbers: success 0.670 → 0.725 (12 stars gained, 1 lost), >30 d 0 → 0.75, 10–30 d 0.698 → 0.767, wrong 0.055 → 0.060, declined 0.24 → 0.185. All regression guards and G8 pass. New baseline at main df7699d; previous kept as `synth200_s0_pre_acf*`. At merge, its proposals P2–P4 became P9–P11 and its open issues 13–14 became 14–15. Rae's decisions: pair-count weighting kept (re-check in T16b); factor-of-2 threshold and few-peak noise guard folded into P1; `min_valid_frac` to T16c; `baseline_days` kept with a scale note; P11 (runtime) folded into T12.
- **2026-10-14: T18, arbitration and reliability gates.** Two new default gates in `fit_rotation_period`:
  a **half-lag consistency gate** (every ACF term h(kP) - h((k-1/2)P), k=1..6, > 0, or their mean = T9's
  `acf_half_lag_contrast` >= 0.1) and a **few-peak ACF-height guard** (few-peak candidates need raw ACF
  height >= 0.4 at P). New keyword args `half_lag_gate`, `half_lag_strong_contrast`, `half_lag_fallback_min`
  (softer option D), `few_peaks_min_acf_height`; `CandidateResult.passed_base_gates`; T9 helpers moved to
  `alias_metrics.py` (ML features bit-identical, tested). Tuned on synth200 seed 0 + pb_scan dev seeds,
  validated on seed 1 + pb_scan eval: wrong 0.060 -> 0 / 0.090 -> 0.015, alias 0.030 -> 0 / 0.050 -> 0.005,
  success 0.725 -> 0.730 / 0.700 -> 0.710, declined 0.185 -> 0.270 / 0.160 -> 0.270 (seed 0 / seed 1); red-noise
  FP 21/60 -> 0/60. check_goals PASS. Factor-2 threshold 0.7 and cross-method weighting measured neutral,
  unchanged. Open issues 10, 14b mostly resolved; 12 narrowed. New proposals P12-P14. **Rae (2026-10-15): strict kept, low-to-medium confidence; re-evaluate on dev_small (P8).** See `reports/2026-10-14_t18-arbitration-gates/`.
- **2026-10-15: T1 phase 1, SMARTS catalog audit.** Catalog (1M rows, 4.2 GB FITS incl. a 64×64 wavelet cube) extracted to the Data_Vault drive; scripts in `tools/audit/`. Clean: no NaNs, duplicates or bad periods; Table 1 confirmed except OVERLAP (bound max(CYCLE, √10)). Key traps: file number = catalog row index, not "Simulation Number"; SIM_ID ≥ 800k uses a disjoint galaxy-noise pool (use for holdout). ~12% of galaxies have corrupt orbits (flux far from 1); `load_smarts_fits` drops nearly all of it. 15% of stars have their whole active band outside P_eq ± 10%. Facts added to `SMARTS_REFERENCE.md` §6. See `reports/2026-10-15_t1-smarts-catalog-audit/`; phase 2 proposed as P15 (P16: short-P signal strength).
- **2026-10-14: T16a, patchy-coverage infrastructure.** New `claude_files/coverage.py` and
  `datasets/coverage/cycle1_patterns.csv` (121 Cycle-1 patterns with star counts from
  `target_df_shortform.csv`: 3,438 stars, 74.2% single, 15.9% consecutive-2, 1.4% gapped-2, 4.7% 3-5,
  2.7% 6-12, 1.1% full; supersedes `sector_configs.txt`). New `synthetic.make_lightcurve/make_suite(sectors=...)`,
  which mirrors `load_smarts_fits(sectors=...)` exactly on the time grid and NaNs. New scorecard
  `--coverage {full,real-weighted,<class>,pattern:...}` with `--coverage-seed`, and seven per-star coverage
  descriptor columns. `SCORECARD_VERSION` is now 3; metrics are unchanged, and `full` keeps the v2 dataset string.
  The default is bit-identical: synth200 matches the baseline 200/200, with only the P4 wavelet residue.
  Demo on the same 100 stars: success 0.69 (full) → 0.28 (single) / 0.30 (gapped-2) / 0.33 (real-weighted),
  mostly via confident ~0.2 d wrong answers (wrong 0.08 → 0.36 / 0.44 / 0.37), which extends open issue 10.
  Proposal P18 (numbered P12 in the report). See `reports/2026-10-14_t16a-coverage-infrastructure/`.
- **2026-10-15: T16a review: coverage libraries for every TESS year; classes split.** Rae's decisions:
  - split 3-5 and 6-12 into consecutive/gapped;
  - sample patterns uniformly within a class (don't tune to Year 1), then switch to frequency-weighted
    sampling once Years 1–7 were pooled (now the default);
  - descriptor columns approved;
  - `target_df_shortform.csv` = the brightest TESS stars, with tess-point coverage.

  New `datasets/coverage/cycle<N>_patterns.csv` for N = 1–7 (S1–13, 14–26, 27–39, 40–55, 56–69, 70–83,
  84–96), plus the pooled `all_cycles_patterns.csv`: 23,776 star-seasons, 534 patterns, now the default library.
  - Cycles 4–6 (14–16 sectors) are shifted (1,197 star-seasons) or truncated (635) into the 13-sector SMARTS year.
  - Northern-heavy years (2, 4, 6) are 48–58% single-sector, versus 73–75% for the others, with far more gapped coverage.
  - 413 of the 534 patterns never occur in Cycle 1, and no star-season after Cycle 3 is a full 13 sectors.

  The dataset string now records `coverage_library`. See the T16a report.
- **2026-10-15: T16a merged; baseline refreshed after T18.** New `synth200_s0.json` + tables = the post-T18 run (success 0.730, correct 0.500, diffrot 0.230, alias 0, wrong 0, declined 0.270), main @ bd9bed2; T16a's default output verified bit-identical to it (threads pinned). Previous baseline kept as `synth200_s0_pre_t18*`. `tests/conftest.py` now pins BLAS threads before importing numpy (T14 rule), so bit-identity tests are stable. T16a's proposal P12 renumbered P18 at merge. Rae's decisions recorded in TASKS.md: T16 scope = whole extended mission (T16b–f updated, new T16g multi-season sims via butterpy, to be discussed first), D2 leaning season-by-season, new D3 (training data beyond SMARTS).
- **2026-10-15: TESS sector vs year length.** From the TESS Science Office orbit times
  (tess.mit.edu/public/files/TESS_orbit_times.csv), sectors stay ~27 d in every year (median start-to-start
  26.8–27.7 d). The 14–16-sector years are longer than a year: 434 d (Year 4), 383 d (5), 376 d (6), against
  349–358 d for Years 1–3 and 7. So a 356-d SMARTS light curve can't hold Years 4–6 (why `coverage.fit_to_year`
  exists). Year 8 (S97–111) is irregular: ~55–58-d four-orbit sectors (S97–98) and single-orbit sectors (S102–103),
  15 sectors over 450 d. See `datasets/coverage/README.md`.
- **2026-10-15: T16a demo rerun after T18; Year 8 deferred.** Same 100 synthetic stars with the new
  coverage defaults (pooled Years 1–7, frequency-weighted) on the post-T18 pipeline:

  | coverage | success | wrong | none |
  |---|---|---|---|
  | full | 0.73 | 0 | 0.27 |
  | single | 0.29 | 0 | 0.70 |
  | consecutive-2 | 0.39 | 0.03 | 0.56 |
  | gapped-2 | 0.29 | 0.02 | 0.68 |
  | gapped-3-5 | 0.40 | 0.02 | 0.58 |
  | real-weighted | 0.31 | 0 | 0.69 |

  T18's gates turned the pre-T18 patchy-coverage wrong answers (0.36–0.44) into declines. The open problem is
  now over-declining on short data: half the single-sector stars under 1 d are declined. See T16b and the T16a
  report. Rae deferred Year 8+ coverage (irregular pointings).
- **2026-10-15: T16a open questions closed (Rae).** Keep `fit_to_year`'s truncation of over-long Year-4–6
  patterns (635 star-seasons), and keep pooling years in proportion to their star counts.
- **2026-10-15: T2 dev sets built (merged 2026-10-21; Rae took the recommended answer on all its open questions).** `datasets/dev_small.csv` (50), `dev_medium.csv` (1,000; both local in `data/smarts_dev/`), `holdout.csv` (3,000; drive `extracted/holdout/`), all SIM_ID ≥ 800k, disjoint, stratified per Rae's spec, coverage classes balanced per period bin; built by `tools/build_dev_sets.py` (seed 20261015; 455/5,200 candidates excluded as corrupt noise). New scorecard dataset options `--data-root` and `--ignore-manifest-sectors` (recorded in `config.dataset`; metric definitions unchanged). Trap: `load_smarts_fits(sectors=[...])` drops a requested sector if either orbit is flagged, unlike the default load, so coverage patterns are drawn from the new `native_whole_sectors` column (P20: the loader crashes when nothing survives). First real-SMARTS results: dev_medium full-year success 0.248 / wrong 0.088 / declined 0.650 vs synth200 0.730 / 0 / 0.270; assigned coverage 0.148 / 0.054 / 0.789. Mostly signal strength (low-amplitude tercile success 0.033), but ~8% confident wrong answers in every amplitude tercile (P19). Baselines `benchmarks/baselines/dev_{small,medium}_{assigned,fullyear}`. See `reports/2026-10-15_t2-dev-sets/`.
- **2026-10-15: T16e, coverage-context ML features (merged 2026-10-21).** Seven new columns in `ml_features.py` (FEATURE_DOCUMENTATION
  Section H), computed only from the light curve's finite-flux sampling (no sector label; no-leakage test):
  star-level `n_sectors_observed`, `n_contiguous_segments`, `longest_contiguous_days`, `largest_gap_days`; per-candidate
  `P_over_longest_segment`, `n_cycles_observed_total`, `tess_timescale_rel_dist`. ~38 µs/candidate. Existing
  definitions unchanged; `coverage.py` gained `contiguous_segments` / `sampling_descriptors` (descriptor output
  bit-identical). Skipped as duplicates: total span, P/total span, duty cycle, 1/P_over_longest_segment. On synthetic
  n=200 seed 0 under 6 coverage specs: several existing features change meaning on short data (success AUC full ->
  single: `P_over_lc_baseline` 0.26 -> 0.51, `n_cycles_spanned` 0.74 -> 0.49, `mean_height_snr` 0.88 -> 0.75;
  `s_ph` 0.61 -> 0.72), while `mean_peak_trough_depth_snr` and the T9 alias features are stable. The new features
  add little alone (logistic candidate AUC +0.000); with coverage x feature interactions +0.014 [0.007, 0.021],
  top-1 not significant. `tess_timescale_rel_dist` shows no signal on synthetic data. Multi-season features deferred
  (Rae). See `reports/2026-10-15_t16e-coverage-features/`.
- **2026-10-21 — T20 phase 1: galaxy-ID pass** (`reports/2026-10-21_t20-galaxy-pass-phase1/`):
  `tools/audit/galaxy_pass.py` (resumable per-bundle stream: int8 noise fingerprint, corrupt flags, stats)
  and `galaxy_groups.py` (template clustering, O(N x T x D)). On 31 bundles (172k files) -> 1,718 identical-noise
  light curves (999 used only by SIM_ID < 800k, 718 only by >= 800k, **1 by both**: ~0.14% of >= 800k files share
  a train-pool noise LC). New: **"sibling" noise LCs** (distinct LCs correlated at r = 0.5-0.99; T1's r > 0.8 cut
  lumped 5 such pairs). Full-corpus estimate ~45-60 min wall, ~2 CPU-h, 17.7 GB; awaiting Rae's go-ahead.
- **2026-10-21: T21 loader error (P20 resolved).** `preprocessing.load_smarts_fits` now raises `NoUsableSectorsError` (a `ValueError`; attributes `requested`/`flagged`/`absent`) before regridding when nothing survives selection: every requested sector flagged or beyond the file, `sectors=[]`, or every orbit flagged with `sectors=None`. Previously it crashed in `to_even_cadence` ("argmax of an empty sequence"). The scorecard, batch runners and `build_dev_sets.py` already caught per-star exceptions, so no caller changes; the scorecard records `none` + the error text (but leaves `true_period`/`period_bin` empty for that row). dev_small assigned and full-year bit-identical to baselines. See `reports/2026-10-21_t21-loader-no-usable-sectors/`.
- **2026-10-21: Noise-galaxy provenance from Zach Claytor.** `galaxy_files.csv` maps each simulation's
  *original* number (= catalog SIM_ID) to its noise light curve's TIC ID (1,719 sources). Validated against
  T20's blind fingerprint groups (99.65% single-TIC; pool split reproduced, with exactly one shared TIC). This gives
  T20 ground-truth galaxy IDs and real TIC IDs for source lookups. See `docs/SMARTS_REFERENCE.md` s7.
  T21 merged (`NoUsableSectorsError`).

- **2026-10-21 — T20 phase 2: SMARTS noise sources that aren't quiet** (`reports/2026-10-21_t20-noise-sources-phase2/`).
  Amplitude-aware median stacks of each of the 1,719 TIC noise light curves (Zach's IDs) show **206 periodic sources
  (121,502 files, 12.2%)**, mostly contamination by neighbouring variable stars (90 next to Gaia-classified variables;
  242 "galaxies" lie in crowded LMC fields); only 8 are common-mode systematics. **51 of dev_medium's 88 wrong answers
  equal the pipeline's period on the star's noise source alone** (null 0.4); wrong rate 43% on periodic sources vs 3.8%
  on quiet ones. Injection is multiplicative (f = g x s). Per-file flags on the drive:
  `extracted/galaxy_ids/noise_source_flags.csv.gz`. Keep/flag/exclude and a `noise_period` outcome are Rae's decisions.
- **2026-10-21 — T20 follow-up (f): independent FFI light curves.** No SPOC products exist for the noise sources
  (TIC EXTENDED); QLP/TASOC/ELEANOR-LITE light curves (sectors 1-5) for 19 sources confirm the SMARTS period in 9/12
  testable periodic sources (P < 5 d) and 0/4 quiet controls; amplitudes differ strongly between reductions
  (aperture-dependent neighbour light), supporting contamination. See the T20 phase-2 report.
- **2026-10-21: T16b gate half (instrumentation + diagnosis).** `fit_rotation_period` now records per-candidate `gate_results`/`failed_gates` and a structured `decline_reason` (named in `message`); scorecard gains descriptor columns (`decline_gates`, `best_success_failed_gates`, ...) and `per_candidate.csv`. Final periods/outcomes bit-identical on synth200 s0, dev_small and dev_medium (assigned). Diagnosis: the half-lag gate fires on every decline but the declined stars' correct candidates are mostly noise-like (dev_medium 202/312 with contrast < 0.02); single-sector losses of full-year successes are mostly Stage 1 / coverage-limited (only 6/79 have a candidate matching the full-year P). Gate statistics barely scale with coverage (height SNR ~0.95x); the few-peak guard's applicability does. Best offline rule (few height 0.25 + contrast-conditional SNR/vertex for >= 4 teeth) is +1-7 per 200 synthetic with 0 wrong/0 noise FP (validated on seed 1) but +10/+17 wrong on dev_medium; recommendation: no default change until T20. Proposed P21. See `reports/2026-10-21_t16b-gate-diagnosis/`.
- **2026-10-21: Rae's decisions on T16b's open questions** (all as recommended): no gate relaxation before T20; per-method mechanisms folded into T16c (waits for D4); height-SNR denominator + pair-count weighting into T16d; band-width breakdown + scorecard v4 as new small task T22; year tag deferred. See TASKS.md T16b.
- **2026-10-21: Own SMARTS-like simulations recorded as an option under D3** (not committed): quiet TESS galaxy light curves (T20-vetted; existing FFI products first, own reduction as fallback) + newest butterpy; for multi-year baselines and more short-period training data. Validate against SMARTS before scaling. See TASKS.md D3.
