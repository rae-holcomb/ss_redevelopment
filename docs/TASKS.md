# Task board

Each task is sized for one agent session. Rae owns this file's priorities
and acceptance criteria. Agents update only the **Status** line of their
own task (and add follow-up tasks under "Proposed", never directly to the
active list).

Status values: `blocked` · `ready` · `in progress (<branch>)` · `in review (<report folder>)` · `done (<report folder>)`

Agent roles are defined in `.claude/agents/`. Every task ends with a
report (`write-report` skill) and, if code changed, a before/after
`benchmarks/check_goals.py` table.

---

## Phase 0: infrastructure

### T0. Test suite, synthetic generator, scorecard, report contract
- **Status:** done (2026-10-13, built in-session with Rae; see changelog in PROJECT_CONTEXT.md)

---

## Phase 1: data

### T1. Audit the SMARTS training set
- **Agent:** `data-auditor`
- **Status:** blocked (data being copied to external drive)
- **Goal:** know what's in the ~1M-file corpus and what's wrong with it before
  training on it.
- **Done when:** report covers inventory, label sanity, recoverability
  (amplitude vs background noise), gap/baseline coverage, distributions and
  biases, duplicates/blends. checks for patterns in sectors with spuriously 
  large noise. Includes a CSV of files recommended for exclusion
  (corrupt) or flagging (hard-but-valid), each with a reason.

### T2. Build frozen dev sets and holdout
- **Agent:** `dev-set-builder`
- **Status:** blocked (on T1 inventory)
- **Note:** SMARTS periods are uniform in 0.1–180 d, so only ~17% of the corpus
  has P < 30 d (the 1-yr-baseline limit) and ~5.5% has P < 10 d. Random sampling
  would make dev sets mostly unrecoverable long periods. Stratify, and use the
  MAST input-parameter catalog rather than opening FITS headers.
- **Note (2026-10-14):** dev sets must also stratify on **sector-coverage pattern**,
  not just full-year light curves. See T16f.
- **Goal:** `datasets/dev_small.csv` (~30–50), `dev_medium.csv` (~1000),
  `holdout.csv` (~2000–5000), stratified by period/SNR/baseline, reproducible
  from `tools/build_dev_sets.py`.
- **Done when:** Rae approves the strata and counts; disjointness test passes;
  baseline scorecard run on dev_small committed to `benchmarks/baselines/`.

### T3. Synthetic-vs-SMARTS realism check
- **Agent:** `data-auditor`
- **Status:** blocked (on T1)
- **Goal:** compare `claude_files/synthetic.py` light curves against real SMARTS
  ones (noise spectra, gap patterns, amplitude distribution, ACF morphology) and
  tune `make_suite` so synthetic results predict dev-set results.
- **Done when:** scorecard per-bin outcomes on synthetic vs dev_small are
  compared side by side; discrepancies explained. `PERIOD` is equatorial and the
  shear can be large (PROJECT_CONTEXT open issue 11): quantify how often
  the spot-latitude periods fall outside REL_TOL of P_eq, and propose a
  correctness criterion for Rae.

---

## Phase 2: candidate generation and fitting

All Phase 2 tasks use `method-improver`. Until T2 lands, judge on
`--synthetic 200 --seed 0`. After that, judge on dev_small, with dev_medium
for confirmation (ask Rae first).

### T4. Fix `guess_pairwise_histogram` crash (open issue 8)
- **Status:** done, merged 2026-10-13 (report `reports/2026-10-13_t4-pairwise-histogram-no-raise/`).
  Its open questions 1–3 became T13–T15.
- **Goal:** return `[]` instead of raising when <2 ACF peaks are found.
- **Done when:** the two strict-xfail entries in `tests/test_guesses.py::KNOWN_FAILURES`
  are removed and the suite passes. A good first task for trying out the agent workflow.

### T5. Short-period recovery (<3 d) (open issue 1)
- **Status:** ready
- **Goal:** improve Stage 1 recall and final correctness in the `<1d` and `1-3d`
  bins. Separate the Nyquist/points-per-cycle wall (unfixable) from search
  failures (fixable).
- **Done when:** goals G2/G3 in `benchmarks/goals.json` pass with no regression guard
  tripped.

### T6. Long-period recovery and honest failure (>10 d)
- **Status:** in progress. Long-period half **done and merged 2026-10-14**
  (report `reports/2026-10-13_acf-long-period-limit/`: P up to ~B/3, synth200 >30 d success
  0 -> 0.75). The low-SNR ~0.2-0.5 d wrong-answer half (open issue 10) is still open; see open issue 15
  (the lag-0 tooth) for a clue.
- **Goal:** understand final-answer failures at long P: how many are the
  baseline limit (<4 ACF peaks in 1/3 baseline) versus fixable. Consider whether
  the 1/3-baseline ACF lag cap or `min_peaks_required=4` should adapt.
  **Also:** the Phase 0 baseline found low-SNR (1–3) long-period stars where the
  pipeline confidently reports a ~0.2–0.5 d period instead of declining (7/200
  stars). Find out why those candidates pass the reliability gates.
- **Done when:** a report with a failure taxonomy; a code change only if
  justified; G6 (wrong rate) improves and G1 is not regressed.

### T7. Per-method audits (one session per method)
- **Status:** ready
- **Goal:** for each of `lombscargle`, `acf_fft`, `acf_fft_highpass`, `wavelet`,
  diagnose its failure modes per period bin and propose or implement one
  improvement each.
- **Done when:** one report per method.

### T8. TESS orbit-gap aliasing check (open issue 7). Low priority
- **Agent:** `method-improver` (diagnostic; a code change only if a problem is found)
- **Status:** ready, low priority (Rae, 2026-10-13)
- **Background:** gaps multiply the light curve by a window function, so periodic
  gaps (period T_gap) put spurious peaks at f_rot ± k/T_gap. FFT-based methods
  (`acf_fft*`, the masked-FFT ACF, `wavelet` via interpolation) are the most exposed;
  Lomb-Scargle the least. Earlier checks found real SMARTS/TESS gaps irregular
  enough to be low-risk overall, but the ~1 d gap every ~13.7 d TESS orbit *is*
  periodic and sits inside our 1–30 d period range.
- **Scope (TESS orbit gaps only):**
  1. Sweep rotation period densely around T_gap = 13.7 d and its simple ratios
     (6.85, 27.4 d, …) on synthetic light curves with the standard orbit-gap grid.
  2. For each method and for the final answer, measure success rate and whether
     candidates or answers land on the predicted alias periods
     1/|1/P_rot ± k/T_gap|.
  3. Show the window function's spectrum and ACF for reference.
- **Done when:** a short report says whether stars near 13.7 d (and its ratios)
  show a measurable dip in success or excess aliases relative to neighbouring
  periods, per method, with a recommendation (no action, or a targeted fix).
- **Out of scope for now:** ground-based day/night cadence, momentum-dump gaps, and
  multi-sector yearly clumping. **Re-evaluate if the pipeline is applied to other
  missions (e.g. Roman)**, whose observing windows may be strongly periodic.

### T13. Recalibrate ACF peak-finding prominence
- **Agent:** `method-improver`
- **Status:** ready (from the T4 report, approved by Rae 2026-10-13)
- **Problem:** the adaptive threshold (prominence = 5 × std of the ACF's second
  difference, `guesses._acf_peak_candidates`) is miscalibrated both ways. It misses
  rotation bumps that are obvious by eye (T4 report stars 00003, 00004) and lets
  hundreds of noise wiggles through elsewhere (median 15 peaks, max 585).
- **Ideas to compare:** second difference of a smoothed ACF; prominence relative
  to the ACF envelope; retry at lower prominence when < 2 peaks are found.
- **Done when:** a peak-count/recall diagnostic by period bin and SNR before and
  after, `pairwise_histogram` recall improves (especially <3 d, where it is weakest),
  and no regression guard trips. Note that `_acf_peak_candidates` may also feed
  other code paths, so check its callers.

### T14. Make scorecard runs bit-reproducible
- **Agent:** `performance-optimizer` or `method-improver` (small task)
- **Status:** done, merged 2026-10-14 (report `reports/2026-10-14_t14-bit-reproducible-scorecards/`). Cause: Apple Accelerate's alignment-dependent ddot. Fix: `ss_openblas` env + BLAS pinned to 1 thread per process. See `reports/2026-10-14_t14-bit-reproducible-scorecards/`.
- **Problem:** with identical inputs, 17/200 final periods differed between runs by up to
  2.3e-4 relative. That's harmless at REL_TOL=0.10, but it makes before/after
  diffs noisy and blocks an exact output-equivalence test (needed by T12).
- **Goal:** find the cause (suspects: multithreaded BLAS/FFT inside workers,
  worker scheduling, optimizer tolerances, unseeded randomness in fitting/RANSAC) and
  fix it, e.g. single-threaded BLAS (`OMP_NUM_THREADS=1` etc.) in scorecard
  workers. Measure the runtime effect, since single-threading may also *help* with 10 workers.
- **Done when:** two consecutive scorecard runs give identical `per_star.csv`
  final periods, plus a test that checks determinism on a few stars.

### T15. No guess_* method raises on degenerate input
- **Agent:** `method-improver`
- **Status:** done, merged 2026-10-14 (report `reports/2026-10-14_t15-no-raise-all-guess-methods/`).
- **Rae's decisions (2026-10-14) on its open questions:** keep silent `[]` for wavelet
  on NaN flux; a missing astropy may still raise `ImportError` (an environment error);
  keep `[]` for the non-raising degenerate cases; adopted goal **G8** (every method's
  error_rate = 0, wildcard). The wavelet threshold idea is P6.
- **Problem:** `guess_lombscargle`, `guess_acf_fft` and `guess_wavelet` still
  `raise RuntimeError` on degenerate input (guesses.py around lines 380, 510 and
  1000–1030). That breaks the "return [], never raise" contract T4 established.
- **Goal:** same treatment as T4: write tests that reach each early exit (degenerate
  inputs: all-NaN, constant flux, too-short baseline, empty period window), then
  make them return `[]`.
- **Done when:** the new tests pass and every guess method's error rate is 0 on the
  synthetic scorecard. Consider promoting a goal "all methods' error_rate = 0"
  (Rae edits goals.json).

### T18. Arbitration and reliability gates (promoted from P1, 2026-10-14)
- **Agent:** `method-improver`
- **Status:** done, awaiting review (2026-10-14). Half-lag consistency gate + few-peak ACF-height guard
  (new defaults). synth200 seed 0: success 0.725 → 0.730, wrong 0.060 → 0, alias 0.030 → 0, declined
  0.185 → 0.270; seed 1 (held out): 0.700 → 0.710, wrong 0.090 → 0.015, alias 0.050 → 0.005, declined
  0.160 → 0.270. pb_scan red-noise FP 21/60 → 0/60 (eval). check_goals PASS. Factor-2 threshold and
  cross-method weighting unchanged (measured neutral). Strict-vs-softer (option D) decision left to Rae.
  See `reports/2026-10-14_t18-arbitration-gates/`; follow-ups P12–P14.
- **Why:** Stage 1 lists a success period for 88–100% of stars per bin, but the final
  answer succeeds for only 65–86% (open issue 12). 30/36 of the lost stars fail the
  *gates* (T9). Meanwhile red noise gets a confident period 22/60 times in
  `benchmarks/pb_scan.py`, and low-SNR stars get ~0.2–0.5 d noise answers (open issues
  10, 14b, 15).
- **Scope** (everything collected in P1, P7 and P10):
  - candidate selection and gating in `fit_rotation_period` (`comb_fit.py`);
  - T9's `acf_half_lag_contrast` gate/selection;
  - a few-peak noise guard (ACF peak height, PDM theta, contrast);
  - the factor-of-2 guard's 0.7 threshold vs T9's `fold_half_corr`;
  - cross-method support (`n_contributing_methods`).
- **Measure on:** synth200 seed 0 **and seed 1**, plus `pb_scan.py` (signal + 120
  noise LCs). Tune on seed 0 and pb_scan's dev seeds; validate on seed 1 and pb_scan's
  eval seeds.
- **Done when:** a before/after on all of those, success up (G1, G4), wrong down (G6),
  declines down (G7), red-noise FP well below 22/60, no regression guard tripped, and the
  seed-1 validation holds up. Any success-vs-wrong trade-off goes to Rae.

### T17. Add `s_ph_5P` (literature-standard S_ph) alongside `s_ph`
- **Agent:** `feature-researcher` (small)
- **Status:** ready (Rae, 2026-10-14, from T9 open question 3)
- **Do:** add `s_ph_5P`: S_ph computed over segments of 5·P, as in Mathur+2014 and
  García+2014. Keep the existing `s_ph` (1·P bins) **unchanged**, so feature-importance
  analysis decides between them. Add a docstring, a FEATURE_DOCUMENTATION entry and a
  test. Report its cost and its AUC next to `s_ph` on synthetic n=200 (re-check on
  dev_small via P8).

### T16. Patchy sector coverage: robustness, search limits, and safeguards
- **Requested by:** Rae, 2026-10-14. Multi-part; do the subtasks in order.
- **Why:** most real TESS stars aren't observed for 13 consecutive sectors. Many
  have 1–2 sectors or non-consecutive ones. We want as much period information as
  possible from patchy data, and honest limits where we can't get it.
- **Already exists:**
  - `preprocessing.load_smarts_fits(sectors=[...])` keeps only chosen sectors of a
    SMARTS light curve (via `split_smarts_lc.recombine_segments`).
  - `sector_configs.txt` has 121 unique real-star sector patterns: Cycle 1 sectors
    from `target_df_shortform.csv`, built in `ss_claude_sandbox.ipynb` cell 10.
    By length: 13 one-sector, 22 two-sector, …, 1 thirteen-sector; 31 fully consecutive.
- **Known limitation:** SMARTS spans one year, so multi-year revisit gaps can't be
  emulated (see the window_function work and open issue 6). Stars observed only
  after Cycle 1 were dropped when the patterns were built.
- **Coordination:** the "Extend ACF period limit" session (branch
  `claude/angry-einstein-cb10da`) is changing the ACF lag / period limit. T16c must
  build on whatever lands from it.

#### T16a. Coverage-degradation infrastructure
- **Agent:** `method-improver` (infrastructure only)
- **Status:** in progress (started 2026-10-14)
- **Do:**
  - Add a `sectors` / coverage-pattern option to `synthetic.make_lightcurve`,
    mirroring `load_smarts_fits(sectors=...)`.
  - Rebuild the real-star coverage library *with frequencies*: count how many real
    stars share each pattern, don't just dedupe. Commit it as data with a builder
    script.
  - Add scorecard options to run on a named coverage class or to sample patterns by
    real-star frequency, e.g. `--coverage real-weighted | single | consecutive-2 |
    gapped`.
  - Record per-star coverage descriptors in `per_star.csv`: n_sectors, total
    baseline, longest contiguous span, duty cycle, largest gap.
  - Tests.
- **Note:** scorecard *metric definitions* stay unchanged. This adds dataset
  options and descriptor columns, so bump `SCORECARD_VERSION` and get Rae's OK on the
  new columns.

#### T16b. How do candidate methods degrade with coverage?
- **Also (Rae, 2026-10-14):** re-evaluate the comb fit's sqrt(pair-count) residual
  weighting (measured neutral on full-year data in the ACF long-period report). It
  should matter most for gappy coverage. Keep or remove it on that evidence.
- **Agent:** `method-improver` (diagnostic report; no method changes)
- **Status:** blocked on T16a
- **Do:**
  - Run the scorecard across coverage classes: 1 sector; 2 consecutive;
    2 non-consecutive; 3–5 gapped; ≥ 6.
  - Map per-method recall/success and final success against P relative to the
    longest contiguous span and to the total baseline.
  - Identify failure mechanisms per method. Expected suspects:
    - ACF truncation by `min_valid_frac` across big gaps;
    - `pairwise_histogram` running out of peaks;
    - Lomb-Scargle aliasing from the sector-gap window (relates to T8);
    - wavelet interpolating across months-long gaps (likely invalid).
- **Done when:** a report with a method × coverage-class × period-bin success table
  and failure-mode examples.

#### T16c. Search upper limits for patchy data
- **Also (Rae, 2026-10-14):** test lowering `compute_acf`'s `min_valid_frac` from 0.3
  to ~0.25. At P = B/3 on 2 sectors only 6/12 are correct, because P sits on the
  ACF pair-coverage edge (ACF long-period report, open issue 14a).
- **Agent:** `method-improver`
- **Status:** blocked on T16b (the ACF long-period change it builds on was merged 2026-10-14)
- **Do:** define the maximum searchable / reportable period as a function of
  coverage (longest contiguous span, total baseline, ACF valid-pair coverage).
  Measure reliability (success vs wrong) as P approaches each candidate limit, and
  pick limits that keep "wrong" low.
- **Done when:** a `max_period(coverage)` rule implemented in one place, with
  justification plots, and goals proposed for Rae.

#### T16d. Method policy and safeguards for patchy data
- **Agent:** `method-improver`
- **Status:** blocked on T16b/c
- **Ideas to evaluate (pick by evidence):**
  - Per-coverage method selection, e.g. skip wavelet when gaps exceed X.
  - Segment-wise analysis: ACF or Lomb-Scargle per contiguous chunk, then require
    agreement across chunks. A cross-sector consistency check as a reliability signal.
  - Gap-aware alias rejection using the coverage window's own spectrum.
  - Short-data reliability gates: the gates were tuned on full-year data, so
    recalibrate their thresholds.
  - **Partial-information outputs:** see discussion item **D1** (searched-range
    reporting). Rae leans toward reporting it as pipeline output rather than scoring
    it. Don't build it until D1 is settled.
- **Done when:** success on patchy classes improves without regressing full
  coverage, and the "wrong" rate on 1–2-sector stars stays low.
- **Prior art:** Astraea (Lu+2020, AJ 160, 168) predicts periods *longer than one
  sector* from 27-d light-curve variability statistics plus Gaia parameters (~9%
  uncertainty on Kepler 27-d segments up to 150 d; ~55% on real TESS). It's a
  natural comparison and a possible prior or limit companion for single-sector stars.

#### T16e. Coverage-aware ML features
- **Agent:** `feature-researcher`
- **Status:** blocked on T16a
- **Do:** add coverage descriptors (n_sectors, longest span, duty cycle, P/longest
  span, n_cycles observed) to `ml_features.py`, so the ranker can learn how much to
  trust a candidate under a given coverage. Relates to open issue 6 (multi-sector
  support).

#### T16f. Gappy coverage in the standard development datasets
- **Agent:** `dev-set-builder` (for the real-data sets), `method-improver` (for synthetic)
- **Status:** blocked on T16a (and on T2 for the real-data part)
- **Why:** so every agent's routine before/after scorecard exercises patchy data,
  not just the clean full-year case. Regressions on gappy stars should be caught
  as they happen, not discovered at the end.
- **Do:**
  - **Synthetic:** define a standard coverage mix for the default synthetic
    benchmark (a fixed fraction of full-year, consecutive, non-consecutive and
    single-sector stars), using the T16a coverage library. Re-baseline
    `benchmarks/baselines/` once, recording the old baseline as history, as with v2.
  - **Real data (T2):** add coverage pattern as a stratification axis for
    `dev_small`, `dev_medium` and `holdout`. Apply patterns from the T16a library to
    SMARTS files via `load_smarts_fits(sectors=...)`, stored as a per-row column
    in the manifest (and read by the scorecard), so each set is frozen and reproducible.
  - **Scorecard/goals:** report metrics per coverage class as well as per period bin,
    and add a regression guard per coverage class. Rae owns `goals.json`, so propose
    the entries in the report.
- **Done when:** the default synthetic and dev scorecards include gappy stars, the
  per-coverage-class metrics appear in `metrics.json` and the figures, and the new
  baselines are committed.

### T12. Efficiency pass on the production pipeline
- **Agent:** `performance-optimizer`
- **Status:** ready (synthetic is enough; re-check on dev_small once T2 lands)
- **Why:** at ~1.16 s/star on the default path (`ss_local`, 2026-10-13), one pass over
  1M stars is ~322 CPU-hours (~32 h on 10 workers), and the corpus will be
  re-run many times. Each 10% speedup saves ~32 CPU-hours per pass.
- **Goal:** profile the production path (load → ACF → default guesses → comb fit →
  features), find where time goes (including slow-tail stars), and make it
  faster **without changing results**.
- **Done when:**
  - `benchmarks/profile_pipeline.py` exists;
  - an output-equivalence test exists in `tests/`;
  - optimizations are implemented with before/after timings and no regression
    guard trips;
  - the report includes projected 1M-star CPU-hours and a ranked list of
    further opportunities;
  - any accuracy-for-speed trades are proposed separately for Rae.
- **Coordination:** touches the same files as Phase 2 method work (`guesses.py`,
  `comb_fit.py`). Don't run it concurrently with a method-improver on the same
  module. Rerun after major method changes land, since new methods add cost.
- **Follow-up:** once the profiler exists, Rae may add a runtime goal on its
  output to `goals.json`.

---

## Phase 3: features

### T9. Feature literature and empirical search
- **Agent:** `feature-researcher`
- **Status:** done, merged 2026-10-14 (`reports/2026-10-14_t9-feature-search/`). Rae answered its open questions 2026-10-14 (see P1, T10, T11, T17, P8).
- **Goal:** find features that separate correct candidates from P/2 and 2P
  aliases.
- **Done when:** report with a literature table, per-feature separation (AUC)
  by period bin, and any new features implemented, documented and tested.

---

## Phase 4: ranking model

### T10. Feature-table generation at scale
- **Agent:** `ranking-model-builder`
- **Status:** blocked (on T2)
- **Goal:** resumable parallel script producing candidate feature tables for
  dev_medium plus a training sample. Rae runs it locally.
- **Decision needed first:** `extract_candidate_features` labels a candidate 1
  only if its fitted P is within 3% of P_eq. Under differential rotation, a
  correct spot-latitude period gets label 0. Consider a graded relevance label
  for the ranker (2 = P_eq, 1 = in active-band range per
  `evaluation.active_band_period_range`, 0 = otherwise). XGBoost ranking supports this.
- **Requirement (Rae, 2026-10-14, from T9):** check that T9's alias features
  (`fold_half_corr`, `acf_half_lag_contrast`, …) don't penalize spot-latitude
  (`diffrot`) periods relative to P_eq. Compare their distributions for diffrot vs
  correct candidates (`reports/2026-10-14_t9-feature-search/data/candidates.csv.gz`)
  before settling the label scheme.

### T11. Train and evaluate the ranker
- **Agent:** `ranking-model-builder`
- **Status:** blocked (on T10)
- **Goal:** XGBoost ranker, star-grouped splits, beats hand-tuned arbitration
  per period bin, with a calibrated "decline" threshold.
- **External benchmark (Rae, 2026-10-14, from T9):** where SMARTS CNN predictions
  are available (Claytor+2022 σ_P, the only published *learned* reliability score
  for SMARTS-like data), compare the ranker's decline threshold and
  coverage-vs-accuracy curve against the CNN's uncertainty cut.

---

## Future features (Rae-approved direction, not yet scheduled)

### F1. Detect and measure differential rotation
- **Requested by:** Rae, 2026-10-13. Tackle separately from period recovery.
- **Idea:** identify which stars show latitudinal differential rotation and
  estimate how much (e.g. shear, or the spread of spot-latitude periods). SMARTS
  gives exact truth for training and validation (`DIFFROT`, `MINLAT`, `MAXLAT`;
  see `docs/SMARTS_REFERENCE.md`).
- **Possible signals:** period drift across sectors (sliding-window ACF/LS),
  split or broadened periodogram peaks, wavelet ridge wander, comb-fit residual
  structure or per-peak jitter (`allow_jitter` in `comb_fit.py`), and
  multiple non-harmonic candidates clustered within ~0.5–2x P that all fit well.
- **Note:** Claytor+2022's CNN could *not* recover differential rotation from
  binned wavelet spectra, so a positive result would be novel.
- **Prerequisite:** T1/T2 (real data + dev sets). The scorecard's `diffrot`
  outcome already tags stars where we measured a non-equatorial spot period.

### F2. Re-validate on the Apple Silicon workstation (when Rae switches)
- **Trigger:** Rae moves back to the 14-core Apple Silicon machine.
- **Do:** build the env from `environment.yml`, run `pytest`, re-check scorecard
  determinism (T14), re-time the pipeline (T12 profiler) and choose `--n-workers`
  for 10 P-cores. Update CLAUDE.md's Hardware section.

## Open discussions (Rae to decide; not ready for an agent)

### D1. Report the searched parameter range ("no periodicity below X d")
- **Raised:** 2026-10-14 (from T16d). Rae likes it: it tells an astronomer what
  range of period space the data and pipeline effectively searched.
- **Leaning:** a reported output (e.g. a per-star `searched_period_range` plus
  "no significant periodicity in [a, b] d"), **not** part of scoring.
- **To decide:** how to define the range (sampling/Nyquist, longest contiguous span,
  ACF valid-pair coverage, T16c's `max_period(coverage)`), what "no periodicity"
  threshold means statistically, and the output format.

- **Related:** Astraea (Lu+2020) shows what can be inferred about periods beyond the
  searched range from short light curves plus stellar parameters.

### D2. Combining widely separated sectors in real data
- **Raised:** 2026-10-14. SMARTS spans one year, so the pipeline has never seen
  multi-year gaps (e.g. a star observed in Cycle 1 and again in Cycle 3+).
- **To decide, before running on real TESS data:**
  - analyze each observing season separately and combine the results, or run on
    the stitched light curve (spots evolve, so phase coherence across years is lost);
  - how to treat inter-season normalization offsets;
  - how much the window function's clumping (see the window_function figures and
    PROJECT_CONTEXT changelog 2026-09-01) affects long periods;
  - whether to simulate it by stitching separate SMARTS light curves, or by
    generating multi-year synthetic data.
- **Relates to:** open issue 6 (multi-sector support), T16, F2.

## Proposed (agents add here; Rae promotes)

### P1. Quick arbitration improvement before the ML ranker
- **Promoted to T18** (2026-10-14).
- **Proposed by:** Phase 0 report (2026-10-13)
- **Idea:** in 10–30 d, Stage-1 recall is 0.95 but final correctness is 0.65.
  Try simple, explainable changes to candidate selection in `fit_rotation_period`
  (e.g. weighting cross-method support `n_contributing_methods`, penalizing
  candidates whose multiples also fit well) and measure them against G4.
- **Also try (Rae, 2026-10-14):** T9's alias-feature gate (P7): select by or gate on
  `acf_half_lag_contrast`. In-sample it gave 71.5% success vs 67% at the same answer
  rate, and may rescue >30 d stars. Its threshold was tuned on seed 0, so **validate
  on a different synthetic seed** (e.g. `--seed 1`) before adopting it, and later on
  dev_small.
- **Also in scope (Rae, 2026-10-14, from the ACF long-period report):**
  - **Few-peak noise guard (was P10).** Red noise gives a high false-positive rate:
    22/60 red-noise light curves got a period in `benchmarks/pb_scan.py`. Test the
    raw ACF peak height, PDM theta and T9's `acf_half_lag_contrast` as gates, and
    judge them on red-noise FP *and* success.
  - **Factor-of-2 guard threshold (0.7, hand-set).** A clean two-spot star with
    ratio 0.81 still goes to P/2. Test alternatives with T9's `fold_half_corr`, the
    best P-vs-2P separator.
  - Measure on synth200 (two seeds) **and** `pb_scan.py`.


### P2. Other guess_* methods should return [] instead of raising
- **Promoted to T15** (2026-10-13).
- **Proposed by:** T4 report (2026-10-13)
- **Idea:** `guess_lombscargle`, `guess_acf_fft` and `guess_wavelet` still raise
  `RuntimeError` on degenerate input (too few points, empty period range, no
  peak above SNR). The contract in `tests/test_guesses.py` says returning `[]`
  is fine but raising is not. Apply the T4 pattern, with a test for each exit.

### P3. Recalibrate guess_pairwise_histogram's adaptive ACF-peak prominence
- **Promoted to T13** (2026-10-13).
- **Proposed by:** T4 report (2026-10-13)
- **Idea:** the threshold, 5 x std(second difference of the ACF), misses visible
  rotation bumps when spot lifetimes are short. In other stars it lets through
  hundreds of noise peaks (median 15, max 585, on synthetic n=200). Try
  smoothing the ACF before taking the second difference, an envelope-relative
  prominence, or a retry at lower prominence when fewer than 2 peaks are found.
  Measure the change with pairwise_histogram recall by bin. See
  `reports/2026-10-13_t4-pairwise-histogram-no-raise/` (Figure 2).

### P4. Explain the last guess_wavelet run-to-run residue
- **Proposed by:** T14 report (2026-10-14)
- **Idea:** under `ss_openblas`, final periods and every other method's candidates are
  bit-identical across runs and worker counts, but `guess_wavelet`'s top-1 differed by
  ~2e-11 (relative) on one star: synthetic n=200 seed 0, star 00058, two distinct values
  across 7 runs (2 of the 3 four-worker runs vs everything else); also star 00038 once
  on n=40. It doesn't reproduce
  in-process, under heap perturbation, or after replaying full scorecard history.
  wavelet isn't pooled into the fit, so final answers are unaffected, but T12's exact
  per-method equivalence check would trip on it. Suspects: per-process state in
  `scipy.fft` (pocketfft plan cache) or the bounded `curve_fit` in
  `_fit_and_subtract_gaussian`. Low priority; until it's explained, T12 can compare
  wavelet candidates with rtol=1e-9.

### P5. Re-verify determinism and the default worker count on Apple Silicon
- **Proposed by:** T14 report (2026-10-14)
- **Idea:** T14's fix was verified only on the Intel i5. numpy/scipy's macOS arm64 PyPI
  wheels use Accelerate, so on an Apple Silicon machine build `ss_openblas` from
  conda-forge with `libblas=*=*openblas` (mamba/micromamba; see `environment.yml`).
  Then run `pytest tests/test_determinism.py` (must PASS, not XFAIL), two 200-star
  scorecards + `tools/compare_runs.py`, and the worker-count timing sweep from the T14 report.
  Overlaps with F2 if Rae has filed it there.

### P6. Recalibrate guess_wavelet's GWPS peak-SNR threshold
- **Proposed by:** T15 report (2026-10-14)
- **Idea:** `guess_wavelet` keeps a Gaussian peak only if it exceeds
  median + `min_peak_snr`·1.4826·MAD of the GWPS. The GWPS of a typical star
  rises toward long periods (red noise), which inflates the median/MAD, so clean,
  isolated true-period peaks fall below the threshold (synthetic stars 00055 and
  00131; T15 report Figure 3). Try a noise estimate relative to a smoothed or
  power-law continuum, or SNR in log power. The wavelet counterpart of T13.
  Measure with wavelet recall by bin.

### P7. Alias-feature gate for arbitration (interim fix for open issues 10/12)
- **Folded into P1** (Rae, 2026-10-14).
- **Proposed by:** T9 report (2026-10-14)
- **Idea:** 30 of 36 stars lost in arbitration (synthetic n=200) lose because the
  success candidate fails the gates. Most wrong answers are ~0.2 d noise combs on
  the lag-0 decay with `acf_half_lag_contrast` ~ 0. Try selecting by, or gating on,
  `acf_half_lag_contrast` (in-sample: "max contrast, answer if >= 0.03" gives 71.5%
  success vs 67% at the same answer rate). Also check why >30 d stars are 0/8.
  Measure against G4/G7 with a before/after scorecard; validate on dev_small.

### P8. Re-test star-level SNR and relative peak widths on SMARTS
- **Rae, 2026-10-14:** also re-validate T9's synthetic findings (AUC table, the P1
  gate) on dev_small once T2 lands.
- **Proposed by:** T9 report (2026-10-14)
- **Idea:** `rvar_p2p_snr` (r_var / point-to-point noise; code in the T9 report's
  `new_features.py`) can't be evaluated on synthetic data (constant white noise).
  Also test `*_nearest_peak_width / fitted_P` (current widths are in days) and
  `s_ph` with 5·P bins. Needs dev_small (T2).

### P9. Make the P/B scan a standard benchmark with its own goal
- **Proposed by:** ACF long-period report (2026-10-13)
- **Idea:** `benchmarks/pb_scan.py` (P set as a fraction of the baseline, 1/2/13 sectors, plus 120
  noise LCs) probes the short-baseline regime that synth200 (13 sectors) barely touches. Candidate
  goal: correct >= 0.8 at P/B 0.10-0.30 on 1-2 sectors, with white/red noise FP not above baseline.

### P10. Noise guard for few-peak answers
- **Folded into P1** (Rae, 2026-10-14).
- **Proposed by:** ACF long-period report (2026-10-13)
- **Idea:** red noise occasionally yields long `few_peaks` false positives (1/60 eval, 3/30 dev seeds). Raw
  ACF peak height and PDM theta separated them cleanly from true few-peak detections on the
  handful of cases seen. Test a gate on a larger noise set, or leave it to the ranker.

### P11. Recover the runtime the long-period change cost
- **Folded into T12** (2026-10-14): T12 should profile the post-merge pipeline and treat this as its first target.
- **Proposed by:** ACF long-period report (2026-10-13)
- **Idea:** median per-star time rose 1.6x on 1-2 sectors and 1.3x on synth200 (more teeth and candidates per
  fit; profile in the report). Try capping teeth per candidate on the long ACF, skipping the long-view
  Stage-1 pass when a near-identical candidate already exists, or a cheaper first-pass fit. Verify no
  outcome change on pb_scan + synth200. Good fit for the `performance-optimizer` agent.

### P12. Mis-phased comb fits of slightly-off candidates
- **Proposed by:** T18 report (2026-10-14)
- **Idea:** synth-s0-00144 (P = 0.531 d, SNR 14) has clean ACF peaks, but its only near-true candidate
  (P0 = 0.502, 5% off) is fitted with teeth walking off the peaks and fails the vertex gate (T18 report,
  example 5). Try refining P0 (e.g. from the ACF peak spacing or a local comb_score scan) before the
  t0 grid search, and measure how many vertex-gate declines it rescues without admitting noise combs.

### P13. TESS orbit-period (13.7 d) false answers
- **Proposed by:** T18 report (2026-10-14)
- **Idea:** after T18, 2 of the 3 remaining wrong answers on synth200 seed 1 are ~13.6-13.7 d for
  P < 1 d stars: an orbit-gap / ramp periodicity that passes every gate. Relates to T8 / open issue 7.
  Consider flagging or penalizing candidates within a few % of the orbital period (or its multiples)
  when the light curve has orbit ramps.

### P14. Stage 1 recall for short-period stars with no success candidate
- **Proposed by:** T18 report (2026-10-14)
- **Idea:** 28 of the 55 lost seed-0 stars have no fitted success candidate at all (24 at P < 1.6 d,
  SNR 1-18), so no arbitration change can recover them. Feed into T5 (short-period recovery).
