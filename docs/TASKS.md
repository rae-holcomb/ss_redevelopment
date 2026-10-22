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
- **Status:** phase 1 in review (`reports/2026-10-15_t1-smarts-catalog-audit/`, branch `claude/t1-data-audit-phase1-41159b`): catalog + 50-file sample + 1,050-file noise/leakage check. Phase 2 = full-corpus scan (bundle index, corrupt-noise list, per-file galaxy IDs), if Rae wants it; see P15.
- **Goal:** know what's in the ~1M-file corpus and what's wrong with it before
  training on it.
- **Done when:** report covers inventory, label sanity, recoverability
  (amplitude vs background noise), gap/baseline coverage, distributions and
  biases, duplicates/blends. checks for patterns in sectors with spuriously 
  large noise. Includes a CSV of files recommended for exclusion
  (corrupt) or flagging (hard-but-valid), each with a reason.

### T2. Build frozen dev sets and holdout
- **Agent:** `dev-set-builder`
- **Status:** done, merged 2026-10-21 (report `reports/2026-10-15_t2-dev-sets/`).
- **Rae's decisions on T2's open questions (2026-10-21, all as recommended):**
  1. Keep `native_whole_sectors` (both orbits usable) for coverage assignment, so the assigned pattern is exactly what loads.
  2. Loader crash on an all-flagged requested pattern: raise a clear, named error that callers record as a decline (task T21).
  3. Real-data goals: draft G9–G12 on dev_medium full-year added to `goals.json`; synthetic G1–G8 stay as regression checks.
  4. **Default judging set for method changes: dev_medium full-year**
     (`--manifest datasets/dev_medium.csv --data-root data/smarts_dev --ignore-manifest-sectors`), with Rae's
     standing OK to run it. Use assigned coverage (drop `--ignore-manifest-sectors`) for coverage-robustness work (T16b–e).
  5. dev_small's thin >60 d bins are fine for a smoke test.
  6. No holdout_v2 from the <800k pool (it would break galaxy separation).
  7. Keep the 774 MB build cache on the drive (`extracted/t2_build/`).
  `datasets/dev_small.csv` (50), `dev_medium.csv` (1,000), `holdout.csv` (3,000) from
  `tools/build_dev_sets.py` (seed 20261015); baselines `benchmarks/baselines/dev_{small,medium}_{assigned,fullyear}`.
  Dev sets score with `--data-root data/smarts_dev`; `--ignore-manifest-sectors` gives full-year.
  Deviation: coverage patterns come from `native_whole_sectors` (both orbits unflagged), since an
  explicit `sectors=` request drops half-flagged sectors. Headline: dev_medium full-year success
  0.248 (wrong 0.088, declined 0.650) vs synth200 0.730.
- **Note:** SMARTS periods are uniform in 0.1–180 d, so only ~17% of the corpus
  has P < 30 d (the 1-yr-baseline limit) and ~5.5% has P < 10 d. Random sampling
  would make dev sets mostly unrecoverable long periods. Stratify, and use the
  MAST input-parameter catalog rather than opening FITS headers.
- **Note (2026-10-15, T1 phase 1):** file number = catalog **row index**, not "Simulation
  Number". SIM_ID >= 800,000 uses a disjoint galaxy-noise pool: draw holdout (and dev) from it,
  train on SIM_ID < 800k. ~12% of galaxies have corrupt orbits. Plan, strata and sizes in
  `reports/2026-10-15_t1-smarts-catalog-audit/` ("Recommendations for T2").
- **Rae's decisions (2026-10-15, on the T1 phase 1 open questions):**
  - Dev and holdout may share galaxies (both from SIM_ID >= 800k); no galaxy IDs needed for T2.
  - **Keep "band disjoint" stars** (whole active band outside P_eq +/-10%, ~15%) in every set,
    including ranker training, and **track them**: every manifest gets `band_disjoint`,
    `band_extends`, `P_lo`, `P_hi` columns (from `evaluation.active_band_period_range`), so they
    can be analysed as their own case (see P17). Don't stratify them away.
  - **Exclude files with corrupt galaxy noise** (raw max |flux - 1| > 0.1) from the dev sets and
    holdout. Costs ~12% of the pool (>= 800k pool ~176k left; smallest stratum 73 -> ~64), so
    diversity is fine. Practical route without a full scan: check each drawn file at extraction
    (`tools/audit/flux_outliers.scan`) and redraw replacements from the same stratum; log the
    excluded rows.
  - **Native usable sectors (decided 2026-10-15):** while extracting, run `load_smarts_fits`
    once per file and store its kept sectors (`native_sectors`, e.g. "1 2 3 5 ... 13") in the
    manifest. T16f coverage patterns are then chosen only from each file's native sectors, so the
    assigned pattern is exactly what the pipeline sees.
- **Note (2026-10-14):** dev sets must also stratify on **sector-coverage pattern**,
  not just full-year light curves. See T16f.
- **Rae's design decisions (2026-10-15, overnight run):**
  1. Sizes: dev_small **50**, dev_medium **1,000** (both local under `data/`), holdout **3,000** (drive only).
  2. Period mix: **equal per bin** for <1, 1–3, 3–10, 10–30, 30–60 d; fewer for 60–120 and 120–180 d
     (~1/7 of each set, as "should decline" tests). Record per-row **sampling weights** for reweighting
     to the corpus.
  3. Other strata: amplitude-proxy tercile × shear class (solar / zero / anti-solar).
  4. **T16f's real-data part is folded in:** each row gets a coverage pattern from the pooled
     all-cycles library (`datasets/coverage/all_cycles_patterns.csv`), drawn only from its
     `native_sectors`, stored in a per-row `sectors` column (the scorecard already reads it), plus
     `coverage_class`.
  5. Coverage mix: **balanced across coverage classes** (diagnostic power), with sampling weights
     so results can be reweighted to the real-world mix.
  6. Add a scorecard switch to **ignore the manifest's `sectors`** (full-year scoring of the same
     stars), for paired coverage comparisons, and make manifest `path`s resolve relative to
     `SMARTS_DATA_ROOT` (or `--data-root`). These are dataset options; metric definitions stay untouched.
  7. Baselines: score **dev_small and dev_medium** (both with assigned coverage *and* full-year) and
     commit them to `benchmarks/baselines/`. Rae's standing OK for the dev_medium runs. **Never run
     the holdout.**
- **Goal:** `datasets/dev_small.csv` (50), `dev_medium.csv` (1,000),
  `holdout.csv` (3,000), stratified as above, reproducible
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
- **Status:** done, merged 2026-10-15 (report `reports/2026-10-14_t18-arbitration-gates/`).
- **Decision (Rae, 2026-10-15):** keep the **strict** gate (default). Low-to-medium confidence,
  since it was chosen on synthetic data only. **Re-evaluate strict vs option D**
  (`half_lag_fallback_min=0.0`) on SMARTS dev_small (P8). Its other open questions
  (restating G7, factor-2 ratio 0.7 vs 0.8, training data with `half_lag_gate=False`) are
  still pending Rae.
- **Caveat found 2026-10-15 (T16a demo rerun):** the strict gates were tuned on *full-year* data, and on
  single-sector coverage they decline many measurable short-period stars (see the T16b priority finding).
  Re-evaluate strict vs option D **per coverage class**, not just on dev_small overall.
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
- **Scope update (Rae, 2026-10-15):** the pipeline will eventually run on the **whole TESS extended
  mission**, so T16 covers all cycles, not just Cycle 1. SMARTS files are single-season (one year on the
  Cycle-1 27.4 d sector grid), so: SMARTS-based work uses **single-season** patterns drawn from every
  cycle's library; **multi-season** behavior needs other simulations (T16g); patterns should be defined in
  **days**, not sector indices, once Year 8+ (~55 d pointings) is added.
- **Coordination:** the "Extend ACF period limit" session (branch
  `claude/angry-einstein-cb10da`) is changing the ACF lag / period limit. T16c must
  build on whatever lands from it.

#### T16a. Coverage-degradation infrastructure
- **Agent:** `method-improver` (infrastructure only)
- **Status:** done, merged 2026-10-15 (report `reports/2026-10-14_t16a-coverage-infrastructure/`). Libraries for TESS Years 1–7 plus the pooled default.
- **Rae's decisions (2026-10-15):**
  - `per_star.csv` columns approved;
  - 3-5 / 6-12 split into consecutive/gapped;
  - libraries extended to TESS Years 1–7, with `all_cycles_patterns.csv` as the default;
  - within-class sampling first uniform, then **frequency-weighted** (now the default) once Years 1–7 were pooled.
- **Settled (Rae, 2026-10-15):** keep truncating the 635 over-long Year-4–6 star-seasons (Years 4–6 are 376–434 d
  long with normal ~27-d sectors), and keep pooling years in proportion to their star counts. No open questions remain.
- **Deferred (Rae, 2026-10-15):** Year 8+ (S97+, irregular ~13–58 d pointings). It needs a tess-point rerun and
  a days-based, not sector-index, mapping.
- **Headline for T16b:** before T18, 36% of single-sector synthetic stars got confident ~0.2 d wrong answers.
  The post-T18 rerun (2026-10-15) has 0 wrong on one sector, but 70% declined, including 17/34 stars under 1 d
  and 27/36 at SNR > 5. See the T16a report.
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
- **Priority finding (T16a, 2026-10-15): T18's gates are too strict on short coverage.**
  - On one sector, T18 turned wrong answers into declines: wrong went 0.36 → 0.00, but 70% are now declined.
  - Of 73 full-year successes among 100 stars, single-sector coverage declines 43, including
    5 at <1 d, 6 at 1–3 d and 8 at 3–10 d. Those short-period losses shouldn't be coverage-limited.
  - The 10–30 d and >30 d losses are expected.
  - The pipeline message doesn't say which gate fired. **First step:** instrument and identify the
    responsible gate(s) per declined star (half-lag consistency, few-peak height, n_peaks, vertex-in-window,
    height SNR), then propose coverage-aware thresholds (e.g. scaled by teeth available or cycles
    observed) for T16d, without bringing back the ~0.2 d noise answers T18 removed.
  - Also check: the comb_fit `decline reason` should name the failing gates (useful for D1 and debugging).
- **Update 2026-10-15:** use the pooled all-cycles library (single-season patterns from every TESS year),
  report by coverage class *and* by year/hemisphere, and include T18's gates (check the ~0.2 d single-sector
  wrong answers T16a found).
- **Also (Rae, 2026-10-14):** re-evaluate the comb fit's sqrt(pair-count) residual
  weighting (measured neutral on full-year data in the ACF long-period report). It
  should matter most for gappy coverage. Keep or remove it on that evidence.
- **Agent:** `method-improver` (diagnostic report; no method changes)
- **Status:** done, merged 2026-10-21 (report `reports/2026-10-21_t16b-gate-diagnosis/`): gate instrumentation + diagnosis.
  The per-method half is folded into T16c (below).
- **Rae's decisions on T16b's open questions (2026-10-21, all as recommended):**
  1. **No gate relaxation before T20 reports.** Afterwards, rerun T16b's offline trade-off replay (section 5;
     `make_data.py` + notebook) excluding the periodic noise galaxies T20 flags, then decide in T16d.
     **Unblocked (2026-10-22):** T20 phase 2 is done. Exclude stars whose noise source has
     `exclude_from_training` (or, stricter, `periodic`) in `<root>/extracted/galaxy_ids/noise_source_flags.csv.gz`
     (join on catalog row), and feed the result to T16d.
  2. Wide diffrot bands inflate success. **Report `correct` alongside `success` by band-width class** as an
     *added* metric (success definition unchanged); see T22. Capping the band waits for T20/D4.
  3. Height-SNR denominator from well-sampled lags only → **T16d**. Keep `mean_height_snr` unchanged and add the
     new version as a parallel ML feature.
  4. Per-method degradation mechanisms (ACF truncation, pairwise_histogram peaks, LS window aliasing, wavelet
     across gaps) → **folded into T16c**.
  5. Year/hemisphere breakdown: **deferred**; add the year tag to scorecard draws only if T16c finds year sensitivity.
  6. **Bump `SCORECARD_VERSION` to 4** for T16b's new columns / `per_candidate.csv` (T22). Rule from now on: bump
     whenever output columns or files change.
  7. sqrt(pair-count) weighting re-evaluation on gappy data → **T16d**.
- **Do:**
  - Start from the T16a demo, **rerun post-T18 (2026-10-15)**. On the same 100 stars, patchy coverage now
    declines rather than answers wrong: none is 0.70 single / 0.56 consecutive-2 / 0.68 gapped-2 /
    0.58 gapped-3-5 (full 0.27), with wrong ≤ 0.03. Single-sector declines include half the <1 d stars,
    where the data hold dozens of cycles. Find which gate fires on short data, and whether it can be
    recalibrated without bringing back the pre-T18 ~0.2 d false positives (36% wrong on one sector
    before T18).
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
- **D4 decided (2026-10-22); this replaces the open-ended "define max_period(coverage)" below.** Do:
  - **`searchable_range(time, flux)`** in `coverage.py`, from the time sampling only:
    - returns P_min (cadence-based), the ACF reach R and P_max = R/2;
    - works per contiguous batch, so T16h can call it per batch;
    - one implementation, reused by the pipeline output (D1) and as the reference for T23's frozen evaluation version.
  - **Stage 1 stays uncapped.**
    - Tag candidates with P > R/2 as `beyond_reach`. The comb fit skips them at entry; they stay in the candidate record.
    - Measure runtime before/after; expect a gain on 1–2 sectors.
    - Results must be bit-identical except where the fit used to try a beyond-reach candidate.
  - **Record the range in `EnsembleResult`**, plus a P/R ML feature (with docs).
  - **Validate R/2 per coverage class** (option E): success and wrong vs P/R, including on dev_medium.
  - **Test `min_valid_frac` 0.25** (below).
  - **Per-method mechanisms** (below).
- **Now also includes (from T16b, Rae 2026-10-21):** the per-method degradation mechanisms by coverage (ACF
  truncation by `min_valid_frac`, pairwise_histogram running out of peaks, LS window aliasing, wavelet across gaps).
  **Waits for the D4 decision.**
- **Update 2026-10-15:** define limits **per season** (from season length and coverage). Separately, ask
  whether combining seasons (D2) can ever justify periods longer than one season. Probably weakly at best.
- **Also (Rae, 2026-10-14):** test lowering `compute_acf`'s `min_valid_frac` from 0.3
  to ~0.25. At P = B/3 on 2 sectors only 6/12 are correct, because P sits on the
  ACF pair-coverage edge (ACF long-period report, open issue 14a).
- **Agent:** `method-improver`
- **Status:** ready (T16b done; D4 decided 2026-10-22)
- **Do:** define the maximum searchable / reportable period as a function of
  coverage (longest contiguous span, total baseline, ACF valid-pair coverage).
  Measure reliability (success vs wrong) as P approaches each candidate limit, and
  pick limits that keep "wrong" low.
- **Done when:** a `max_period(coverage)` rule implemented in one place, with
  justification plots, and goals proposed for Rae.

#### T16d. Method policy and safeguards for patchy data
- **Now also includes (from T16b, Rae 2026-10-21):** (a) height-SNR noise from well-sampled lags only (new parallel
  ML feature; old `mean_height_snr` unchanged); (b) re-evaluate sqrt(pair-count) residual weighting on gappy data;
  (c) gate relaxations only after T20, via T16b's replay excluding flagged noise galaxies. T16b's do-nots stand:
  no option D on short coverage, don't relax the half-lag gate, and contrast-conditional relaxations need ≥ 4 teeth.
- **Update 2026-10-22 (D4):** cross-batch consistency and segment-wise analysis move to **T16h**. T16d keeps the
  gates, method selection and safeguards.
- **Update 2026-10-15:** with D2 leaning season-by-season, **cross-season consistency** becomes the central
  safeguard: answer when seasons agree, flag when they disagree (possibly a spot-evolution or diffrot signal; F1).
  Needs T16g's multi-season data. Status: blocked on T16b/c **and T16g**.
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
- **Decision (Rae, 2026-10-15):** **defer the multi-season features** (n_seasons, per-season agreement/scatter) until T16g/D2. Single-season coverage features only for now.
- **Update 2026-10-15:** add n_seasons, per-season period agreement and per-season scatter alongside the
  single-season coverage descriptors. **Unblocked by T16a (merged 2026-10-15).**
- **Agent:** `feature-researcher`
- **Status:** in review (`reports/2026-10-15_t16e-coverage-features/`, branch `claude/t16e-coverage-features-450578`): 7 single-season coverage features (FEATURE_DOCUMENTATION Section H); mostly null on synthetic data, but several existing features change meaning with coverage (see report).
- **Do:** add coverage descriptors (n_sectors, longest span, duty cycle, P/longest
  span, n_cycles observed) to `ml_features.py`, so the ranker can learn how much to
  trust a candidate under a given coverage. Relates to open issue 6 (multi-sector
  support).

#### T16f. Gappy coverage in the standard development datasets
- **Update 2026-10-15:** SMARTS dev sets stay single-season, sampling from the pooled all-cycles library
  constrained by each file's `native_sectors`. Add a small **multi-season** dev set from T16g. Synthetic part
  **unblocked by T16a**.
- **Agent:** `dev-set-builder` (for the real-data sets), `method-improver` (for synthetic)
- **Status:** real-data part folded into T2 (2026-10-15); synthetic part ready.
- **Why:** so every agent's routine before/after scorecard exercises patchy data,
  not just the clean full-year case. Regressions on gappy stars should be caught
  as they happen, not discovered at the end.
- **Do:**
  - **Synthetic:** define a standard coverage mix for the default synthetic
    benchmark (a fixed fraction of full-year, consecutive, non-consecutive and
    single-sector stars), using the T16a coverage library. Re-baseline
    `benchmarks/baselines/` once, recording the old baseline as history, as with v2.
  - **Real data (T2):** pick each file's coverage pattern only from its native usable sectors
    (manifest column `native_sectors`, decided 2026-10-15; see T2). Add coverage pattern as a stratification axis for
    `dev_small`, `dev_medium` and `holdout`. Apply patterns from the T16a library to
    SMARTS files via `load_smarts_fits(sectors=...)`, stored as a per-row column
    in the manifest (and read by the scorecard), so each set is frozen and reproducible.
  - **Scorecard/goals:** report metrics per coverage class as well as per period bin,
    and add a regression guard per coverage class. Rae owns `goals.json`, so propose
    the entries in the report.
- **Done when:** the default synthetic and dev scorecards include gappy stars, the
  per-coverage-class metrics appear in `metrics.json` and the figures, and the new
  baselines are committed.

#### T16g. Multi-season simulations (for D2 and T16d)
- **Agent:** `method-improver` (or a data/sim agent)
- **Status:** not started. **Discuss with Rae before starting**, since the approach may have evolved (see D3).
- **Plan (Rae, 2026-10-15, tentative):** use **`butterpy`**, the generator behind SMARTS, to simulate
  multi-year light curves with spot evolution and activity cycles, sampled with real multi-season TESS
  coverage from the per-cycle libraries. It's a new package install, so confirm with Rae when it starts. Fallback:
  extend `synthetic.py` to multi-year baselines.
- **Needed by:** D2 experiments (season-by-season vs stitched) and T16d/T16f.

#### T16h. Batch-wise search and combination across sector clusters (from D4, 2026-10-22)
- **Agent:** `method-improver`
- **Status:** blocked on T16c (`searchable_range`). Phase 1 can use SMARTS now; multi-year tests need T16g.
- **Idea (Rae):** split the whole-mission light curve into **batches** of adjacent or near-adjacent sectors. Search each
  batch separately, then compare the batches. This generalizes D2's season-by-season approach to clusters within a year.
- **Do:**
  - **Batch definition:**
    - group sectors whose gaps are < G. Calibrate G; the data-driven variant merges across a gap only when that
      increases the ACF reach R.
    - Each batch gets its own `searchable_range`: [P_min, R_b/2].
  - **Testability-aware combination.** Batch b can confirm or contradict a period P only if P ≤ R_b/2.
    - **Testable, b finds ≈ P:** agreement.
    - **Testable, b finds ≈ P/2:** alias suspect.
    - **Testable, b finds something else:** conflict.
    - **Testable, b declines:** weak tension only (amplitude varies between seasons).
    - **Not testable, b finds ≈ P/2:** consistent with P (expected for a short batch).
    - **Not testable, any other result:** silent.
    - **Star-level reliability** = the number of batches that could test P and agreed.
    - A long P seen only by the one batch that can reach it is "unconfirmable by design": flagged, not penalized.
    - Differences within the active-latitude band are physical (F1), not conflicts.
    - The star's searchable range is the union of the batch ranges. `searchable` scoring (T23) uses the same union.
  - **Alternative to compare: the stacked ACF.**
    - Run `compute_acf` on the stitched light curve, counting only pairs within the same batch (per-batch mean removal
      and normalization). That equals the pair-count-weighted average of the per-batch ACFs.
    - Fit one comb to it: short periods gain signal from every batch, and long lags come from whichever batch reaches them.
    - Candidate design: the stacked ACF gives the **answer**, the per-batch results give the **consistency** check
      (features or flag).
  - **Optional (not required): the ranker combines batches** (T11), using features such as n_batches_testable and
    n_batches_agree.
  - **Optional (not required; Rae: likely too expensive to do much of): a quasi-periodic Gaussian process with a period
    shared across batches** (e.g. celerite's rotation term). At most a validation reference on a small subset.
  - **Phase 1:** SMARTS's within-year gapped classes (gapped-2 / 3-5 / 6-12) are already multi-batch stars, so test there
    first (dev_medium assigned plus synthetic `--coverage gapped-*`).
  - **Phase 2:** multi-year gaps with spot evolution on T16g simulations.
- **Done when:** a batch-combination rule plus a stacked-ACF comparison is in a report, with success and wrong per
  coverage class vs the current stitched-ACF pipeline, and no regression on full or consecutive coverage.

### T19. Repository cleanup and public-release readiness (recurring + final)
- **Agent:** `repo-curator`
- **Requested by:** Rae, 2026-10-15. The repo will be released publicly. Keep it free of
  cruft, and clean it periodically during development, not just at the end.
- **Cadence:** a **hygiene pass (T19a)** at each milestone, e.g. after a batch of task
  merges, after T2, after the ranker lands. The **release pass (T19b)** comes near the
  final pipeline. Avoid running a pass while other agents edit the same modules.

#### T19a. Hygiene pass (repeatable)
- **Status:** ready. A first pass is worthwhile once T18/T16a/T1 are merged.
- **Do:**
  - remove dead code and duplication;
  - fix docstring/doc drift;
  - condense finished tasks in TASKS.md into changelog pointers;
  - untrack artifacts that shouldn't be in git;
  - produce a removal-proposal table for anything human-made, which Rae approves.
  Results must stay bit-identical vs `benchmarks/baselines/synth200_s0_tables/`.
- **Known candidates (inventory 2026-10-15):**
  - **`results_products/batch1/candidates.csv` (90 MB) and other result CSVs are
    tracked**, despite CLAUDE.md saying they're gitignored. GitHub rejects files > 100 MB.
    Untrack them (and decide on history, see T19b).
  - `.DS_Store` is tracked; gitignore it.
  - Root-level `split_smarts_lc.py` (used by `preprocessing.py`) belongs in the package.
  - `claude_files/old/` (11 superseded files).
  - `batch_test_acf_fft_highpass.py` may be redundant (open issue 9), and so may the
    older `run_debug_baseline*.py` runners now that the scorecard exists.
  - `sector_configs.txt` is superseded by T16a's coverage library (once merged).
  - Root notebooks, `smarts_lc_gaps1.npz`, `target_df_shortform.csv`, `tic283722336/`:
    Rae's scratch/data. Propose keep / move / archive; Rae decides.

#### T19b. Public-release pass (near the final pipeline)
- **Status:** blocked (until the pipeline is close to final).
- **Do:**
  - Rename the `claude_files/` package to a real one (e.g. `spinspotter/`) with a
    `pyproject.toml`, so it can be installed and imported normally.
  - Public README: what it does, install, quick start, citation (Holcomb+2022 + new
    paper), license check.
  - Decide what of `reports/`, `docs/` and the agent infrastructure (`.claude/`,
    CLAUDE.md, TASKS.md) ships publicly, versus an archive branch.
  - Optionally purge large files from git history (**destructive; Rae's explicit OK**).
  - Scan for absolute local paths, secrets, and personal info.
  - Set up CI to run the test suite on push.
- **Done when:** a fresh clone installs and passes its tests with no local paths and no
  oversized files, and the README gets a new user to a measured period in a few commands.

### T20. Are the SMARTS noise galaxies really quiet? (training-set flaws)
- **Agent:** `data-auditor`
- **Status:** done, merged 2026-10-22 (`reports/2026-10-21_t20-noise-sources-phase2/`; phase 1:
  `reports/2026-10-21_t20-galaxy-pass-phase1/`; branch `claude/t20-galaxy-quietness-c3bb9b`). 206/1,719 noise sources
  (12% of files) are periodic, mostly neighbour-star contamination; 51/88 dev_medium wrong answers equal their noise
  source's own pipeline period. Follow-up (f): independent QLP/TASOC/ELEANOR-LITE light curves confirm 9/12
  testable periodic sources (P < 5 d), 0/4 quiet controls. Full-set FFI check (2026-10-22): known-variable
  contamination 38/60 confirmed, crowded 13/34, unexplained 4/25 = chance rate (quiet null 4/26); exclusions unchanged
  (report open questions 6-8). Flags file on the drive regenerated with `exclude_from_training` + signal metrics.
- **Rae's decisions on phase 2 (2026-10-22; "go with your recommendations"):**
  - **Flag everywhere:** per-file flags (`<root>/extracted/galaxy_ids/noise_source_flags.csv.gz`: classification,
    `periodic`, `exclude_from_training`, `P_ref`, signal-strength metrics) go into every training and evaluation table.
  - **Exclude from ranker training:** `contamination_known_variable`, `contamination_candidate_crowded` and (for now)
    `source_specific_unexplained`: 197 sources, 114,829 files (11.5%). **Keep** `systematic_common_mode` (8).
    The single `misclassified_star` source (299 files) isn't in the list: flagged, kept.
  - **`noise_period` scoring outcome: yes**, as task T24.
  - **Keep the current "periodic" flags**, with signal-strength metrics stored for later tightening (LS power and
    amplitude, A/B agreement, pipeline period, common-mode retention).
  - **Open:** Rae asks Zach about the 242 near-LMC sources.
- **Rae's decisions (2026-10-21):**
  - **Scope: all 1,000,000 SMARTS light curves** (both galaxy pools), so we know what we're working with.
  - **Measure the runtime before the full pass.** Phase 1 builds the script, times it on 2–3 bundles and
    reports a full-corpus estimate, then stops for Rae's go-ahead.
  - **Run locally, not in the cloud:** a cloud session can't see the external drive, and uploading
    89 GB isn't practical. The full pass is resumable per bundle, for overnight / interruptible runs.
  - **Be careful with injected amplitudes in the stacking** (ACTIVITY 0.1–10× solar sets spot amplitude;
    inclination and spot latitudes change the visible fraction). Build the galaxy estimate from
    low-expected-amplitude members, hold out members whose injected signal should dominate the noise,
    verify empirically via residuals (member − galaxy estimate), and require any recovered galaxy
    period to persist across disjoint member subsets.
  - Rae is asking Zach Claytor (SMARTS PI) for the noise-galaxy TIC list; if it arrives, use it in step E.
- **Why (Rae):** real-SMARTS recovery is low, which is expected (red noise), but some SMARTS features
  are suspect. If a "quiet" galaxy light curve actually contains a periodic signal, every SMARTS star
  injected into it (~1,000 each) carries that signal, and the pipeline may "wrongly" report it. We want
  to develop the pipeline knowing the training set's weaknesses, **without over-designing to fit
  its flaws**.
- **Rae's core test:** group SMARTS light curves by noise galaxy and look for the **same wrong period
  recurring across many stars from one galaxy**. That would mean the signal existed before injection.
- **Plan:**
  - **A. Galaxy groups.** No header identifies the galaxy (checked 2026-10-21: `HLSPTARG` is just the
    simulation name; there are no TIC/RA/Dec keys). Use T1's noise-fingerprint method (files sharing a
    galaxy have identical noise) over the corpus, or at least the SIM_ID ≥ 800k pool that the dev
    sets come from, to assign a `galaxy_id` to every file. This is P15, revived; the same pass can
    record the corrupt-noise flags. It's a drive-bound streaming pass; estimate the time first.
  - **B. Recover each galaxy's own light curve** by stacking its member files. The injected spot signals
    have unrelated periods and phases, so they average away over hundreds of members while the shared
    galaxy noise remains. First establish from member pairs whether injection was additive or
    multiplicative (the paper says only "convolved").
  - **C. Search each recovered galaxy light curve for periodicity,** with the pipeline and with
    Lomb-Scargle. List the periodic galaxies: period, amplitude relative to typical injected
    amplitudes, and how many SMARTS files they affect.
  - **D. Rae's test, made direct:** for dev_medium wrong (and diffrot/alias) answers, do they match their
    galaxy's own period? How much of the 8.8% wrong rate is "the pipeline correctly found a real
    pre-existing signal"?
  - **E. Classify each periodic signal by origin:**
    - **Spacecraft systematics:** shared across many galaxies at the same times (the galaxies sit in
      the southern CVZ and were observed simultaneously, so common-mode or PCA analysis across galaxies
      isolates them); TESS timescales (13.7 d orbit, momentum-dump cadence, ~1 d scattered light);
      sector-dependent.
    - **Source-specific:** unique to one galaxy, coherent through the year. That points to a
      contaminated aperture (a nearby variable star) or a misclassified source (e.g. actually a
      star).
- **Decision afterwards (Rae):**
  - systematics → **keep** (real TESS data has them too), possibly with a label category;
  - contamination / misclassification → **exclude** from training, or **include with a flag**.
  The choice depends on what the signals look like. Labels matter too: a pipeline that correctly
  detects a contaminant's period is currently scored "wrong".
- **Relates to:** T3 (realism), P15 (galaxy IDs / corrupt noise), P16 (short-period signal strength),
  T10/T11 (training labels), D3 (other training data).

### T21. `load_smarts_fits` should fail clearly when no requested sector survives
- **Agent:** `method-improver` (small)
- **Status:** done 2026-10-21, awaiting review (branch `claude/t21-loader-error-c98179`; report `reports/2026-10-21_t21-loader-no-usable-sectors/`). `NoUsableSectorsError(ValueError)` raised before regridding; callers already record it as a decline; dev_small assigned + full-year bit-identical.
- **Problem:** `load_smarts_fits(sectors=[...])` raises deep inside `to_even_cadence` when every requested
  sector is flagged. **Do:** raise a named exception (e.g. `NoUsableSectorsError`) early, with a clear message,
  record it as a decline (`none`, with the reason) in the scorecard and batch runners, and add a test.
  Results on the existing dev sets must be unchanged.

### T22. Scorecard v4: band-width breakdown + version bump
- **Agent:** `method-improver` (small; scorecard reporting only)
- **Status:** ready (Rae, 2026-10-21, from T16b questions 2 and 6)
- **Do:**
  - Bump `SCORECARD_VERSION` to 4, covering T16b's descriptor columns and `per_candidate.csv`.
  - Add **metrics** `final.by_band_width.<class>.{...outcome fractions...}`, using classes of
    (P_hi − P_lo)/P_eq (e.g. 0 / < 0.2 / 0.2–0.5 / ≥ 0.5; justify the cuts), so `correct` vs `success` can be seen
    per class. The success definition is unchanged.
  - Update `check_goals` docs and the baselines README (old baselines lack the new keys: they're reported, not
    guarded).
  - Tests. Existing metrics must be bit-identical.

### T23. Scorecard/evaluation: `searchable` flag and searchable-subset metrics (from D4, 2026-10-22)
- **Agent:** `method-improver` (evaluation + scorecard reporting; definitions approved by Rae in D4)
- **Status:** ready. Can share T22's v4 bump if they run close together; otherwise bump to v5.
- **Do:**
  - **`evaluation.py`:**
    - a frozen `acf_reach(time, flux)` (valid-pair fraction 0.3, independent of the pipeline's `compute_acf` default);
    - `is_searchable(period_lo, reach)` = P_lo ≤ R/2, where P_lo is the low end of the success range (P_eq for
      non-diffrot stars). For multi-batch stars (T16h), use the max over batches of R_b.
  - **Scorecard:**
    - per-star `acf_reach`, `P_over_reach`, `searchable`;
    - metrics `final.searchable.*` (and by period bin / coverage class) next to the existing all-star metrics, plus
      `final.unsearchable.{none,wrong,success}` (correct-decline rate, wrong on unsearchable stars, bonus successes).
      The existing metrics stay bit-identical.
  - **Completeness-map figure:** success vs P/R × amplitude (or SNR) × coverage class.
  - **Goals:** propose searchable-subset goals alongside the existing ones (Rae edits `goals.json`); e.g. G7 restated as
    declines among searchable stars.
  - Tests; update the baselines README (old baselines lack the keys).

### T24. Scorecard/evaluation: `noise_period` outcome (from T20, 2026-10-22)
- **Agent:** `method-improver` (evaluation + scorecard reporting; approved by Rae via T20's recommendations)
- **Status:** ready once T20 is merged (needs the noise-source flags). Coordinate the scorecard version bump with
  T22/T23.
- **Why:** 51 of dev_medium's 88 `wrong` answers equal the period the pipeline finds on the star's noise source alone
  (T20 phase 2). The pipeline is right about the data, but the label doesn't contain that signal.
- **Do:**
  - Join each manifest star to its noise source (catalog row → SIM_ID → TIC; flags in
    `<root>/extracted/galaxy_ids/noise_source_flags.csv.gz`). Store the noise source's reference period(s) with the
    truth, e.g. a small committed per-TIC table, so scoring doesn't need the drive.
  - **`evaluation.py`:** a `noise_period` outcome: a finite answer that isn't correct/diffrot/half/double but matches
    (within `REL_TOL`) the noise source's own period (`P_ref`: the pipeline's period on the source stack, else its LS
    peak). Decide with Rae whether halves/doubles of the noise period count. `success` is unchanged.
  - **Scorecard:** report `noise_period` separately from `wrong`; existing metrics bit-identical where possible.
  - Tests; baselines README.

### T25. dev_medium metrics on the quiet-noise subset (from T20, 2026-10-22)
- **Agent:** `method-improver` (scorecard reporting only)
- **Status:** ready once T20 is merged; can share T24's join.
- **Why:** 128 of dev_medium's 1,000 stars sit on periodic noise sources, with a 43% wrong rate vs 3.8% on quiet
  sources. Method changes should be judged on both.
- **Do:** a per-star descriptor column (`noise_class`, `noise_periodic`, `noise_exclude`) and metrics on the quiet
  subset (`final.quiet_noise.*`) next to the full-set metrics, plus a `--noise-subset` option or equivalent. Existing
  metrics stay bit-identical. Propose quiet-subset goals; Rae edits `goals.json`.

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
- **Update 2026-10-22 (D4 decided):** the upper end of the range is **R/2**, where R is the ACF reach. T16c computes it per
  batch (`searchable_range`); for multi-batch stars, see T16h. What "no significant periodicity" means statistically
  is still open.
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
- **Leaning (Rae, 2026-10-15):** strong prior for **season-by-season analysis, then combining**
  (phase coherence across multi-year gaps is unlikely given spot evolution; per-season agreement is a
  reliability signal). Still to be *tested* on T16g's multi-season simulations before it's final.
  Now needed before T16c/T16d, since the pipeline will run on the whole extended mission.

### D3. Training data beyond SMARTS
- **Raised:** 2026-10-15. Rae wants to eventually add other sources of training data, so the ranker and
  validation aren't solely reliant on SMARTS.
- **To discuss:**
  - candidates: fresh butterpy simulations (T16g), SMARTS DR2 (Kepler-based), real stars with
    trusted literature periods as a *validation* set (beware label circularity; see PROJECT_CONTEXT s3),
    other simulators;
  - how to combine sources without the ranker learning source-specific artifacts (e.g. source as a
    grouping/held-out factor);
  - whether dev and holdout sets should span sources.
- **Relates to:** T10/T11 (ranker), T16g, T3 (realism).
- **Recorded option (2026-10-21; Rae: keep as an option, not committed): generate our own "SMARTS-like" light
  curves** from quiet TESS galaxy light curves + butterpy. Purposes: (1) multi-year baselines beyond SMARTS's
  13 sectors (for D2 / T16g); (2) **more short-period training light curves**, since SMARTS is thin there (P uniform
  0.1–180 d, so only 1.6% < 3 d).
  - **Why it's feasible:** of the 8,156 bright stars in `target_df_shortform.csv`, 90% are observed in ≥ 2 TESS
    years and 63% in ≥ 3; CVZ-like stars get 4–5 years (median 41 sectors). SMARTS's noise sources are southern-CVZ
    galaxies with known TIC IDs (Zach's `galaxy_files.csv`; SMARTS_REFERENCE s7), so the *same* sources have
    ~4–5 years of FFIs. T20 vets which are truly quiet: **use only sources with `periodic == False`** in
    `<root>/extracted/galaxy_ids/noise_source_flags.csv.gz` (or `sources.csv` in
    `reports/2026-10-21_t20-noise-sources-phase2/data/`), and prefer sources away from the LMC (`near_lmc`).
  - **Rae's choices:**
    - **Use the newest butterpy** (v1.1.0 as of 2025-10; has activity cycles, butterfly, diffrot, spot evolution,
      any duration, plus temperatures/filters including a Roman filter curve). Best tools over exact equivalence
      with the 2021 version SMARTS used. Its install requirements include `cartopy` (plotting only), so install
      without it if possible; any install needs Rae's OK.
    - **Noise light curves: Option B first.** Use existing FFI products (TESS-SPOC, QLP, GSFC-ELEANOR-LITE, TGLC,
      ...) for the 1,719 noise TICs. Prefer SAP-like products (PDC-style detrending suppresses long periods). Check
      availability first: these mostly target stars, with magnitude cuts. **Fall back to Option A** (reproduce
      Zach's TESScut + regression-corrector reduction; possibly hundreds of GB of cutouts; ask Zach for his script)
      only if availability is poor. Bin mixed FFI cadences (30 min / 10 min / 200 s) to a common cadence.
  - **Recipe:** vetted quiet noise light curves (T20) → butterpy signals over the full span (SMARTS Table 1
    distributions, with the period prior adjusted as needed, e.g. oversampling short periods, and per D4) → combine
    with the operation T20 determines (additive vs multiplicative) and SMARTS's normalization. Optionally match the
    noise level to real stars' magnitudes, since SMARTS's SNR distribution is somewhat arbitrary.
  - **Validate before scaling:** 1-year sims with SMARTS's exact parameters on the same galaxies, compared with
    SMARTS files (amplitudes, ACF properties, pipeline outcomes). Also SMARTS file minus T20's recovered galaxy
    light curve (= the injected signal) vs fresh butterpy output for the same parameters. Expect small
    differences from the newer butterpy; that's acceptable.
  - **Unknowns:** the combination operation (T20); noise-product availability (cheap MAST check, could ride
    along with T20's lookups); butterpy speed per multi-year light curve (unmeasured); storage, est.
    ~0.4 MB per 5-year 30-min light curve, ~40 GB per 100k (estimate, not measured).

### D4. Coverage-aware expectations: searchable period range, at the start and at the end
- **Raised:** Rae, 2026-10-21. A long-standing concern, reinforced by T16b: about half of the single-sector
  declines are 10–30 d / > 30 d periods that one sector simply can't measure. **Don't penalize the algorithm
  for not finding a 30 d period in 20 d of data.** Needs a joint strategy (Rae + Claude) before T16c/T16d build
  on it.
- **Two places it enters:**
  1. **Start of the pipeline:** what period range is searchable for *this* star's sectors and their
     clustering, and how that steers candidate search, gates and the ranker.
  2. **End of analysis:** recovery rates across a sample should be judged against what was searchable
     (completeness), not against all injected periods.
- **Options for the start, to discuss (not exclusive):**
  - **A. Explicit pre-search "search plan" step.** From the time sampling alone (contiguous segments, their
    lengths and spacing, ACF valid-pair coverage, cadence), compute `[P_min, P_max_reliable, P_max_possible]`,
    pass it as min/max_period to the guess_* methods and the fit, and record it for D1 ("no periodicity in
    [a, b] d").
  - **B. Soft, learned limits:** search everywhere, but give every candidate its detectability context
    (cycles observed, teeth available, P/longest segment; T16e already has these) and let the gates/ranker
    weigh it. More flexible, but less transparent.
  - **C. Hybrid:** hard limits only where physics forbids (fewer than ~2 cycles in any segment); soft
    weighting between "possible" and "reliable".
  - **D. Two-tier output:** a confident period inside the reliable range; beyond it, a lower limit or
    "possible long period" flag (cf. Astraea, Lu+2020, for single-sector long periods).
  - **E. Empirical limits:** set P_max per coverage pattern from measured recovery curves (T16c), e.g. where
    success drops below X or wrong rises above Y, rather than from first principles.
- **Options for the end:**
  - A per-star **`searchable`** truth flag (true P inside the star's reliable range). Report recovery among
    searchable stars separately. A decline on an unsearchable star counts as a **correct decline**, and a
    success there as a bonus. This touches `evaluation.py` / scorecard metric definitions, which are Rae's.
  - **Completeness maps:** recovery vs (P, amplitude, coverage), injection-recovery style, for science
    (occurrence-rate work) as well as development.
  - Restate goals accordingly (e.g. G7 "declines among measurable stars"; T18's open question).
- **Relates to:** D1 (report searched range), D2 (multi-season), T16b (finding), T16c (limits), T16d (policy),
  T16e (features), T18 open questions, G7, `evaluation.py`.
- **DECIDED (Rae + Claude, 2026-10-22).** Implementation: T16c (start), T23 (end), T16h (multi-batch).
  - **Grounding** (T16b's `stars.csv.gz`: synthetic 1/2-sector classes ×200 plus held-out seed 1, and dev_medium assigned):
    - **Reach.** The ACF reach R (the largest lag with valid-pair fraction ≥ 0.3) is ≈ 0.69× the longest contiguous segment.
      The ACF does **not** bridge sector gaps: on gapped-2, R / total baseline = 0.12. On gapped-6-12, R can exceed the longest
      segment.
    - **Cliff at R/2.** Success is flat up to P/R = 0.5 and then collapses:
      - synthetic short coverage: 0.46 / 0.38 / 0.46 at P/R ≤ 0.25 / 0.25–0.4 / 0.4–0.5, then 0.07 at 0.5–0.6 and 0 beyond;
      - dev_medium: 0 successes at P/R > 0.5 (n = 323).
      - This is the comb fit's own limit: it needs ≥ 2 teeth inside R. It agrees with pb_scan (~0.94 correct up to P/B = 0.33).
    - **Wrong answers on unsearchable stars are short false detections, not mis-measured long periods.**
      - 25–50% of all wrong answers are on stars with P > L/3 (dev_medium 44%). Almost all are 0.2–7 d answers for true
        periods of 20–170 d.
      - So a search limit doesn't remove them; they belong to the T16b/T20 false-detection problem.
    - **Effect of scoring on searchable stars.** Single-sector success goes 0.33 → 0.43 among searchable stars, and 93–100%
      of unsearchable stars are already declined. dev_medium only goes 0.15 → 0.22, so the search range isn't the main
      real-data bottleneck.
  - **Start (options A + C, with E as T16c's validation):**
    - a `searchable_range()` computed from the time sampling alone, per contiguous batch (T16h): P_min, the reach R, and
      **P_max = R/2** (Rae: R/2 is the line).
    - **Stage 1 stays uncapped**:
      - candidates beyond R/2 are kept and tagged `beyond_reach`, and the comb fit skips them at entry;
      - this keeps the information for a later "possible long period" flag (option D, deferred) and for ranker features,
        and keeps Lomb-Scargle cross-gap candidates testable;
      - most of the speed of a hard cap is kept.
    - The range is recorded as pipeline output (D1).
    - Add a P/R ranker feature (option B as a complement).
    - Gates are unchanged by D4; T16d's coverage-aware tweaks key on teeth available, which is consistent with R.
  - **End:**
    - **Per-star `searchable` truth flag:** true when the **low end of the success range** (P_lo of the active-latitude band,
      i.e. the lowest period that scores success) ≤ R/2. That means some success answer is reachable.
    - **R is frozen in `evaluation.py`** (valid-pair fraction 0.3, from the sampling only), **not** taken from the
      pipeline's `compute_acf` default, so the pipeline can't move its own goalposts. If T16c lowers `min_valid_frac`, the
      extra reach shows up as bonus successes.
    - **Report both:** every metric on all stars and on the searchable subset, plus a correct-decline rate and a wrong rate
      on unsearchable stars. Wrong answers on unsearchable stars still count as `wrong`.
    - **Goals:** keep the existing goals and propose searchable-subset goals alongside them (Rae-owned).
    - **Completeness maps:** recovery vs (P/R, amplitude, coverage class) as a scorecard figure.
  - **Multi-season / clustered sectors (Rae's batch approach):** recorded as **T16h**.

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
  **T16a (2026-10-14) adds:** the residue also changes wavelet's candidate *count*: star 00058
  gave 6, 7 or 8 in different invocations of the same code, and top-1 on 00038 shifted by
  1.5e-10. So rtol alone won't make T12's per-method check pass; compare top-1 only, or skip wavelet.

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
- **Rae, 2026-10-15:** also re-evaluate T18's strict half-lag gate vs option D on dev_small
  (success vs wrong/alias/declines and the noise false-positive rate), since the strict choice was made on synthetic data.
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

### P15. T1 phase 2: full-corpus scan for corrupt noise and galaxy IDs
- **Proposed by:** T1 phase 1 report (2026-10-15)
- **Idea:** one resumable streaming pass over all 180 bundles (per-bundle outputs,
  `ProcessPoolExecutor` over bundles, run locally by Rae) recording per file: max |flux − 1|
  and the bad orbits (`tools/audit/flux_outliers.py` logic), fill/gap template, and a noise
  fingerprint; then cluster fingerprints into galaxy IDs (≈ 2,000 galaxies). Gives a complete
  exclusion list (~12% of files) and lets T2 make dev and holdout galaxy-disjoint, not just
  train vs holdout. Vectorize `lc_spotcheck.interp_runs` first (~0.25 s/file in Python now).
  Start with `tools/audit/bundle_index.py --all` (~18 min).

### P16. Short-period signal strength in SMARTS vs synthetic
- **Proposed by:** T1 phase 1 report (2026-10-15)
- **Idea:** in the 50-file sample, a raw LS periodogram found 0/10 periods below 2 d (power
  ≤ 0.06 even when restricted near P) because galaxy red noise dominates. Quantify per-file
  injected amplitude vs galaxy noise at P < 3 d on dev_small once T2 lands, and feed it into T3's
  realism check (synthetic SNR may be optimistic at short P).

### P17. Case study: stars whose spot periods never include P_eq
- **Proposed by:** T1 phase 1 report (2026-10-15), Rae's request
- **Idea:** ~15% of SMARTS stars have their whole active-band period range outside P_eq +/-10%
  (strong shear, high-latitude spots). They stay in training as a physically real case. Once dev
  sets and pipeline runs exist: measure how far the reported period sits from P_eq and from the
  band (P_lo..P_hi) as a function of shear and latitudes, how often it lands in the band vs wrong,
  and whether any star-level or candidate-level features (e.g. ACF peak drift, wavelet spread,
  per-sector period scatter) distinguish this case. Needs the `band_disjoint`/`P_lo`/`P_hi`
  manifest columns from T2.

### P19. Confident wrong answers on undetectable SMARTS stars
- **Proposed by:** T2 report (2026-10-15)
- **Idea:** on dev_medium (full year), 8.4% of low-amplitude-tercile stars, which have almost no
  recoverable signal (success 0.033), still get a wrong period that passes every gate. Wrong
  answers span 0.8–34 d, so this isn't only the ~0.2 d failure T16a found. Characterize what lets
  them through (which method, ACF peak heights, comb-fit statistics) and test a noise guard on
  dev_medium, alongside T18's P8 strict-vs-option-D re-check. Synthetic data has 0 wrong answers,
  so this can only be tuned on SMARTS.

### P20. load_smarts_fits on a fully-flagged requested pattern
- **Promoted to T21; done 2026-10-21** (`NoUsableSectorsError`).
- **Proposed by:** T2 report (2026-10-15)
- **Idea:** `load_smarts_fits(path, sectors=[s])` raises `ValueError: attempt to get argmax of an
  empty sequence` inside `to_even_cadence` when every requested sector has a flagged orbit
  (e.g. row 005139, sector 11). Return an empty light curve or raise a clear error, with a test.
  The T2 manifests avoid it via `native_whole_sectors`, but `--coverage` runs on SMARTS can hit it.

### P21. Height-SNR gate denominator from well-sampled lags (from T16b)
- **Proposed by:** T16b report (2026-10-21)
- **Idea:** `mean_height_snr` divides by std(ACF) over all lags to ~0.7 B. On 1-2 sectors the long-lag ACF (few pairs)
  wanders to +/-0.2 and inflates the std, so clean combs fail the gate (T16b example synth 00007, dev_medium 011716).
  Estimate the noise from lags with pair fraction >= ~0.5 (or from pair counts). Needs re-runs, not a replay; changes the
  `mean_height_snr` ML feature's meaning, so add it as a parallel statistic first. Check dev_medium wrong answers (T20).

### P18. Coverage library from a better-defined target sample
- **Partly done (2026-10-15):** T16a now has per-cycle libraries for TESS years 1–7, which cover
  the northern sky, plus a pooled default. Rae documented the target table's provenance: the
  brightest TESS stars, with coverage from tess-point. The remaining idea is a fainter or field
  sample, and Year 8+ (S97 on, longer pointings; deferred by Rae, 2026-10-15).
- **Proposed by:** T16a report (2026-10-14)
- **Idea:** `datasets/coverage/cycle1_patterns.csv` comes from `target_df_shortform.csv`, whose
  selection is unknown. It is Cycle 1 only, so southern only, and drops the 4,718 of 8,156 stars
  observed only later. Rebuild it from a defined sample (e.g. all 2-min targets, or a magnitude-limited
  FFI sample from TESS-point) for Cycle 1 and Cycle 2 separately. Compare class fractions, and pass the
  file with `--coverage-library`. `tools/build_coverage_library.py` only needs a table with a `sec`
  column; for Cycle 2, map sectors 14-26 to 1-13.

### P21. `wavelet` candidate counts aren't reproducible from run to run
- **Proposed by:** T21 report (2026-10-21)
- **Idea:** on dev_small, `guess_wavelet`'s candidate count for a few stars changes between runs of the same
  code (e.g. smarts-tess-v1.0-005139 full-year: 10 or 13). This happens at unmodified HEAD too. Final answers are
  unaffected (`wavelet` isn't pooled), but `tools/compare_runs.py` then reports `DIFFERENT` for identical results.
  Find the source (a threshold at a float tie? worker-dependent state?) and fix it, or have `compare_runs` report
  non-pooled methods separately. See `reports/2026-10-21_t21-loader-no-usable-sectors/`.
