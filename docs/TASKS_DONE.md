# Done tasks

Sections moved verbatim from `docs/TASKS.md` on 2026-11-11 (P30), once their status was `done`. Grouped by the
board phase they were filed under, in board order. Statuses for T1, T16e, T21, T26 and P26 were corrected to
"merged" at the move (they still said in review / awaiting). New entries go at the end of their phase.

---

## Phase 0: infrastructure

### T0. Test suite, synthetic generator, scorecard, report contract
- **Status:** done (2026-10-13, built in-session with Rae; see changelog in PROJECT_CONTEXT.md)

---

## Phase 1: data

### T1. Audit the SMARTS training set
- **Agent:** `data-auditor`
- **Status:** done: phase 1 merged 2026-10-15 (`reports/2026-10-15_t1-smarts-catalog-audit/`): catalog + 50-file sample + 1,050-file noise/leakage check. Phase 2 = full-corpus scan (bundle index, corrupt-noise list, per-file galaxy IDs), if Rae wants it; see P15.
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

---

## Phase 2: candidate generation and fitting

### T4. Fix `guess_pairwise_histogram` crash (open issue 8)
- **Status:** done, merged 2026-10-13 (report `reports/2026-10-13_t4-pairwise-histogram-no-raise/`).
  Its open questions 1–3 became T13–T15.
- **Goal:** return `[]` instead of raising when <2 ACF peaks are found.
- **Done when:** the two strict-xfail entries in `tests/test_guesses.py::KNOWN_FAILURES`
  are removed and the suite passes. A good first task for trying out the agent workflow.

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

### T16a. Coverage-degradation infrastructure
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

### T16b. How do candidate methods degrade with coverage?
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

### T16c. Search upper limits for patchy data
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
- **Status:** done, accepted 2026-11-09 (`reports/2026-10-22_t16c-searchable-range/`). Rae kept every recommendation: a single R/2 for all classes, `min_valid_frac` 0.3, P_min = 4 cadences, and the draft goals (to add once v4 is on main). T16h unblocked (`searchable_range`).
- **Do:** define the maximum searchable / reportable period as a function of
  coverage (longest contiguous span, total baseline, ACF valid-pair coverage).
  Measure reliability (success vs wrong) as P approaches each candidate limit, and
  pick limits that keep "wrong" low.
- **Done when:** a `max_period(coverage)` rule implemented in one place, with
  justification plots, and goals proposed for Rae.

### T16e. Coverage-aware ML features
- **Decision (Rae, 2026-10-15):** **defer the multi-season features** (n_seasons, per-season agreement/scatter) until T16g/D2. Single-season coverage features only for now.
- **Update 2026-10-15:** add n_seasons, per-season period agreement and per-season scatter alongside the
  single-season coverage descriptors. **Unblocked by T16a (merged 2026-10-15).**
- **Agent:** `feature-researcher`
- **Status:** done, merged 2026-10-21 (`reports/2026-10-15_t16e-coverage-features/`): 7 single-season coverage features (FEATURE_DOCUMENTATION Section H); mostly null on synthetic data, but several existing features change meaning with coverage (see report).
- **Do:** add coverage descriptors (n_sectors, longest span, duty cycle, P/longest
  span, n_cycles observed) to `ml_features.py`, so the ranker can learn how much to
  trust a candidate under a given coverage. Relates to open issue 6 (multi-sector
  support).

### T20. Are the SMARTS noise galaxies really quiet? (training-set flaws)
- **2026-11-10:** coherent-quiet sources reclassified `periodic_low_amplitude` (excluded); quiet P_ref investigation in `reports/2026-11-10_quiet-noise-periods/` (no further exclusions recommended; awaiting Rae).
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
- **Status:** done, merged 2026-10-21 (report `reports/2026-10-21_t21-loader-no-usable-sectors/`). `NoUsableSectorsError(ValueError)` raised before regridding; callers already record it as a decline; dev_small assigned + full-year bit-identical.
- **Problem:** `load_smarts_fits(sectors=[...])` raises deep inside `to_even_cadence` when every requested
  sector is flagged. **Do:** raise a named exception (e.g. `NoUsableSectorsError`) early, with a clear message,
  record it as a decline (`none`, with the reason) in the scorecard and batch runners, and add a test.
  Results on the existing dev sets must be unchanged.

### T22. Scorecard v4: band-width breakdown + version bump
- **Agent:** `method-improver` (small; scorecard reporting only)
- **Status:** done, merged 2026-11-09 (was branch `claude/t22-t23-scorecard-v4-f5b694`; report `reports/2026-10-22_scorecard-v4-bandwidth-searchable/`; single bump to SCORECARD_VERSION 4; v4 baselines in benchmarks/baselines/*_v4.json).
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
- **Status:** done, merged 2026-11-09 (was branch `claude/t22-t23-scorecard-v4-f5b694`; report `reports/2026-10-22_scorecard-v4-bandwidth-searchable/`; single bump to SCORECARD_VERSION 4; v4 baselines in benchmarks/baselines/*_v4.json).
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
- **Status:** done, merged 2026-11-10 (`reports/2026-11-10_scorecard-v5-noise-period/`), implemented as a breakdown of `wrong`. Rae: merge as is, so `noise_period` matches the noise source's `P_ref` for every class, quiet included; halves and doubles stay a diagnostic (`noise_alias`). Revisit if the quiet-noise probe's shuffled null shows the 12 quiet-class matches are chance.
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
- **Status:** done, merged with T24 2026-11-10; goals for Rae to set. **Follow-up:** once the quiet-noise reclassification (`periodic_low_amplitude`, branch prefix `quiet-noise-reclass`) regenerates the flags file, rebuild `benchmarks/noise_sources_dev.csv` from it (`tools/build_noise_table.py`) instead of the review's `noise_quiet_coherent.csv` list, and re-baseline if any v5 key moves.
- **Why:** 128 of dev_medium's 1,000 stars sit on periodic noise sources, with a 43% wrong rate vs 3.8% on quiet
  sources. Method changes should be judged on both.
- **Do:** a per-star descriptor column (`noise_class`, `noise_periodic`, `noise_exclude`) and metrics on the quiet
  subset (`final.quiet_noise.*`) next to the full-set metrics, plus a `--noise-subset` option or equivalent. Existing
  metrics stay bit-identical. Propose quiet-subset goals; Rae edits `goals.json`.

### T26. Detectability ceiling for SMARTS stars (roadmap step 1, 2026-11-09)
- **Agent:** `data-auditor` (+ scorecard reporting; can share T24/T25's scorecard bump)
- **Status:** done, merged 2026-11-11 (`reports/2026-11-10_t26-detectability/`).
  Per-star detectability D_band·√N_cyc (injected signal = SMARTS flux / T20 stack, band-limited S/N on the pipeline's cadences)
  for dev_medium in both coverage modes. Proposed `detectable` = ≥ 5 (keeps 97% of successes). Detectable full-year success
  0.43 / wrong 0.10 / declined 0.47 (n=560); 385 of 650 declines are undetectable stars, 265 are detectable. Scorecard flag
  proposed as P26, then **implemented as scorecard v6 at Rae's request (2026-11-10)**: `benchmarks/detectability_dev.csv`
  (dev_small + dev_medium, both coverage modes) joined into per-star `detectability`/`detectable` and `final.detectable*` /
  `final.undetectable` metrics; existing metrics bit-identical (dev_small check). **Rae (2026-11-10): undetectable stars
  are not ranker negatives;** use the noise-only stack light curves (T27) for that. Threshold default 5 (T26 proposal).
- **Why:** dev_medium declines 65% of stars, and T16b found most declined stars' correct candidates look like noise.
  Without a per-star "is the signal detectable at all" measure, we can't tell whether success 0.27 is poor or near the ceiling.
- **Do:** per-star injected-signal strength. Multiplicative injection (T20) means signal ≈ SMARTS flux / recovered
  noise light curve, or use the catalog's activity/amplitude. Compute e.g. signal amplitude / noise RMS on the same
  coverage. Plot success / declined / wrong against it, and propose a `detectable` descriptor + metrics on the
  detectable subset (as with `searchable`). Existing metrics stay bit-identical.
- **Done when:** a report with outcome-vs-detectability plots and a proposed threshold (Rae decides).

### T27. Non-rotator negatives from SMARTS noise light curves (roadmap step 2, 2026-11-09)
- **Agent:** `data-auditor` or `ranking-model-builder`
- **Status:** done, **merged 2026-11-10** (branch `claude/t27-nonrotator-negatives-008ccf`; `reports/2026-11-10_t27-nonrotator-negatives/`). Synthetic red noise not run. **Rae accepted the coordinator's recommendations on open questions 4–7 (2026-11-11):** (4) drop from the negatives any source whose stack has a persistent weak LS peak (A/B halves agree, like T20's `persist`) and list them alongside `periodic_low_amplitude`; do this inside the ranker-v1 task; (5) no separate task: fold the 6 `systematic_common_mode` full-year FPs into T29; (6) keep the 125 spike stacks out, consistent with the rotator sets; revisit if the pipeline gains spike masking; (7) skip synthetic red noise for now; reconsider for Roman (T28).
- **Why:** every SMARTS star is a rotator, so the ranker's decline threshold and false-positive rate would be calibrated
  only against weak rotators. T20 already isolated the noise sources: ~1,500 quiet ones (real-noise **non-rotators**) and
  206 periodic ones (mostly **real neighbouring variable stars**: contaminating-variable examples).
- **Do:** a scorecard/feature-table mode for noise-only light curves (truth = "no rotation"). Run on the quiet set
  plus synthetic red noise. Report the false-positive rate and which gates and features fire, and produce negative
  rows for T10. Use the coverage libraries so the negatives see the same sampling patterns.
- **Done when:** a negatives table T10 can join, plus false-positive metrics by coverage class.

---

### T5. Short-period recovery (<3 d) (open issue 1)
- **Status:** done, merged to main 2026-11-11 (Rae OK). Optional methods: none pooled (Rae accepted). Report
  `reports/2026-11-11_t5-short-period-recovery/`). Full-year <1 d deficit = top-10 crowding by red-noise peaks in
  `lombscargle`/`acf_fft`; fixed with hybrid raw-power / local-SNR peak ranking (new default, no extra compute).
  dev_medium full-year <1 d Stage-1 union 0.44 -> 0.70, fitted ceiling 0.33 -> 0.65; final success barely moves
  (`half_lag_consistency` rejects the rescued candidates). No optional method recommended for the pool. G2/G3 still fail.
- **Goal:** improve Stage 1 recall and final correctness in the `<1d` and `1-3d`
  bins. Separate the Nyquist/points-per-cycle wall (unfixable) from search
  failures (fixable).
- **Update (coordinator, 2026-11-11; `reports/2026-11-11_candidate-recall-by-method/`):** Rae's priority is <1 d recall,
  with minimal extra compute and no new false positives that survive the ranker. On dev_medium, all 7 methods together
  put the true period in the raw guesses for only 44% of <1 d stars full-year (63% assigned); the 3 pooled methods 34% / 60%.
  Best single addition is `acf_fft_highpass` (+0.099 full-year, +0.029 assigned). Every method is worse at <1 d full-year
  than on the same stars' assigned coverage (`acf_fft` 0.60 → 0.34): diagnose that shared deficit first, then choose which
  optional method(s) to pool by marginal recall, per-star compute, and FP rate on the T27 non-rotators.
- **Done when:** goals G2/G3 in `benchmarks/goals.json` pass with no regression guard
  tripped.

## Phase 3: features

### T9. Feature literature and empirical search
- **Agent:** `feature-researcher`
- **Status:** done, merged 2026-10-14 (`reports/2026-10-14_t9-feature-search/`). Rae answered its open questions 2026-10-14 (see P1, T10, T11, T17, P8).
- **Goal:** find features that separate correct candidates from P/2 and 2P
  aliases.
- **Done when:** report with a literature table, per-feature separation (AUC)
  by period bin, and any new features implemented, documented and tested.

---

## Proposed (agents add here; Rae promotes)

### P26. Scorecard `detectable` descriptor from a per-star detectability side table (from T26, 2026-11-10)
- **Status:** done, merged with T26 2026-11-11 (Rae said yes 2026-11-10): scorecard v6.
- **Proposed by:** T26 (`reports/2026-11-10_t26-detectability/`).
- **Idea:** a scorecard v6 `detectable` flag + detectable-subset metrics (like `searchable`; existing metrics bit-identical).
  The scorecard can't compute it itself (needs the T20 noise stack from the drive), so commit a per-star side table
  (`star_id, coverage pattern, snr_rot`) built by `tools/audit/detectability.py` for dev_small/dev_medium, and add a
  `--detectability <csv>` option that joins it. Keyed by coverage pattern, since detectability depends on it.
- **Not done yet:** a table for the holdout or other coverage draws (build with `tools/build_detectability_table.py`).

### P28. Headline card with fixed definitions (from the midway review, 2026-11-11)
- **Status:** done, merged 2026-11-12 (`facc568`). **Period population decided (Rae, 2026-11-12): `dev_medium`'s own
  period-bin mix is the default for R2–R4, not McQuillan+2014** (Kepler's selection has almost no P < 1 d stars, and we
  must stay sensitive to fast rotators). **Rae (2026-11-13) took the report's/coordinator's recommendations:** quote
  prevalence f = 0.1 as the canonical R4 number (0.3 / 0.5 stay in the JSON); activate H1, H3, H4, with H2 and H5 as
  regression guards; R5 reports hand both ways (declines as misses, and among answered stars only); P29's 70 d run is
  deferred. Follow-up: H1–H5 live only in the report. Encoding them needs `check_goals.py` to read the headline-card
  JSON (a protected-file change, now approved in principle).
- **Proposed by:** midway review (rec. 5).
  `tools/headline_card.py` built and tested; goals H1–H5 proposed in the report (`goals.json` untouched).
  Awaiting Rae: period population, prevalences, which goals to activate.
- **Idea:** one small, fixed table that every method/ranker report quotes, computed from the existing scorecard per-star
  CSVs. It needs no new scorecard subsets. Rows:
  - assigned coverage at the real coverage mix (`coverage_weight`);
  - success re-weighted to a named period population;
  - correct-only beside success;
  - precision at a stated non-rotator prevalence;
  - a Claytor+2022-comparable "fraction within 10% / 20% at the most-credible half".
- Activate 3–5 goals in `goals.json` on these numbers (Rae's call).

### P30. Repo and docs diet (from the midway review, 2026-11-11)
- **Agent:** `repo-curator`.
- **Status:** done, merged to main 2026-11-11 (Rae OK) (`reports/2026-11-11_p30-repo-docs-diet/`); results bit-identical on dev_small.
- **Idea:** one session, results bit-identical:
  - **Code:**
    - delete `claude_files/old/`;
    - settle open issue 9 (`batch_test_acf_fft_highpass.py`);
    - retire `batch_test_guesses.py` and `run_debug_baseline*.py` (the scorecard and `build_feature_tables.py` cover them);
    - prune superseded `benchmarks/baselines/`.
  - **Docs:**
    - refresh `PROJECT_CONTEXT.md` §4 (issues 2–5 and 12 are stale) and `METHODOLOGY_NOTES.md` §7 (both still say there is no ranker);
    - fix the code map (lists a missing `example_usage.py`);
    - fix the stale TASKS statuses (T1, T16e, T21, T26);
    - move done tasks to `TASKS_DONE.md` and old changelog entries to an archive file;
    - cap report TL;DRs at ~150 words in the write-report skill.

