# Proposed tasks

Split out of `docs/TASKS.md` on 2026-11-13 to keep the board short (agents read only the section they need). Agents add new follow-up ideas at the end of this file; Rae promotes them to `docs/TASKS.md`. When one is merged, move its section to `docs/TASKS_DONE.md`.

---

*Done here (moved to `docs/TASKS_DONE.md`): P26, P28, P30.*

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
  **T12 (2026-11-12) located it:** `_morlet2_kernel`'s `np.exp(-0.5 * x**2)` (numpy 1.26.4, AVX-512 on the i5)
  returns two different byte patterns for byte-identical input within one process, depending on allocation
  state (input alignment alone doesn't trigger it), so the CWT coefficients, the GWPS and the wavelet
  candidates/features vary in the last bits. Unchanged code gives 2-4 distinct GWPS per star over 8 in-process
  calls (dev_small 031596, 419951, 107369). Any change that alters allocation patterns can flip it, and it
  reaches the ranker tables (`wavelet_nearest_peak_*`). Final answers are unaffected (wavelet isn't pooled).
  Fix options (each changes wavelet bits once, so Rae's call): build the kernel's Gaussian envelope from a
  deterministic path (e.g. `np.exp` on a fixed scratch buffer, or `math.exp` per element for the short
  kernels), or round the GWPS to ~1e-12 before peak fitting. See `reports/2026-11-12_t12-speed-up/`.
  **Resolved by T31 (2026-11-13):** two SIMD loops, the float64 `np.exp` and the complex `np.abs`; the second one flipped
  in 4 of 12 fresh processes. The kernel is now one complex exp, the power is re*re+im*im, and the Gaussian fits use a libm
  exp. A dev_medium rerun is bit-identical. See `reports/2026-11-13_t31-comb-fit-speed/`.

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

### P22. Searchable-range follow-ups from T16c
- **Proposed by:** T16c (2026-10-22), `reports/2026-10-22_t16c-searchable-range/`.
- **Ideas:** (a) **done 2026-11-09:** `test_reach_matches_frozen_evaluation_reference` checks that `evaluation.acf_reach`
  equals `coverage.searchable_range(t, f)["reach"]`. It found the missing 0.7-baseline cap in `searchable_range`, now fixed;
  (b) record `EnsembleResult.beyond_reach_candidates` in the scorecard's `per_candidate.csv` (`beyond_reach`
  column), e.g. in T23's scorecard bump; (c) for T16d: skip `pairwise_histogram` when R < ~30 d (it returns no
  candidates for ~2/3 of 1–2-sector stars); investigate single-sector LS top-1 errors (sector/ramp systematics,
  not window aliasing); (d) a synthetic suite with log-uniform P/R in 0.2–0.8 per coverage class to tighten the R/2
  validation near the edge; (e) a short-period synthetic suite to measure a "reliable from" P floor (T16d or D1; Rae's call).

### P23. Don't discard beyond-reach candidates that some method can still recover (Rae, 2026-11-10)
- **Proposed by:** Rae, from scorecard v5's P22(b) rows (`reports/2026-11-10_scorecard-v5-noise-period/`).
- **Observation:** on dev_medium full-year, 108 of the 1,261 `beyond_reach` candidates (P0 > R/2, skipped by the comb
  fit since T16c) score as success by their raw P0: 57 `correct`, 51 `diffrot`. So Stage 1 sometimes proposes the right
  period above the ACF's reach. Not yet known: how many of those stars are searchable (truth can sit just under R/2
  while P0 sits just over it, or the diffrot range can straddle R/2) and how many currently have no success.
- **Principle:** R/2 is a limit of the *ACF* comb fit, not of every method. Wavelet or LS may still identify a period the
  ACF can't confirm. A hard cap shouldn't stop such a candidate from reaching final selection.
- **Ideas:**
  - (a) Quantify first: count the 108 by star searchability and current outcome.
  - (b) In the ranker (T10/T11), keep `beyond_reach` candidates as rows, with `P_over_reach` and per-method support as
    features, and let the model learn when they're trustworthy.
  - (c) Consider a non-ACF scoring path for P > R/2, e.g. wavelet/LS-based evidence, instead of skipping outright.
  - (d) Re-check the other hard limits the same way (P_min, gate cutoffs) once the ranker exists.
- **Note:** T10's feature tables already include `beyond_reach` candidates as rows (brief, 2026-11-10).

### P24. Ranker decline stage: period-aware accept model (from T11 v0, 2026-11-10)
- **Proposed by:** T11 v0 report. The ranker's top-1 ordering is far ahead of hand arbitration (no-decline success
  0.44 vs 0.25 full-year), but one global top-score threshold over-declines long periods: 34 of the 39 full-year stars
  hand arbitration gets and the ranker misses had a *correct* top-1 below threshold (10–30 d, > 30 d), and OOF→dev
  precision drops 0.89 → 0.67–0.75.
- **Idea:** star-level accept classifier on OOF top-1 rows (top score, margin, top period, `P_over_reach`, coverage
  descriptors), or per-candidate-period-bin thresholds set on OOF; add a full-year (or mixed-coverage) training
  table so long-period full-year stars are represented; refit once T27 negatives exist.
- **Update (T27, 2026-11-10):** negatives exist. A star-level `binary:logistic` head on the top-1 row (+ score, margin),
  trained on OOF scores with T27 negatives, is prototyped in `tools/train_ranker_negatives.py` (`v0+neg_all+head`): at
  prec 0.89 (negatives 1:1) it beats v0 at the same rule, and matches hand success with 40–60% fewer wrong answers; > 30 d
  is still under-answered. Needs a ratio/precision decision from Rae, a learning curve and per-bin thresholds.
- **Decision (Rae, 2026-11-11):** adopt the star-level decline model for ranker v1. Use all three coverage draws per
  negative source (~2.2 negatives per rotator); the ratio only needs to be good enough for now, since Roman and
  variable-star negatives will be added later. **No full-year training table yet:** build v1 with the decline model and
  per-bin thresholds first; if 10–30 d / > 30 d full-year still lag hand arbitration, build a full-year table for the
  1k subset as a direct test.

### P25. Use the light curves' noise level: empirical noise features, then per-sector weighting (Rae said yes, 2026-11-11)
- **Proposed by:** coordinator (2026-11-10), after Rae asked whether the pipeline should use photometric uncertainty.
- **Idea:** the pipeline ignores how noisy each light curve is. Two steps:
  1. **Empirical noise features** in `ml_features.py`: per-sector and pooled point-to-point scatter, and white vs
     red noise on rotation timescales, so the ranker and the decline model know how noisy a star is. Measured from
     the flux, not from SMARTS' error column. Judge by ranker/decline OOF and dev_medium, with feature importances.
  2. **Per-sector inverse-variance weighting** when sectors are pooled (ACF/periodograms), so one noisy sector doesn't
     dominate. That's a method change, so it goes through the dev_medium scorecard with `check_goals.py --baseline`.
- Step 1 can ride along with ranker v1; step 2 is its own task.

### P27. Real-TESS validation set: known rotators and known non-rotators (from the midway review, 2026-11-11)
- **Proposed by:** midway review (`reports/2026-11-11_midway-review/`, rec. 1).
- **Why:** nothing has been scored on real TESS light curves yet. SMARTS noise comes from faint "galaxies" reduced closer to
  SAP, while real rotation work uses PDCSAP/QLP/TESS-SPOC (long-period attenuation, real stellar noise). The non-rotator
  false-positive rate on real stars (pulsators, EBs, blends) is the number a Roman/TESS catalogue depends on.
- **Idea:** a small (≤ 500 stars), frozen manifest of FFI light curves: known rotators (Kepler-field overlap with
  McQuillan+2014, cluster members with K2 periods, an independent TESS CVZ catalogue), plus known non-rotators/variables
  (Prša+2022 EBs, Gaia DR3 classes). Add a scorecard dataset string, with correct-within-10%/20%, alias and FP rates
  per period bin. Rae picks the sources (review open question 1). The downloads are Rae's local job.
- **Rae's initial direction (2026-11-11; fuller answer to come):**
  - **Known rotators:** start with members of young open clusters. Their light curves are well studied, and each period can
    be sanity-checked against the gyrochronology expected for the cluster's age (isochrone).
  - **Known non-rotators** are harder. Options: white dwarfs, or follow Claytor+2025 and use red clump stars (slow rotators).
    Zach Claytor shared the TIC IDs of the 2025 paper's training stars; retrieve them and check that they are quiet.

- **Rae's additions (2026-11-12):**
  - Check the white-dwarf / red-clump instinct against the literature, and note the problems other papers report with
    each as a "non-rotator" sample.
  - Keep a running list of useful papers and report it as a citation list.
  - For every literature period, record the method (photometric spot modulation, v sin i / rotational broadening,
    spectroscopic activity, …) and the data set (ground-based survey, K2, Kepler, TESS, …).
  - Zach's TIC list is committed at `datasets/external/claytor2025_nonrotator_ticids.txt` (23,332 IDs; provenance and
    caveat in `datasets/external/README.md`).
  - Consider other light-curve sources too, e.g. TGLC and CDIPS, alongside QLP / TESS-SPOC / eleanor.
- **Status:** scoping session started 2026-11-12. Report on `claude/p27-validation-scoping-4c1c1b`
  (`reports/2026-11-12_validation-set-scoping/`), unmerged.
- **Rae's decisions (2026-11-13), on the report's Q3–5:**
  - Q3 mix (268 rotators / 122 negatives / 40 expected-undetectable / 50 EBs): about right.
  - Q5 long periods: **not OK to have nothing at 70–180 d.** Add a small long-period block; options in the coordinator's
    note below. Rae picks.
  - Q4: a 0.2–0.5 d detection on a red-clump giant is its own outcome, named **`pulsation`** rather than `seismic`
    (coordinator agreed). `pulsation` also covers δ Sct / γ Dor hits on hot stars, and the star's class
    tells the mechanisms apart.
  - Q7 (freeze): after the long-period block is settled.
- **Long-period block options (coordinator, 2026-11-13; to quantify before Rae picks):**
  - (a) Ground-based M-dwarf periods independent of TESS: MEarth (Newton+2016, 2018; periods to ≳ 100 d) and ASAS
    (Kiraga & Stępień 2007; Suárez Mascareño+2016). Pick the stars TESS covered with ≥ 10 sectors in a year (CVZs).
  - (b) Kepler-field stars at 50–100 d (Santos+2019/2021 extend past McQuillan+2014's ~70 d ceiling). Few reach > 70 d
    with TESS coverage.
  - (c) Gaia DR3 `vari_rotation_modulation`, but its periods are mostly short; check how many exceed 70 d.
  - Whatever the source, a 70–180 d star needs ~2 cycles, i.e. 140–360 d of near-contiguous coverage (CVZ). In most
    cases the honest outcome on TESS is a decline. So this block mainly measures decline honesty and SAP-vs-PDCSAP
    attenuation, plus recovery for the few CVZ stars. Target ~20–40 stars.

### P29. Roman-season proxy on SMARTS (from the midway review, 2026-11-11)
- **Proposed by:** midway review (rec. 2).
- **Status:** first cut done (2026-11-12, `reports/2026-11-12_headline-card/`): `consecutive-3-5`, P ≤ 30 d =
  0.205 hand (T5) / 0.284 ranker v0 on dev_medium assigned. The true 70 d window (per-star-sectors manifest +
  `--max-span-days 70` dataset option, ~14 min) is specified but not run. **Rae (2026-11-13): not now.** Keep
  `consecutive-3-5` until the WG6 Roman sets (T28) give real Season-1-like light curves.
- **Idea:** until the WG6 sets exist (T28), report SMARTS performance on one contiguous ~70 d window: 5 consecutive
  sectors trimmed to 70 d, or the `consecutive-3-5` class as a first cut. Today `consecutive-3-5` at P ≤ 30 d
  gives 0.20 hand / 0.28 ranker v0 on dev_medium assigned. Cheap, and it says what Season 1 should expect from the TESS-trained model.

### P31. Rebuild ranker feature tables after the T5 Stage-1 ranking change
- **Proposed by:** T5 report (2026-11-11)
- **Idea:** T5 changed which peaks `lombscargle`/`acf_fft` return (hybrid ranking) and the meaning of guess `rank`/`strength`.
  The <1 d fitted-candidate ceiling doubled (0.33 -> 0.65 full-year) but hand arbitration declines most of the new
  candidates, so the gain reaches answers only through the ranker. Rebuild the dev_medium T10 tables and train_3k on the
  T5 code, re-run ranker v0/v1 CV, and report <1 d success at matched precision plus FP rate on the T27 negatives.

### P32. `half_lag_consistency` vs sub-day (and 1-30 d) success candidates
- **Proposed by:** T5 report (2026-11-11)
- **Idea:** after T5, every <1 d star with a fitted success candidate but no success answer has all its success
  candidates failing `half_lag_consistency` (79/79 full-year); the gate also rejects 55-68% of 1-30 d success candidates.
  Measure the gate's recall-vs-FP curve on rotators and the T27 negatives (and whether 30-min cadence makes the P/2
  lag test unreliable below ~1 d) before any threshold change.

### P33. `pairwise_histogram` on long baselines (degenerate pair support)
- **Proposed by:** T5 report (2026-11-11)
- **Idea:** full-year it finds no <1 d candidate even at snr_rot ~140: with ~190-240 clean ACF peaks every multiple kP
  has ~m-k supporting pairs, so the winner is set by bin merging, not the fundamental (re-binning to relative widths
  doesn't fix it). Try nearest-neighbour spacings (j-i <= K) or a comb fold of the peak list. Low priority: the fixed
  `acf_fft`/`lombscargle` now cover these stars.

### P34. Roman-cadence ROT (spotted rotators) (from T28 scoping, revised 2026-11-12)
- **Proposed by:** T28 scoping (`reports/2026-11-12_wg6-dataset-scoping/`, Part 2).
- **Idea:** WG6 shipped GP-resampled Roman light curves (6 × 71 d, F146 at 12.1 min) for every class except the 3,172
  OGLE spotted rotators. The only Roman-cadence rotators are 54 TESS M-dwarf templates (FL), with no spot evolution.
  Either:
  - (i) ask WG6 to resample ROT (their `add_rotators` branch already holds the archival ROT); or
  - (ii) do it ourselves on WG6's grid (`roman_times_shortcadence.npy`) with WG6's improved σ(mag): a quasi-periodic GP
    per OGLE season (`celerite2`/`tinygp`: Rae's OK), or inject ROT periods and amplitudes into `synthetic.spot_signal`.

  Add spot evolution either way.
- **Needs Rae:** (i) vs (ii), and the GP package.

### P35. `claude_files/wg6.py` loader + `datasets/wg6_roman_v0.csv` manifest (from T28 scoping, revised 2026-11-12)
- **Proposed by:** T28 scoping. Design in the report's Part 2 "Revised game plan".
- **Idea:**
  - Loader: `load_wg6_roman(zip_or_path, member=None, band="F146", season=None)` reads in place from the zips (no
    42 GB extraction needed). It takes the JD column, or the `.npy` grid for simple, converts mag to relative flux with
    a NaN-safe median (FL has NaN cadences), cuts one 71-d season, and returns σ. It returns the `load_smarts_fits`
    container. Keep `load_wg6_raw` (OGLE I, HJD fix) for ROT work.
  - Manifest: `zip`/`member` relative to `WG6_RESAMPLED_ROOT`, with `variant`, `cls`, `P_true`, `p_convention`,
    `P_phot_expected`, `fl_template`, `wfmag`, `season`, `is_rotator`, `is_eb` and `eval_set`. **Split by FL template
    and by OGLE star.**
  - Dataset strings: `wg6r_fl_v0_improved_s<k>` and `wg6r_contam_v0_improved_s<k>`.
  - Test with a tiny synthetic FITS-in-zip fixture (no drive).

### P36. WG6 contaminants as the first astrophysical ranker negatives (from T28 scoping, revised 2026-11-12)
- **Proposed by:** T28 scoping.
- **Idea:**
  - With P35: ~300 improved-noise stars per class (ECL, ELL, T2CEP, RRLYR, CEP + FL "flat"; HB, DSCT, LPV and CV as
    stress tests), one season each, as star-level declines next to T27's noise stacks. Every candidate is at relevance 0.
  - Tag ECL/ELL candidates near P_orb or P_orb/2 as `eb_period` for T29.
  - Report the per-class FP rate at matched rotator precision on the FL rotators. Today 14/16 contaminants get confident
    periods (4/4 T2CEP and RRLYR).
  - Compute: 1.1 s per star-season, so ~1.4 CPU-h for 300 × 5 classes + 1,600 FL × 2 magnitudes.
- **Needs Rae:** EBs/ELLs as plain negatives vs a flagged class.

### P37. TESS flare stars as a real-star validation set (from T28 scoping, 2026-11-12)
- **Proposed by:** T28 scoping.
- **Idea:**
  - The 390 *raw* WG6 flare files are real TESS 20-s PDCSAP M dwarfs; 273 have literature rotation periods
    (Günther+2020, Medina+2020).
  - Bin to 30 min and score the current pipeline and ranker per sector: the first real-star (non-SMARTS) TESS check,
    i.e. the over-tuning test from roadmap step 3.
  - Complements the Roman-cadence FL set, which is built from 65 of these templates.
  - Caveats: literature periods come from different methods, and Medina's MEarth periods run to 180 d.

### P38. Further bit-identical speed-ups not done in T12 (from T12, 2026-11-12)
- **Proposed by:** T12 report (`reports/2026-11-12_t12-speed-up/`).
- **Idea (ranked by measured share of the optimized per-star time):**
  1. The comb fit's `scipy.optimize.least_squares` internals (~57% of the scorecard path; SVD per trust-region
     step, Python overhead in `trf_bounds`). Bit-identical options are exhausted; anything faster (looser tolerances,
     `tr_solver="lsmr"`, fewer points per window, a C-level solver) changes results and needs a before/after scorecard.
  2. `guess_wavelet` (~20%): 200 kernel FFTs + inverse FFTs at the padded length per star. Same-bits options are
     exhausted. A result-changing option is to compute the GWPS from `|FFT(data) x FFT(kernel)|^2` at a reduced
     scale set, or drop `info["wavelet_power"]` (a 200 x N float array kept per guess; memory only, ~27 MB per
     full-year star).
  3. Scorecard scheduling: submit tasks longest-expected-first (e.g. by number of sectors) to trim the 4-worker tail.
     Wall time only, no result change.
  **T31 (2026-11-13) measured items 1-2** as switchable variants (`COMB_VARIANTS`, `CWT_VARIANTS`); see its report.

### P39. Same SIMD-flip pattern in `guess_acf_fft` (from T31, 2026-11-13)
- **Proposed by:** T31 report (`reports/2026-11-13_t31-comb-fit-speed/`).
- **Idea:** `guesses.py` `guess_acf_fft` computes `np.abs(fft_vals) ** 2`. In T31 the complex `np.abs` SIMD loop (numpy
  1.26.4, i5 AVX-512) returned different last bits in 4 of 12 fresh processes. acf_fft is pooled into the final answer,
  and T14's 40-star cross-process test never caught a flip there. Switch it to `guesses._abs2` (one-time bit change of acf_fft
  candidates; needs a before/after scorecard and new baselines), and grep for other `np.abs(complex)` / float64 `np.exp`
  on per-star arrays that feed scored values.

### P40. A vertex gate for the `local` comb solver (from T31, 2026-11-13)
- **Proposed by:** T31 report.
- **Idea:** the closed-form `local` comb variant is 6x faster on the production path and gains 33 stars on dev_medium
  (29 of them declined by main's `frac_vertex_in_window`). But its own vertex gate is nearly vacuous (it fails on 9%
  of candidates vs 60% under lsq), and assigned wrong rises 0.055 -> 0.066. Give it a meaningful equivalent, e.g. the
  fraction of windows whose free local vertex was accepted, or the scatter of vertices about the fitted line, and a
  robust (soft-L1 / Huber) per-window height fit. Then rescore. This could make `local` the fast default without the extra wrongs.


### P41. Skeptical reviewer of the methodology (Rae, 2026-11-13; logged, not scheduled)
- **Proposed by:** Rae.
- **Idea:** a standing skeptical reviewer of the *methodology*, not of one branch (the existing `reviewer` agent checks a
  finished branch's numbers). Questions it would press: is SMARTS representative of real TESS stars (spot model, noise,
  diffrot), does scoring against `REL_TOL` / alias / diffrot outcomes flatter us, are dev_medium and the `_v7` baselines
  being overfit by repeated method tuning, are ranker negatives realistic, and does anything leak between training, dev
  and holdout. Possibly a new `.claude/agents/methodology-reviewer.md` role, run at milestones (e.g. before ranker v1 and
  before the paper). Related: the midway review (`reports/2026-11-11_midway-review/`) covered part of this once.

### P42. Fewer candidates per method: top-N cut and better candidate ordering (Rae, 2026-11-13; logged, not scheduled)
- **Proposed by:** Rae.
- **Idea:** every `guess_*` method currently passes `n_guesses=10` candidates (5 for `acf_fft_highpass`) to the comb fit,
  which dominates per-star cost. If the correct period usually ranks near the top, a smaller N saves time.
  1. **Right N per method:** recall@N (true period or diffrot within `REL_TOL`) vs N = 1–10, per method, per period bin,
     full-year and assigned, plus the fit-time saved. This can come from existing data (per-method ranks in the
     scorecard's `per_candidate.csv` / the T10 feature tables) without rerunning the pipeline.
  2. **Push the right candidate higher:** cheap changes to how each method ranks its peaks (e.g. long-timescale power
     crowding short periods out of the top 10 on full-year light curves; see
     `reports/2026-11-11_candidate-recall-by-method/`, where every method's <1 d recall drops full-year vs assigned).
- **Caution (Rae):** don't tune N or the rankings tightly yet. Other telescopes (Roman, other training sets) differ in
  baseline, cadence and noise, which changes candidate ordering. For now, only a quick check for easy, robust gains
  (e.g. a method whose recall saturates by N=3 everywhere).
