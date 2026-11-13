# Open discussions (Rae to decide; not ready for an agent)

Split out of `docs/TASKS.md` on 2026-11-13 to keep the board short (agents read only the section they need). Coordinators read these only when a task depends on one.

---

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
- **Architecture leaning (2026-11-09):** generate candidates per batch (plus T16h's stacked ACF), then **aggregate
  into fixed-size star-level features** (n_batches_testable, n_batches_agree, n_batches_half, per-batch period
  scatter) **before** the ranker. Not stitch-everything (spot evolution breaks phase coherence), not compare-only-at-the-end
  (that throws away cross-batch agreement as a learnable signal), and no variable-length per-batch inputs to the model.
  Off the Roman Season 1 critical path (one season = one batch).

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

