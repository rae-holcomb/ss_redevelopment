# Task board

Each task is sized for one agent session. Rae owns this file's priorities
and acceptance criteria. Agents update only the **Status** line of their
own task (and add follow-up tasks under "Proposed", never directly to the
active list).

Status values: `blocked` · `ready` · `in progress (<branch>)` · `in review (<report folder>)` · `done (<report folder>)`

Done tasks are in `docs/TASKS_DONE.md` (moved 2026-11-11, P30). When a task is merged, the coordinator
sets its status to done and moves its section there (see `docs/AGENT_WORKFLOW.md`).

Agent roles are defined in `.claude/agents/`. Every task ends with a
report (`write-report` skill) and, if code changed, a before/after
`benchmarks/check_goals.py` table.

---

## Roadmap (agreed with Rae 2026-11-09; coordinators start here)

**Goal:** a working, multi-mission (TESS + Roman required; Kepler/ground-based wishlist), efficient,
beginner-friendly pipeline **ready to deploy on Roman GBTDS Season 1** and publish its rotators quickly.
Critical path: **ranker → Roman training data → Roman validation**. TESS multi-year handling is
off the critical path: one GBTDS season is essentially one contiguous ~70 d batch, so per-season P_max = R/2 ≈
low-20s d. Check this against the current survey design (T28).

**Sequence:**
1. **Close the SMARTS detour cheaply:** T24 + T25 + **T26** (detectability ceiling). Gives one honest SMARTS
   number. dev_medium full-year is at success 0.25 / wrong 0.09 / declined 0.65, and we can't yet tell how many
   of those declines are correct (undetectable stars).
2. **Ranker v0** (T10 → T11): rough and diagnostic, not production. dev_medium + a few-thousand-star training sample,
   quiet noise sources only, star-grouped splits, graded labels (T10 decision), **every candidate including
   gate-failed ones (gates become features)**, and non-rotator negatives from **T27**. Questions it answers: does it beat
   hand arbitration (per period bin), which features and methods matter (feeds T12), and how much data is enough
   (learning curve).
3. **Roman track** (**T28**): WG6 training sets, a Roman loader, a first scorecard. Single-season, so current
   machinery mostly applies. Adding a second data source once ranker v0 exists also gives the over-tuning test
   (train on one source, test on the other).
4. **Then:** **T29** (rotation-like / EB flags), T16h (batch combination), T12 (efficiency), D3 own butterpy
   multi-year sims (T16g), **T30** (mission-agnostic API) → T19b (packaging).

**Paused (the ranker should make most of it moot):** further hand-tuning of the gates: T16d, P1, P7, P21 (height-SNR).
Bug fixes and instrumentation are still fine.

**New training data, order:** (1) SMARTS noise-only non-rotators (T27, nearly free); (2) Roman WG6 sets (T28);
(3) own butterpy multi-year (D3/T16g); (4) Kepler/ground-based (wishlist). Introduce each once there is a model to
test it against, not before.

---

## Phase 0: infrastructure

*Done here (moved to `docs/TASKS_DONE.md`): T0.*

---

## Phase 1: data

*Done here (moved to `docs/TASKS_DONE.md`): T1, T2.*

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

*Done here (moved to `docs/TASKS_DONE.md`): T4, T14, T15, T18, T16a, T16b, T16c, T16e, T20, T21, T22, T23, T24, T25, T26, T27.*

All Phase 2 tasks use `method-improver`. Until T2 lands, judge on
`--synthetic 200 --seed 0`. After that, judge on dev_small, with dev_medium
for confirmation (ask Rae first).

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
- **Status:** ready (`coverage.searchable_range` is on main since the T16c merge, 2026-11-09). Phase 1 can use SMARTS now; multi-year tests need T16g.
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
  - ~~`claude_files/old/`; `batch_test_acf_fft_highpass.py` (open issue 9); `batch_test_guesses.py`;
    `run_debug_baseline*.py`~~: deleted 2026-11-11 (P30).
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

### T28. Roman GBTDS track: data, loader, first scorecard (roadmap step 3, 2026-11-09)
- **Agent:** start with a scoping session with Rae, then `method-improver`
- **Status:** ready for scoping. **Discuss with Rae first** (data access, which WG6 sets).
- **Why:** Roman is a must-have mission and the Season 1 paper is the deadline. It differs from TESS in cadence (~15 min),
  per-season baseline (~70 d, contiguous), crowding, faintness and stellar population.
- **Do:** (a) confirm the current GBTDS design (season length/count, cadence, filters, off-season sampling) and record it
  in a `docs/ROMAN_REFERENCE.md` like SMARTS_REFERENCE; (b) obtain the RGES PIT WG6 variable-star training sets, and check
  whether they include non-rotating variables and EBs; (c) a Roman loader that returns the pipeline's `(time, flux)`
  format; (d) a Roman dev set and scorecard dataset string; (e) a first scorecard run, with per-season `searchable_range`.
  Also note what Roman-specific features (e.g. crowding/blending) the ranker will need.
- **Done when:** a report with Roman baseline numbers and the gaps it exposes.

### T29. Rotation-like vs. non-rotating-variable and EB flags (roadmap step 4, 2026-11-09)
- **Agent:** `feature-researcher`
- **Status:** blocked on T27/T28 (needs negatives and variable-star examples)
- **Scope (Rae-approved):** two flags, not full variable classification: "rotation-like vs not" and "likely EB".
  **Flag EBs, don't discard them**: spotted/synchronized EBs are real rotators.
- **Candidate features:** coherence across batches (stable phase/amplitude = pulsator/EB; spots evolve; ties into
  T16h), periodogram peak sharpness and harmonic structure, folded shape (narrow dips, alternating depths → check
  P vs 2P; sawtooth for RR Lyrae/Cepheids), very short periods (δ Scuti), ACF decay.
- **Labels:** TESS: Gaia DR3 variability classes, the TESS EB catalog (Prša+2022), and T20's 206 contaminating variables.
  Roman: WG6 sets, since no prior catalogs exist there, so the internal flags must carry the load.

### T30. Mission-agnostic core API (roadmap step 4, 2026-11-09)
- **Agent:** `repo-curator` or a design session with Rae
- **Status:** ready, low priority, but **every new code path should follow the rule now**
- **Rule:** the core takes `(time, flux, flux_err)` + a **mission config** (cadence, P_min/P_max policy, batch-gap
  threshold, ...). Loading lives in per-mission adapters (`load_smarts_fits` is one; TESS, Roman, Kepler later). There
  is one obvious entry point (e.g. `spinspotter.measure(time, flux, mission=...)`). Then T19b's beginner-friendly
  packaging is mostly docs and examples rather than a refactor.

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

*Done here (moved to `docs/TASKS_DONE.md`): T9.*

---

## Phase 4: ranking model

### T10. Feature-table generation at scale
- **Agent:** `ranking-model-builder`
- **Status:** v0 done 2026-11-10 (`reports/2026-11-10_t10-ranker-feature-tables/`): builder, schema,
  `datasets/train_v0.csv`, dev_medium + train_1k tables. Full 3k table is Rae's local run
  (`tools/run_t10_train_tables.sh`); T27 negatives still to append. **Roadmap step 2 (2026-11-09):** aim at a rough v0 first; see the Roadmap. Include
  gate-failed candidates (gate results as features), quiet noise sources only, and T27's negatives.
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
- **Status:** v0 done 2026-11-10 (`reports/2026-11-10_t11-ranker-v0/`, branch `claude/t11-ranker-v0-2d9aed`, draft):
  `rank:ndcg` on train_1k beats hand arbitration modestly at matched training precision (dev_medium full-year
  success 0.248 → 0.267, assigned 0.148 → 0.188; no-decline top-1 0.441 / 0.328), loses at 10–30 d / > 30 d full-year
  because the global decline threshold over-declines right answers there (P24). 3k point pending Rae's table run
  (`tools/train_ranker.py --train .../train_3k --sizes 3000 --outdir runs/t11_ranker --resume`).
- **Goal:** XGBoost ranker, star-grouped splits, beats hand-tuned arbitration
  per period bin, with a calibrated "decline" threshold.
- **Decline operating point (Rae, 2026-11-10): deferred.** Working default: calibrate the decline threshold to
  match hand arbitration's OOF training precision (non-rotator:rotator mix stated alongside, currently 1:1). Every
  ranker/decline report must also plot the full purity–completeness (precision vs answer-rate) curve with candidate
  cutoffs marked. Don't list the precision target as an open question for Rae; she will set it once more of the
  pipeline is in place.
- **Revisit the hard period limits (Rae, 2026-11-10; see P23):** don't let the ACF-based R/2 cap remove candidates
  that another method (e.g. wavelet) can still support. Give the ranker `beyond_reach` candidates as rows, with
  `P_over_reach` as a feature, rather than relying on the comb fit's skip.
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

## Proposed (agents add here; Rae promotes)

*Done here (moved to `docs/TASKS_DONE.md`): P26.*

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

### P28. Headline card with fixed definitions (from the midway review, 2026-11-11)
- **Proposed by:** midway review (rec. 5).
- **Status:** in review (2026-11-12), branch `claude/p28-p29-headline-card-dcb5aa`, `reports/2026-11-12_headline-card/`.
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

### P29. Roman-season proxy on SMARTS (from the midway review, 2026-11-11)
- **Proposed by:** midway review (rec. 2).
- **Status:** first cut done (2026-11-12, `reports/2026-11-12_headline-card/`): `consecutive-3-5`, P ≤ 30 d =
  0.205 hand (T5) / 0.284 ranker v0 on dev_medium assigned. The true 70 d window (per-star-sectors manifest +
  `--max-span-days 70` dataset option, ~14 min) is specified but not run; Rae decides.
- **Idea:** until the WG6 sets exist (T28), report SMARTS performance on one contiguous ~70 d window: 5 consecutive
  sectors trimmed to 70 d, or the `consecutive-3-5` class as a first cut. Today `consecutive-3-5` at P ≤ 30 d
  gives 0.20 hand / 0.28 ranker v0 on dev_medium assigned. Cheap, and it says what Season 1 should expect from the TESS-trained model.

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
