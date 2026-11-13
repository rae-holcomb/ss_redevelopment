# Task board

Each task is sized for one agent session. Rae owns this file's priorities
and acceptance criteria. Agents update only the **Status** line of their
own task (and add follow-up tasks to `docs/TASKS_PROPOSED.md`, never directly to the
active list). **Read only your own task's section** (search for its `### T##.` heading), plus the Roadmap if you
are a coordinator; don't read the whole board.

Status values: `blocked` · `ready` · `in progress (<branch>)` · `in review (<report folder>)` · `done (<report folder>)`

Done tasks are in `docs/TASKS_DONE.md` (moved 2026-11-11, P30). **Archive at merge, every time:** when a task
or proposal (T##, P##) is merged, the coordinator sets its status to done and moves its section verbatim to
`docs/TASKS_DONE.md` in the same commit (see `docs/AGENT_WORKFLOW.md`). A done task left here costs every
coordinator session tokens; this file regrew from ~45 KB to 84 KB in two days (2026-11-11 to 11-13) without it.

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
    Untrack them (and decide on history, see T19b). **Untracked 2026-11-13; history kept.**
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
    Deferred to here on 2026-11-13 (Rae): `results_products/batch1/candidates.csv` (90 MB) is still in history,
    most of `.git`'s ~163 MB. Purging rewrites nearly every commit ID (31 IDs are cited in 99 files: reports,
    docs, baselines' `git_commit`), so commit an old→new ID map and update the living docs, and do it when no
    agent session is running, rewriting all local and GitHub branches. Consider the ~12 MB of other
    `batch1/*.csv` at the same time.
  - Scan for absolute local paths, secrets, and personal info.
  - Set up CI to run the test suite on push.
- **Done when:** a fresh clone installs and passes its tests with no local paths and no
  oversized files, and the README gets a new user to a measured period in a few commands.

### T28. Roman GBTDS track: data, loader, first scorecard (roadmap step 3, 2026-11-09)
- **Agent:** start with a scoping session with Rae, then `method-improver`
- **Status:** scoping in review (`reports/2026-11-12_wg6-dataset-scoping/`). Raw WG6 set is archival; the WG6 Roman-resampled set (received 2026-11-11, improved noise recommended) is profiled in Part 2. It has no ROT; FL (TESS M-dwarf templates) are the only Roman-cadence rotators. Next steps: P34 (ROT), P35 (loader + manifest), P36 (negatives).
- **Rae's decisions (2026-11-13):**
  - ROT resampling to Roman cadence: **we do it ourselves** (P34).
  - Noise: the **improved** variant.
  - EBs: a **flagged class**, not plain negatives (consistent with T29).
  - FL's 54 M-dwarf rotators are **not enough**. Build a rotator set with a real mix of spectral types and rotation
    periods (P34 scope).
  - GBTDS design: **3 early + 3 late seasons** is correct. For development use **only the 3 early seasons**. Late-season
    data is years away, so it isn't a pipeline design concern now.
  - Still open: how to handle long ROT periods.
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
- **Status:** done 2026-11-12, awaiting Rae's review (`reports/2026-11-12_t12-speed-up/`, branch `claude/t12-speed-up-2a627c`).
  Six bit-identical changes, 1.5–1.6× per star (production path 1.31×); harness `benchmarks/profile_pipeline.py`,
  equivalence tests `tests/test_t12_speedups.py`, v6 baselines committed. Follow-ups: P4 (updated), P38.
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

### T31. Wavelet determinism fix + result-changing speed-ups (comb fit, CWT) (Rae, 2026-11-13; from P4, P38)
- **Agent:** `performance-optimizer` (with method-improver judgement for the comb-fit variants). **Run overnight, machine alone.**
- **Status (2026-11-13): merged** (`fd6e716`). Rae: default comb fit `local_screen2`, default CWT `lean`, `_v6w`
  replaces `_v6`; dev_medium rerun on the new defaults committed as `_v7` (the judging baseline). Follow-ups: P39, T32.
- **Status before merge: done, awaiting Rae's review** (`reports/2026-11-13_t31-comb-fit-speed/`, branch
  `claude/t31-comb-fit-speed-e7e5bb`; Part A alone on `claude/t31-partA-wavelet-determinism`). Part A: wavelet
  bit-reproducible (np.exp + complex np.abs SIMD flips), candidate `_v6w` baselines. Part B: recommended comb variant
  `local_screen2` = main's answers at 2x production-path speed; `local` 6x but more assigned wrongs; CWT `lean` free 1.25x.
  Variant choice is Rae's.
- **Part A (P4):** make `_morlet2_kernel` / the CWT bit-reproducible (deterministic Morlet envelope). Test: repeated
  in-process calls give identical bytes. One dev_medium run in both modes; only wavelet fields may move, no outcome change.
  New baselines `_v6w` (or similar) after Rae's OK.
- **Part B (P38), Rae's latitude:** the comb fit may change in its details. It must stay true to the idea of finding
  repeated ACF peaks and characterizing their shapes, sizes and locations, but the current global least-squares comb schema
  isn't sacred. Explore:
  - solver settings (tolerances, `tr_solver`, Jacobian, points per window);
  - structural alternatives, e.g. independent per-peak local fits combined afterwards;
  - a cheaper CWT (fewer scales, smaller transform length).
  Score each variant on dev_medium full-year + assigned (~12 min per variant at 4 workers).
- **Done when:** a report with an accuracy-vs-seconds-per-star frontier (success by period bin, per variant), failure-mode
  examples for any variant that loses stars, and a recommended variant with `check_goals.py --baseline` before/after.
  Don't adopt a redesign silently: the chosen variant is Rae's call.
- **Sequencing:** the ranker-table rebuild (P31) waits for this, since ranker features come from the comb fit and wavelet.

---

### T32. Make the cheap `local` comb solver robust to drifting ACF peaks (Rae, 2026-11-13; from T31, P40)
- **Agent:** `method-improver`. Builds on T31 (merged to main 2026-11-13; baseline `_v7`; `SS_COMB_VARIANT=local`,
  `comb_fit._fit_comb_local`). Rae adopted `local_screen2` as the default; this task aims to make the ~6x-faster `local`
  usable on its own.
- **Problem:** `local` gains 33 dev_medium stars (29 are main's `frac_vertex_in_window` declines, many obvious rotators whose
  later ACF peaks drift), but its vertex gate rejects 9% of candidates vs 60% under main, so wrong rises on assigned
  coverage (0.055 -> 0.066; 28 stars none -> wrong).
- **Explore all five (Rae's own ideas matched 1 and 2):**
  1. sequential peak tracking (search near vertex_{n-1}+P, not n*P);
  2. vertex tolerance growing with n (sqrt(n), n, or peak width);
  3. a quadratic drift term in the closed-form line fit, with `a` as a candidate ranker feature;
  4. height/SNR weighting of peaks in the IRLS fit;
  5. replacing `frac_vertex_in_window` with a residual-scatter statistic.
- **First step (no new scorecards):** plot vertex residual vs n for (a) the 29 gains, (b) the 28 none->wrong stars,
  (c) shared-correct stars (`reports/2026-11-13_t31-comb-fit-speed/data/changed_stars_*`).
- **Done when:** each idea is a `SS_COMB_VARIANT` switch scored on dev_medium full-year + assigned with `check_goals
  --baseline`; a speed/success/wrong comparison against `lsq`, `local` and `local_screen2`; residual-vs-n and gained/lost
  example figures; a recommendation. Machine-safety rules apply (`nice -n 15`, 3 workers, one heavy job at a time).

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

## Proposed and open discussions (moved out 2026-11-13)

- Agent-proposed follow-ups (P##): `docs/TASKS_PROPOSED.md`. Add new ideas there.
- Open discussions for Rae (D##): `docs/TASKS_DISCUSSIONS.md`.
- Done tasks: `docs/TASKS_DONE.md`.
