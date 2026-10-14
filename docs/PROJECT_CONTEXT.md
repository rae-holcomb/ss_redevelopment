# Project Context: SpinSpotter Redevelopment

**Purpose of this document**: bring a new chat in this project up to speed
quickly. It covers (1) what the code does and how it's organized, (2) the
reasoning behind key design decisions, and (3) open problems and next
steps. It is a **living document** — see "How to maintain this document"
at the end. If you are a new chat picking this up: read this whole file
before doing anything else, then check `FEATURE_DOCUMENTATION.txt` (for
the ML feature set) or the code's own docstrings for implementation-level
detail this file deliberately doesn't duplicate.

Companion document: `METHODOLOGY_NOTES.md` is the human-facing (Rae's)
notes-to-self for an eventual paper methods section. It covers the same
algorithm from a scientific-writeup angle rather than a
software-engineering angle. Worth reading if you need to understand the
*scientific* motivation for a design choice in more depth than this file
gives.

---

## 1. What this project is

Rae is rewriting **SpinSpotter** (Holcomb et al. 2022) from scratch. The
original tool measures stellar rotation periods by fitting a single
parabola to a pre-selected cutout of the light curve's autocorrelation
function (ACF), centered on an initial period guess. That approach fails
when the initial guess is bad (period unknown a priori, or the star's
true period is far from any reasonable guess) or when the light curve's
structure is complex (multiple periodicities, weak/noisy signal, gaps).

The rewrite's core idea: instead of needing a good initial guess to fit
*one* parabola, generate a broad pool of *candidate* periods from several
independent, cheap methods, then fit a **joint comb of evenly-spaced
parabolae** (one per expected ACF peak) to *every* candidate, and let the
fit quality itself arbitrate which candidate is right. No single
candidate-generation method needs to be reliable on its own — it only
needs to not leave the right answer off the list.

The eventual goal is to run this pipeline across ~1,000,000 simulated
SMARTS light curves (real TESS backgrounds + injected synthetic spot
signals, known true periods) to generate training data for a gradient-
boosted ranking model that picks the best candidate period more robustly
than the current hand-tuned reliability gates.

---

## 2. Code map

All files live in a single flat package (`claude_files/` in the
`rae-holcomb/ss_redevelopment` GitHub repo).

| File | Role |
|---|---|
| `acf_utils.py` | Gap-aware ACF via masked-FFT autocorrelation. Handles NaN-filled cadences without interpolating or discarding data. |
| `guesses.py` | **Stage 1**: candidate-period generation. Every `guess_*` function. Never touches fitting or computes a phase. |
| `comb_fit.py` | **Stage 2**: joint comb-of-parabolae fitting and multi-candidate arbitration. `fit_rotation_period` is the main entry point. Imports `InitialGuess` from `guesses.py`. |
| `preprocessing.py` | Raw SMARTS FITS → pipeline-ready `(time, flux)`. Re-detects the original per-orbit gaps (SMARTS ships gap-*interpolated*), lets you drop bad orbits and/or degrade to a sparser TESS-like sector pattern, regrids to even cadence. `load_smarts_fits()` is the one-call entry point. |
| `plotting.py` | Diagnostic plots mirroring every pipeline stage. |
| `ml_features.py` | Flattens a fitted `EnsembleResult` into a ~100-column-per-candidate feature table for ranking-model training. `extract_candidate_features()` is the entry point. |
| `FEATURE_DOCUMENTATION.txt` | The definitive reference for every ML feature: name, definition, rationale, literature references, pipeline stage. Read this instead of re-deriving feature meanings from code. |
| `batch_test_guesses.py` | Production batch runner: evaluates every `guess_*` method's hit rate across many light curves, in parallel, with auto-discovery of new methods via introspection. |
| `batch_test_acf_fft_highpass.py` | Companion script isolating `guess_acf_fft_highpass`'s per-smoothing-window behavior specifically. Possibly redundant now that `batch_test_guesses.py` auto-discovers it too — unresolved, see open issues. |
| `evaluation.py` | **Scoring convention, single source of truth.** `REL_TOL=0.10`; `classify_period` → correct / half (P/2) / double (2P) / wrong / none; `best_outcome` for ranked candidate lists; `PERIOD_BIN_LABELS`. |
| `synthetic.py` | Synthetic spotted-star light curves (evolving spots, differential rotation, inclination, red noise, TESS-like orbit gaps and ramps) with known period, in the same format `load_smarts_fits` returns. `make_suite(n, seed)` is the benchmark population. |
| `example_usage.py`, `example_messy_data.py`, `example_from_fits.py` | Worked examples of the pipeline at increasing levels of real-world messiness. |

### `guesses.py` in more detail

Seven candidate-generation methods. Three run by default via
`gather_initial_guesses(..., methods=(...))`; four are opt-in only
(never run unless explicitly requested):

- **Always on**: `guess_pairwise_histogram` (ACF peak-spacing histogram,
  ranked by pairwise-support count), `guess_lombscargle` (light curve's
  own LS periodogram), `guess_acf_fft` (FFT of the ACF).
- **Opt-in**: `guess_wavelet` (Global Wavelet Power Spectrum, iterative
  Gaussian peak extraction — needs gap-free flux, more expensive),
  `guess_lombscargle_short` / `guess_acf_fft_short` (same methods,
  restricted to a short-period search band, added to address <10-day
  period underperformance), `guess_acf_fft_highpass` (high-pass filters
  the flux at several smoothing-window scales before computing the ACF,
  to expose short-period signal a longer-timescale trend would otherwise
  dominate).

Every `guess_*` function returns `list[InitialGuess]` (candidate period,
originating method, rank/strength within that method's own ranking, and a
method-specific `info` dict carrying the raw spectrum/histogram for later
inspection or feature extraction) and **never computes a phase (`t0`) or
does any curve fitting** — that's entirely Stage 2's job.

### `comb_fit.py` in more detail

`fit_rotation_period(acf_lags, acf, initial_guesses)` is where the real
evaluation happens:

1. Sanity-filter and deduplicate candidates (tracking which raw guesses
   got merged into which survivor — see decision log below).
2. For each surviving candidate, seed a phase via a coarse grid search
   (`_grid_search_t0`), then run the actual joint least-squares fit
   (`_fit_single_candidate`): a shared `P`/`t0` with every window's
   parabola center algebraically tied to them via `lmfit` expressions,
   robust loss (`soft_l1`), iterative RANSAC-style rejection of
   badly-fitting windows.
3. Apply three reliability gates to every successfully-fit candidate:
   enough peaks survived (`n_peaks_used`), fitted heights are genuinely
   positive bumps (`frac_positive_heights`), peaks are tall enough
   relative to noise (`min_mean_height_snr`).
4. Among candidates passing all three gates, pick the lowest reduced
   chi-squared. **If nothing passes, return `success=False` with an
   explanation rather than a forced answer** — see decision log.

`assess_rotation_candidate` optionally adds two more diagnostics if given
`time`/`flux`/`acf_lags`: `phase_dispersion_stat` (Stellingwerf 1978 PDM,
evaluated on the light curve directly rather than the ACF — genuinely
different failure mode, not fooled by the half-period alias case that
fools redchi) and `acf_peak_prominence_diagnostics`
(G_ACF/H_ACF-style raw-ACF peak prominence, ROOSTER-inspired).

Outside `claude_files/` (added 2026-10-13 for agent-driven development;
see `docs/AGENT_WORKFLOW.md`): `tests/` (pytest), `benchmarks/`
(`scorecard.py`, `check_goals.py`, `goals.json`, `baselines/`), `reports/`
(one folder per task), `tools/` (`new_report.py`, `run_notebook.py`),
`.claude/agents/` (agent roles), `.claude/skills/write-report/`,
`docs/TASKS.md` (task board).

---

## 3. Key design decisions and why (chronological, high level)

This section is deliberately terse — it's a map of *what was decided*,
not a full justification. Search this project's conversation history for
the full reasoning behind any entry if you need it.

- **Two-stage split (candidate generation vs. fitting/arbitration)**,
  rather than each method self-validating: earlier versions had each
  `guess_*` function cross-check its own candidates against the ACF via a
  cheap "comb score." That score was gameable (a candidate with very few
  "teeth" in range could win on weak evidence) and made each function
  harder to reason about independently. Splitting "propose" from
  "evaluate properly" fixed both problems.
- **Honest failure over a forced answer**: `fit_rotation_period` can
  return `success=False`. A best-available answer that didn't clear basic
  plausibility checks is often worse than no answer, especially since
  this pipeline is meant to feed automated, unsupervised large-scale
  processing where a silently-wrong period is far more costly than a
  flagged non-detection.
- **`frac_positive_heights` gate**: added after discovering a comb could
  achieve deceptively low reduced chi-squared by fitting parabolae to the
  ACF's smooth near-zero-lag decay envelope rather than genuine periodic
  peaks — technically a great fit, not a real signal.
- **Gap-aware ACF via masked-FFT autocorrelation** (`acf_utils.py`)
  instead of interpolating gaps or restricting to one contiguous segment:
  interpolating a multi-week TESS downlink gap injects a fabricated
  trend; restricting to one segment is often shorter than a single
  rotation period for realistically gappy data. The masked approach
  (zero out gaps, separately autocorrelate the validity mask to get a
  correct per-lag pair count, divide) handles this correctly in
  O(N log N) with no interpolation artifact.
- **Adaptive ACF-peak-finding prominence** (5x the standard deviation of
  the ACF's second difference) rather than a fixed threshold: a fixed
  value doesn't generalize across targets with very different noise
  scales — it let ~7000 spurious noise wiggles through on one real TESS
  light curve while being simultaneously too strict for a noisier SMARTS
  target.
- **`guess_pairwise_histogram` ranks by raw pairwise-support count**
  (not a cross-validated "comb score"): with m found peaks, the true
  fundamental spacing is supported by up to (m-1) pairs, a harmonic 2P by
  only (m-2), etc. — support count strictly decreases with multiple, so
  this is inherently resistant to harmonic ambiguity without needing any
  extra cross-checking, and is simple enough to explain and debug (this
  replaced an earlier, more complex "coverage fraction" scheme that had
  its own bugs and was hard to reason about).
- **Cross-candidate/cross-method agreement is one of the strongest
  informal signals found for "this is probably the real period"** across
  every real-data test case in this project (SMARTS, TESS, FITS files)
  — the correct period was rarely any single method's #1 pick, but
  consistently had support from more than one method. This motivated
  `n_agreeing_candidates`/`n_agreeing_methods` and
  `n_duplicate_guesses`/`n_contributing_methods` as ML features (see
  `FEATURE_DOCUMENTATION.txt` Section E), and motivates favoring
  ensemble/cross-method approaches generally over single-method tuning.
- **Pre-fit deduplication in `fit_rotation_period` used to silently
  discard cross-method agreement**: candidates within `dedup_rel_tol`
  were merged before fitting with no record of what got merged. Fixed by
  having `CandidateResult` track `n_duplicate_guesses`/
  `contributing_methods` from the merge step itself.
- **LombScargle default `samples_per_peak` lowered from 10 to 1**:
  candidate generation only needs a coarse estimate (the joint fit
  refines it), so a finely-oversampled frequency grid was pure wasted
  cost — ~5x speedup confirmed with no meaningful accuracy loss.
- **Zero-fill FFT vs. LombScargle for speed**: tested empirically (not
  just reasoned about) against real gappy data, including checking the
  gap pattern's own "spectral window" for coherent aliasing structure.
  Found low risk for the specific (irregular) SMARTS/TESS gap patterns
  tested, but flagged that a genuinely *regular*, periodic gap pattern
  (e.g. strict ground-based day/night cadence) is the real stress test
  and hasn't been tried — see open issues.
- **ML feature set deliberately over-generated** (~100 columns/candidate
  as of this writing), including intentional redundant families (e.g.
  rank + fractional-power + significance per spectral method): the plan
  is to prune based on real feature-importance analysis once training
  data exists, not to guess in advance which features matter.
- **Training-label circularity avoided by construction**: training labels
  come from SMARTS's known *injected* periods, not from periods measured
  on real TESS data by other pipelines — avoids training a model to
  reproduce potentially-biased existing measurements on the very data
  this pipeline is meant to analyze independently. (SMARTS light curves
  = real TESS backgrounds + injected synthetic signal, so noise/gap
  characteristics are realistic without the label itself being circular.)
- **`comb_fit.py`/`guesses.py` module split**: `comb_fit.py` had grown to
  2109 lines. Split along the Stage 1/Stage 2 boundary already implicit
  in the code's own design. One non-obvious call: `comb_score`,
  `default_comb_weight`, `_grid_search_t0`, `_teeth_count` were
  positioned with the guess-side code but are actually only ever called
  by `fit_rotation_period` (phase-seeding/sanity-filtering right before a
  fit) — moved to `comb_fit.py` to match actual usage, not just their old
  physical location in the file.

---

## 4. Open issues and to-do items

Roughly in priority order as of this writing — re-prioritize freely.

1. **Short-period (<10 day) recovery is still weak.** On an 8-file hard
   test batch, all methods hit 0/8 at strict 3% tolerance even after
   adding the short-period-focused methods; at a looser, more
   operationally realistic 15% tolerance (matching the joint fit's own
   ±30% refinement window), `guess_acf_fft_highpass` reaches 5/8 — best
   of six methods tested, but not solved. One file in that batch
   (~5.6 points/cycle) appears to be a fundamental Nyquist/cadence
   resolution wall, not fixable by search strategy at all.
2. **No batch run has been done at real scale yet.** Everything above has
   been validated on a small, hand-picked set of ~10-20 known-hard files.
   `batch_test_guesses.py` exists and works, but hasn't been pointed at
   anything close to the full ~1M-star SMARTS corpus.
3. **No ranking-model training code exists yet.** The feature set,
   labeling strategy, and grouped-ranking-model architecture (XGBoost
   `rank:pairwise`/`rank:ndcg`, grouped by star, avoiding per-row splits
   that would leak information between candidates of the same light
   curve) were designed and discussed at length but never implemented.
4. **Feature pruning/importance analysis** is planned but blocked on
   having real training data to run it against (permutation importance
   on a star-grouped held-out split; correlation analysis within the
   deliberately-redundant feature families first, then across families).
5. **Per-star pipeline cost** (~6-11 seconds, dominated by the LS
   periodogram and the joint fit) is a real concern at 1M-star scale.
   Partially addressed (5x LS speedup) but not solved; no parallelization
   strategy has been chosen for the eventual full-corpus run beyond what
   `batch_test_guesses.py` already does for its own smaller test batches.
6. **Multi-sector support is incomplete.** `R_var` (in `ml_features.py`)
   accepts a `sectors` parameter for proper per-sector-then-combine
   treatment; the other star-level variability features (`flicker_std`,
   `flux_std`, skewness/kurtosis, `duty_cycle`, `von_neumann_eta`,
   `fliper_band_*`) don't yet, pending Rae adding multi-sector *input*
   handling to the main pipeline (currently light curves are passed in as
   a single already-sector-combined timeseries).
7. **Zero-fill-FFT-vs-LombScargle aliasing risk for regularly-gapped
   data** was flagged as untested — the empirical checks so far used
   real SMARTS/TESS gap patterns, which turned out to be irregular enough
   not to trigger the classic aliasing failure mode. A synthetic light
   curve with a strictly periodic gap pattern (e.g. rigid ground-based
   day/night cadence) would be the real stress test, and hasn't been run.
8. **`guess_pairwise_histogram` raises outright** (doesn't return an
   empty list, raises an exception) on some ultra-short-period light
   curves where fewer than 2 ACF peaks can be found. Never circled back
   to make this fail gracefully.
9. **`batch_test_acf_fft_highpass.py` may be redundant** now that
   `batch_test_guesses.py`'s auto-discovery already exercises
   `acf_fft_highpass` as one of its methods generically. Not deleted
   pending confirmation it doesn't do something the general runner
   doesn't.
10. **Low-SNR stars get confident wrong answers instead of declines.** On the
   synthetic baseline, 7/200 stars with SNR 1–3 and P = 18–36 d got a
   ~0.2–0.5 d period that passed all reliability gates. The gates don't seem to
   account for noise-driven short-lag ACF structure. Task T6.
11. **Differential rotation vs the P_eq label: addressed in scoring
   (2026-10-13), still open for ML labels.** SMARTS `PERIOD` is equatorial,
   but spot-latitude periods can differ a lot (shear up to +/-1, latitudes to
   80 deg). Scoring now counts a period inside the active band's range as the
   `diffrot` success outcome. The ranker's training label still uses P_eq only
   (see T10 in TASKS.md). Detecting and measuring differential rotation itself
   is future feature F1.
12. **Stage 2 arbitration loses correct candidates.** Stage 1 recall (any
   method) is 0.84–0.98 per period bin, but final correctness is 0.63–0.84
   (0.65 in 10–30 d vs 0.95 recall). Motivates the ranker (T11) and possibly a
   quick hand-tuned fix (TASKS.md P1).

---

## 5. How to maintain this document

This file is meant to accumulate across chats in this project, not be
rewritten from scratch each time. When you (a chat in this project) make
a notable decision, resolve an open issue, or discover something future
chats should know:

- **Add a dated entry to the changelog below**, newest at the bottom.
  Keep entries terse (a few lines) — this is a pointer/index, not a full
  writeup. If the full reasoning matters, it belongs in code docstrings,
  `FEATURE_DOCUMENTATION.txt`, or `METHODOLOGY_NOTES.md`, and the
  changelog entry can just say so.
- **If an open issue from Section 4 gets resolved**, move it out of
  Section 4 and note the resolution in the changelog (don't just delete
  it silently — the history of "we tried X, it didn't work, here's why"
  is often as valuable as the eventual fix).
- **If a new open issue is discovered**, add it to Section 4.
- **If the code map (Section 2) or a design decision (Section 3)
  meaningfully changes**, update that section directly rather than
  leaving it stale and only noting the change in the changelog — Section
  2/3 should always reflect current reality; the changelog is the
  history of how we got here.

### Changelog

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
