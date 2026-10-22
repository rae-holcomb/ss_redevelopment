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
| `alias_metrics.py` | Shared ACF-window / phase-fold alias metrics (T9 features; moved out of `ml_features.py` in T18 so `comb_fit` can use them without a circular import). `half_lag_contrast_terms` drives the half-lag consistency gate. |
| `ml_features.py` | Flattens a fitted `EnsembleResult` into a ~120-column-per-candidate feature table for ranking-model training. `extract_candidate_features()` is the entry point. Includes coverage-context features (T16e, Section H of FEATURE_DOCUMENTATION) computed from the light curve's own finite-flux sampling. |
| `FEATURE_DOCUMENTATION.txt` | The definitive reference for every ML feature: name, definition, rationale, literature references, pipeline stage. Read this instead of re-deriving feature meanings from code. |
| `batch_test_guesses.py` | Production batch runner: evaluates every `guess_*` method's hit rate across many light curves, in parallel, with auto-discovery of new methods via introspection. |
| `batch_test_acf_fft_highpass.py` | Companion script isolating `guess_acf_fft_highpass`'s per-smoothing-window behavior specifically. Possibly redundant now that `batch_test_guesses.py` auto-discovers it too — unresolved, see open issues. |
| `evaluation.py` | **Scoring convention, single source of truth.** `REL_TOL=0.10`; `classify_period` → correct / half (P/2) / double (2P) / wrong / none; `best_outcome` for ranked candidate lists; `PERIOD_BIN_LABELS`. |
| `determinism.py` | Process setup for bit-reproducible runs (T14): `pin_blas_threads()` (call before importing numpy), `init_worker()` (ProcessPoolExecutor initializer), `blas_backend()`, `default_n_workers()` (one per physical P-core). Requires the OpenBLAS env `ss_openblas` (`environment.yml`); Accelerate is not reproducible. |
| `synthetic.py` | Synthetic spotted-star light curves (evolving spots, differential rotation, inclination, red noise, TESS-like orbit gaps and ramps) with known period, in the same format `load_smarts_fits` returns. `make_suite(n, seed)` is the benchmark population. `sectors=[...]` keeps only those sectors of the year, with `load_smarts_fits(sectors=...)` conventions (T16a). |
| `coverage.py` | Sector-coverage patterns (T16a): pattern parsing, coverage classes (`single`, `consecutive-/gapped-2`, `consecutive-/gapped-3-5`, `consecutive-/gapped-6-12`, `full`), real-star libraries per TESS year 1–7 plus a pooled default (`datasets/coverage/*_patterns.csv`, built by `tools/build_coverage_library.py`; `fit_to_year` maps 14–16-sector cycles into 13), `sample_patterns` for the scorecard's `--coverage`, and `coverage_descriptors` (baseline, longest contiguous span, largest gap, duty cycle), whose label-free time-sampling part is `sampling_descriptors` / `contiguous_segments` (shared with T16e's ML coverage features). |
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
3. Apply the reliability gates to every successfully-fit candidate:
   enough peaks survived (`n_peaks_used`, adaptive for few-peak long
   periods), fitted heights are genuinely positive bumps
   (`frac_positive_heights`), peaks are tall enough relative to noise
   (`min_mean_height_snr`), vertices inside their windows; plus (T18,
   2026-10-14) the half-lag consistency gate (every ACF h(kP) - h((k-1/2)P)
   term > 0, or mean >= 0.1) and, for few-peak candidates, raw ACF height
   at P >= 0.4. Then the factor-of-2 guard (half-tooth ratio < 0.7).
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
   **Scoped 2026-10-13 (Rae):** only the TESS ~13.7 d orbit-gap periodicity
   matters for now (task T8, low priority); revisit other cadences if the
   pipeline is applied to other missions (e.g. Roman).
8. *(Resolved 2026-10-13 by T4; kept as a placeholder so later issue
   numbers stay stable. See the changelog.)*
9. **`batch_test_acf_fft_highpass.py` may be redundant** now that
   `batch_test_guesses.py`'s auto-discovery already exercises
   `acf_fft_highpass` as one of its methods generically. Not deleted
   pending confirmation it doesn't do something the general runner
   doesn't.
10. **Low-SNR stars get confident wrong answers instead of declines.** *(Largely addressed
   2026-10-14 by T18's half-lag consistency gate: synth200 wrong 0.060 -> 0 on seed 0, 0.090 -> 0.015 on
   held-out seed 1; red-noise FP 21/60 -> 0/60. The remaining few are noted in the T18 report.)* On the
   synthetic baseline, 7/200 stars with SNR 1–3 and P = 18–36 d got a
   ~0.2–0.5 d period that passed all reliability gates. The gates don't seem to
   account for noise-driven short-lag ACF structure. Task T6.
   **Much worse on patchy coverage (T16a, 2026-10-14):** on single-sector synthetic stars, 36/100 get
   a 0.13–0.42 d wrong answer, at every SNR (10 of the 36 have SNR > 5), and most often when P exceeds
   the longest contiguous span. Task T16b.
   **T16b (2026-10-21):** after T18 these are declines, and the gates are *not* the main cause of the short-coverage
   over-decline. Declined stars' "correct" candidates mostly have no half-lag ACF structure (noise-like), and only
   3-6/200 per short class are structured combs blocked by the few-peak height guard, height SNR or vertex. Gate
   relaxations that are free on synthetic data admit more confident wrong answers than successes on dev_medium (T20).
11. **Differential rotation vs the P_eq label: addressed in scoring
   (2026-10-13), still open for ML labels.** SMARTS `PERIOD` is equatorial,
   but spot-latitude periods can differ a lot (shear up to +/-1, latitudes to
   80 deg). Scoring now counts a period inside the active band's range as the
   `diffrot` success outcome. The ranker's training label still uses P_eq only
   (see T10 in TASKS.md). Detecting and measuring differential rotation itself
   is future feature F1.
12. **Stage 2 arbitration loses correct candidates.** *(T18, 2026-10-14: the 7 seed-0 stars lost on
   selection to ~0.2 d noise combs are mostly rescued; what remains is Stage 1 recall (28 stars) and
   noise-level success candidates.)* Stage 1 recall (any
   method) is 0.84–0.98 per period bin, but final correctness is 0.63–0.84
   (0.65 in 10–30 d vs 0.95 recall). Motivates the ranker (T11) and possibly a
   quick hand-tuned fix (TASKS.md P1).
   T9 (2026-10-14) found most of the loss is in the gates: in 30 of 36 lost stars
   the success candidate never passed them. See TASKS.md P7.
13. **Bit-reproducibility is verified only on the Intel i5 (T14).** Under
   `ss_openblas`, runs are bit-identical, apart from a ~2e-11 wavelet-only residue
   on one star (TASKS.md P4). numpy's and scipy's macOS arm64 wheels use Accelerate,
   whose `ddot` was alignment-dependent here (on x86_64, so arm64 is untested).
   On Apple Silicon, build the env from conda-forge with `libblas=*=*openblas` and
   re-verify (TASKS.md P5).
14. **Long periods on short baselines: addressed 2026-10-13, edges remain.** The comb
   fit used to need >= 4 ACF peaks within B/3 (P <~ B/12), so most 1-2 sector TESS stars
   could never get a period above ~2-5 d. It now reaches P ~ 0.3 B (see changelog).
   Still open: (a) P = B/3 sits on the ACF coverage edge (min_valid_frac=0.3 -> ~0.67 B);
   (b) [addressed by T18's few-peak ACF-height guard: red-noise few-peak FPs 0 on all pb_scan sets] few-peak (`few_peaks=True`) answers had no extra noise guard, and red noise produced
   long few-peak false positives (1/60 eval, 3/30 dev seeds); (e) runtime +30-60%; (c) the factor-2 guard's 0.7 ratio
   threshold is hand-set; (d) `guess_acf_fft` misbehaves on long ACFs (hence two-view Stage 1).
15. **The comb's lag-0 tooth is an implicit noise filter (found 2026-10-13).** For short
   candidate periods, `_grid_search_t0` puts the n=0 tooth on the trivial lag-0 ACF peak. Removing
   it everywhere made 5 low-SNR synth200 stars return confident ~0.2-0.5 d answers. This
   is likely part of open issue 10's mechanism, and any T6 fix should account for it.
16. **SMARTS galaxy-noise reuse and corrupt galaxies (found 2026-10-15, T1).** Each noise galaxy is
   reused ~800 times; there is no galaxy ID, but `Simulation Number` < 800k vs >= 800k separates two
   disjoint galaxy pools, so train/holdout can be galaxy-disjoint (T2). Dev vs holdout separation
   needs per-file galaxy IDs (TASKS.md P15). ~12% of galaxies contain corrupt orbits (relative flux
   far from 1); `load_smarts_fits` drops nearly all of it, but raw-flux and WAVELET consumers don't.

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
- **2026-10-22: D4 decided (coverage-aware searchable range).** Searchable limit is **P_max = R/2**, with R the ACF reach (valid-pair fraction >= 0.3; ~0.69x the longest contiguous segment; it does not bridge sector gaps). On T16b's data, success is flat to P/R = 0.5 and zero beyond (dev_medium 0/323). Start: `searchable_range()` per contiguous batch; Stage 1 uncapped, with beyond-R/2 candidates tagged `beyond_reach` and skipped by the fit; the range is recorded for D1; a P/R ranker feature (T16c). End: per-star `searchable` = low end of the success band <= R/2, with R frozen in `evaluation.py` (independent of pipeline defaults); report all-star and searchable-subset metrics, plus wrong/decline rates on unsearchable stars, and propose searchable goals alongside the existing ones (T23). Wrong answers on unsearchable stars are mostly short false detections, which a search limit doesn't fix. Multi-season/clustered sectors: batch-wise search with testability-aware cross-batch comparison vs a stacked within-batch ACF (T16h; the GP option is optional). See TASKS.md D4.
- **2026-10-22: Rae's decisions on T20 phase 2** (all as recommended): flag every training/evaluation table with the
  per-file noise-source flags; exclude contamination (known-variable + crowded) and, for now, unexplained
  source-specific sources from ranker training (197 sources, 114,829 files = 11.5%); keep the 8 common-mode
  systematics; keep the current `periodic` flags but store signal-strength metrics; new tasks T24 (`noise_period`
  scoring outcome) and T25 (dev_medium quiet-noise subset metrics); T16b's replay unblocked; D3 points at the flags.
  Open: Rae asks Zach about the 242 near-LMC sources.
- **2026-10-22: T20 full independent-FFI check done; T20 merged.** Of 150 testable (P < 5 d) sources, Gaia
  known-variable contamination is confirmed in independent QLP/TASOC/ELEANOR-LITE light curves 38/60 (p = 4e-5 vs a
  4/26 shuffled-period quiet null), crowded 13/34 (p = 0.05), but `source_specific_unexplained` only 4/25 (chance):
  their signal may be specific to Zach's reduction. Exclusion kept, since training uses the SMARTS light curves.
  `noise_source_flags.csv.gz` on the drive now carries `exclude_from_training` and the signal-strength metrics (T24/T25 read it).
