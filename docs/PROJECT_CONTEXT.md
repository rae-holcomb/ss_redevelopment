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

**Broader goals (Rae, 2026-11-09).** SpinSpotter v2 must work on **TESS and the Roman GBTDS** (Kepler bonus light
curves and ground-based surveys are a wishlist), stay efficient at ~millions of light curves, flag non-rotating
periodic variables and eclipsing binaries, and be well documented with a low start-up cost for beginners.
**Deadline: deployable by the end of Roman GBTDS Season 1**, for a quick Season 1 rotators paper. SMARTS is the
first training set, not the only one: it has no non-rotators or false positives, and over-tuning to it is a standing
risk. The agreed roadmap and sequence are at the top of `docs/TASKS.md`.

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
| `ranking_model.py` | **Ranker v0** (T11): XGBoost learning-to-rank (`rank:ndcg`) over the T10 feature tables, one query per star, noise-TIC-grouped CV, and a decline threshold calibrated on out-of-fold training predictions. Trained by `tools/train_ranker.py`. |
| `evaluation.py` | **Scoring convention, single source of truth.** `REL_TOL=0.10`; `classify_period` → correct / half (P/2) / double (2P) / wrong / none; `best_outcome` for ranked candidate lists; `PERIOD_BIN_LABELS`. |
| `determinism.py` | Process setup for bit-reproducible runs (T14): `pin_blas_threads()` (call before importing numpy), `init_worker()` (ProcessPoolExecutor initializer), `blas_backend()`, `default_n_workers()` (one per physical P-core). Requires the OpenBLAS env `ss_openblas` (`environment.yml`); Accelerate is not reproducible. |
| `synthetic.py` | Synthetic spotted-star light curves (evolving spots, differential rotation, inclination, red noise, TESS-like orbit gaps and ramps) with known period, in the same format `load_smarts_fits` returns. `make_suite(n, seed)` is the benchmark population. `sectors=[...]` keeps only those sectors of the year, with `load_smarts_fits(sectors=...)` conventions (T16a). |
| `coverage.py` | Sector-coverage patterns (T16a): pattern parsing, coverage classes (`single`, `consecutive-/gapped-2`, `consecutive-/gapped-3-5`, `consecutive-/gapped-6-12`, `full`), real-star libraries per TESS year 1–7 plus a pooled default (`datasets/coverage/*_patterns.csv`, built by `tools/build_coverage_library.py`; `fit_to_year` maps 14–16-sector cycles into 13), `sample_patterns` for the scorecard's `--coverage`, and `coverage_descriptors` (baseline, longest contiguous span, largest gap, duty cycle), whose label-free time-sampling part is `sampling_descriptors` / `contiguous_segments` (shared with T16e's ML coverage features). |
| `example_messy_data.py`, `example_from_fits.py` | Worked examples: the pipeline on a gappy, noisy light curve, and end to end from a raw SMARTS FITS file (load → ACF → candidates → comb fit → ML features → plots). (`example_usage.py`, a synthetic-only demo, no longer exists; its last copies were in `claude_files/old/`, deleted 2026-11-11.) |

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

**Peak ranking (T5, 2026-11-11):** `guess_lombscargle` and `guess_acf_fft` (and the
`*_short`/`_highpass` wrappers) no longer keep the top-N peaks by raw power. They keep
half by raw power and half by power over the local continuum (median peak power
within ±0.5 dex; `guesses._select_peaks`, `rank_by="hybrid"`); `rank_by="power"` is the
old behaviour. On full-year baselines raw-power ranking let red-noise peaks crowd every
sub-day period out of the top 10. `rank` is therefore position in this list, not
power order; `strength` is still the peak's power.

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

Outside `claude_files/` (see `docs/AGENT_WORKFLOW.md`): `tests/` (pytest), `benchmarks/`
(`scorecard.py`, the standard benchmark and the batch runner for any manifest; `check_goals.py`, `goals.json`,
`baselines/`), `reports/` (one folder per task), `tools/` (`build_feature_tables.py` for ranker tables,
`train_ranker.py`, `headline_card.py` (P28 fixed-definition headline card), dataset/side-table builders, `audit/` for data-audit scripts, `new_report.py`, `run_notebook.py`),
`.claude/agents/` (agent roles), `.claude/skills/write-report/`, `docs/TASKS.md` (open tasks),
`docs/TASKS_DONE.md` (finished tasks), `docs/CHANGELOG_ARCHIVE.md` (changelog before 2026-10-22).
The old per-method batch runners (`batch_test_guesses.py`, `batch_test_acf_fft_highpass.py`,
`run_debug_baseline*.py`) were retired 2026-11-11 (P30): the scorecard's `per_method.csv` gives per-method Stage-1
recall and ranks, and its `per_star.csv` the full-pipeline answer.

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
   **Update (T5, 2026-11-11):** on dev_medium the full-year <1 d Stage-1 deficit was
   top-10 crowding by red-noise peaks, now fixed (union recall 0.44 → 0.70). The
   remaining bottleneck is arbitration: `half_lag_consistency` rejects the rescued
   candidates (P32), so the gain reaches answers via the ranker (P31).
2. **No corpus-scale (~1M-star) run yet, and none planned soon** *(refreshed 2026-11-11)*. Largest runs so far:
   dev_medium (1,000 stars) in both coverage modes, `train_3k` (3,000 stars) and 4,708 non-rotator negatives
   through `tools/build_feature_tables.py`, all run locally by Rae. The ranker's learning curve is flat from 1k to
   3k stars (`reports/2026-11-11_ranker-3k-learning-curve/`), so corpus-scale tables are parked until the
   candidate pool and features settle (midway review rec. 7). The real driver for scale is Roman deployment.
3. *(Resolved 2026-11-10 by T10/T11; see the changelog. Ranker v0 exists: `ranking_model.py`,
   `tools/train_ranker.py`. Its open part, the decline stage, is issue 12.)*
4. **Feature pruning** *(refreshed 2026-11-11)*. No longer blocked on data: ranker v0 (T11) gives a first
   importance ranking (alias features and the wavelet/LS spectra matter most; gate flags and the
   `acf_fft_highpass`/`pairwise_histogram` spectra least). A systematic pass (permutation importance on
   noise-TIC-grouped folds, within-family correlation first) hasn't been done; the deliberately redundant families
   stay until it is.
5. **Per-star cost at scale** *(refreshed 2026-11-11)*. About 1.16 s/star on the i5 at 10 workers (2026-10-13
   timing), i.e. hundreds of CPU-hours per million stars. Parallelism is settled: `ProcessPoolExecutor` with BLAS
   pinned to 1 thread per worker (`determinism.py`), as in the scorecard and the table builder. The efficiency pass
   (T12) is parked until the pool and features settle; ranker v0 says which methods and features are cheap to drop.
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
9. *(Resolved 2026-11-11 by P30: `batch_test_acf_fft_highpass.py` was deleted together with
   `batch_test_guesses.py`; see the changelog.)*
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
12. **Arbitration loses correct candidates; now mostly a decline problem** *(refreshed 2026-11-11)*. History: on
   synth200, Stage 1 recall was 0.84–0.98 per bin but final correctness 0.63–0.84; T9 traced most of the loss to the
   gates, and T18 rescued the stars lost to ~0.2 d noise combs. On real SMARTS, ranker v0 orders candidates well
   (dev_medium full-year no-decline top-1 0.441 vs an oracle 0.764) but its single global decline threshold
   over-declines correct long-period answers. A perfect decline would add at most +0.106 of all stars full-year and
   +0.059 assigned (midway review, Fig. 1). Next step: ranker v1's star-level, period-aware decline model (P24).
   Hand-gate tuning (P1, P7, T16d) is paused.
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
17. **No non-rotators, and no detectability ceiling (2026-11-09).** Every SMARTS star is a rotator, so the
   false-positive rate on real non-rotators and variables is unmeasured (T27, then T29 for variable/EB flags). Also,
   dev_medium declines 65% of stars, and we can't yet tell how many are undetectable (T26).
18. **Single-mission so far (2026-11-09).** Everything is built and tuned on TESS/SMARTS. Roman (required) has no
   loader, data or scorecard yet (T28), and the core should become mission-agnostic (T30).

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
- **Keep the changelog short enough to read at start-up.** When it grows past roughly 100 lines, move the
  oldest entries verbatim to `docs/CHANGELOG_ARCHIVE.md` and update the pointer at the top of the changelog.
- **If the code map (Section 2) or a design decision (Section 3)
  meaningfully changes**, update that section directly rather than
  leaving it stale and only noting the change in the changelog — Section
  2/3 should always reflect current reality; the changelog is the
  history of how we got here.

### Changelog

Entries before 2026-10-22 (project start through T21/T16b/T20 phase 2: the agent/TDD setup, REL_TOL 0.10,
diffrot scoring, T4, T14, T15, T9, the ACF long-period limit, T18, T1, T16a, T2, T16e, T21, T16b, T20 phases 1–2)
are in `docs/CHANGELOG_ARCHIVE.md` (moved 2026-11-11, P30). Newest entries are at the bottom.

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
- **2026-10-22: T16c done (draft, awaiting review).** `coverage.searchable_range(time, flux)` (P_min = 2 cadences, R at valid-pair fraction 0.3, P_max = R/2; per batch, time sampling only). `fit_rotation_period(max_period="auto")` skips deduplicated candidates with P0 > R/2 (R from `acf_counts`), keeps them in `EnsembleResult.beyond_reach_candidates`, and records `EnsembleResult.searchable_range`; new decline kind `all_beyond_reach`; ranker feature `P_over_reach`. Bit-identical on all 7 before/after scorecard pairs (dev_medium full-year + assigned, 5 synthetic coverage classes): the uncapped fit never reported P/R > 0.51 anyway. Fit time -3% overall (-17% gapped-2). R/2 confirmed per class (success 0 past 0.5 everywhere). `min_valid_frac` 0.25 not recommended (no success gain, +0.4-0.5 pp wrong on dev_medium). pairwise_histogram returns no candidates for ~2/3 of 1-2-sector stars; LS window aliasing not a measurable population failure. Report: `reports/2026-10-22_t16c-searchable-range/`.
- **2026-11-09: T16c accepted and merged.** Rae kept every recommendation: a single R/2 for all coverage classes, `min_valid_frac` 0.3, and the draft goals (added once scorecard v4 is on main). `coverage.PMIN_CADENCES` changed 2 to 4 (the guesses' real floor; recorded only, no result change).
- **2026-11-09: Scorecard v4 (T22+T23) merged.** `SCORECARD_VERSION` 4 adds the `final.by_band_width`, `final.by_coverage_class`, `final.searchable.*` and `final.unsearchable.*` breakdowns, per-star `acf_reach`/`P_over_reach`/`searchable`, and a completeness map. The new baselines are `benchmarks/baselines/*_v4.json`, and every v3 key is unchanged. On merged main, synth200 reproduces `synth200_s0_v4.json` exactly. The goal proposals (v4 report + T16c report) are not yet in `goals.json`.
  - The new cross-check test (P22a) found that `coverage.searchable_range` lacked the 0.7 × baseline lag cap that `compute_acf` and `evaluation.acf_reach` apply. This moved R by one cadence on contiguous data, and only for the `P_over_reach` ML feature. It is now fixed (`coverage.REACH_MAX_LAG_FRAC`, measured on the full array's baseline). The comb fit and scoring were unaffected.
- **2026-11-09: Big-picture roadmap (Rae + Claude, outside the coordinator).** Rae restated the goals (Section 1:
  TESS + Roman, efficiency, variables/EBs, beginner-friendly, deadline = Roman GBTDS Season 1). Agreed sequence, now at
  the top of `docs/TASKS.md`: (1) close the SMARTS detour with T24/T25 + new T26 (detectability ceiling; dev_medium
  full-year is 0.25 success / 0.09 wrong / 0.65 declined); (2) a rough **ranker v0** soon (T10/T11), with gate-failed
  candidates and noise-only non-rotator negatives (new T27); (3) **Roman track** (new T28; one GBTDS season ≈ one
  contiguous batch, so multi-year combination is off the Season 1 critical path); (4) then the rotation-like/EB flags
  (new T29), T16h, T12, own butterpy sims, and the mission-agnostic API (new T30). Gate hand-tuning (T16d, P1, P7, P21)
  is paused in favour of the ranker. D2 leaning: per-batch candidates aggregated into fixed-size features before the
  ranker. New open issues 17–18.
- **2026-11-10 — T10 ranker v0 feature tables** (`reports/2026-11-10_t10-ranker-feature-tables/`):
  `tools/build_feature_tables.py` (every candidate incl. gate-failed / fit-error / beyond-reach,
  gates as features, graded `relevance` 2/1/0, `star_id` + `noise_tic` per row; schema
  `docs/FEATURE_TABLE_SCHEMA.md`; reproduces the scorecard's final answer bit-for-bit on dev_medium).
  `datasets/train_v0.csv`: 3,000 SIM_ID < 800k **strict-quiet** stars (coherent-quiet sources
  excluded, Rae 2026-11-10), nested `nested_1k`. Alias check: T9 features don't penalize diffrot
  within a star, but the half-lag gate passes 32% of diffrot vs 52% of correct candidates.
  Pitfall: spawned pool workers re-import modules, so module-global overrides don't reach them.
- **2026-11-10 — T11 ranker v0** (`reports/2026-11-10_t11-ranker-v0/`): `claude_files/ranking_model.py`,
  `tools/train_ranker.py` (resumable; noise-TIC-grouped 5-fold CV for all selection; decline threshold from OOF).
  `rank:ndcg` (depth 3, eta 0.1, 150 rounds; pairwise tied) on train_1k: dev_medium full-year success 0.248 → 0.267
  at matched precision (strict-quiet 0.254 → 0.275, wrong 0.035 → 0.016), assigned 0.148 → 0.188. Ordering is the
  strength (no-decline top-1 0.441 vs oracle 0.764); the global decline threshold is the weakness (over-declines
  correct long-period answers, P24). Alias features + wavelet/LS spectra matter most; gate flags and
  `acf_fft_highpass`/`pairwise_histogram` spectra least (T12). Label ablations only move the threshold.
- **2026-11-10 — T27 non-rotator negatives** (`reports/2026-11-10_t27-nonrotator-negatives/`): `datasets/negatives_v0.csv`
  (4,708 rows: T20 A+B noise stacks of 743 train-pool + 434 dev_medium quiet/SCM sources, 3 class-balanced coverage draws
  + full-year; pools TIC-disjoint), builder `--nonrotator` mode (relevance 0, `is_rotator` False), tables on the drive
  (`extracted/t27_negatives/`). Stacks are fair (imprint at chance, p2p = single files). Hand-arbitration FP rate 2.5–3.1%
  assigned, ≤ 0.8% full-year (gates `mean_height_snr`, `half_lag_consistency`); ranker v0 1.6–1.8% on dev_medium.
  All-zero queries give `rank:ndcg` no gradient, so negatives only help a decline model: an exploratory star-level
  decline head (prec 0.89 at 1:1) gives full-year 0.251/0.050 wrong/0.5% FP vs hand 0.248/0.088/0%. Many "quiet" FPs are
  consistent weak real periods (label noise). Open: precision target and negative:rotator ratio (Rae).
- **2026-11-10, scorecard v5 (T24/T25/P22b; branch claude/t24-t25-scorecard-v5-84eca3, report
  `reports/2026-11-10_scorecard-v5-noise-period/`):** noise-source join (`benchmarks/noise_sources_dev.csv`, built by
  `tools/build_noise_table.py`) gives per-star `noise_*` columns and `final.noise.*` (wrong split into noise_period vs other:
  65 of dev_medium full-year's 88 wrong match the noise source's P_ref), `final.quiet_noise` / `final.quiet_noise_strict`
  (drops 22 coherent-quiet stars) / `final.periodic_noise`; `--noise-subset`; `per_candidate.csv` `beyond_reach` rows (P22b).
  All v4 keys bit-identical; `*_v5.json` baselines committed. Open for Rae: halves/doubles of the noise period, quiet-class
  P_ref matches, quiet-subset goals.
- **2026-11-10: Scorecard v5 (T24+T25+P22b) merged.** Rae chose "merge as is": `noise_period` (a breakdown of `wrong`) matches the noise source's `P_ref` for every class, quiet included (65/88 dev_medium full-year wrongs; 12 of them on quiet-class sources); halves/doubles are a diagnostic only. The quiet subsets are `final.quiet_noise` and `final.quiet_noise_strict` (drops the 34 coherent-quiet sources, which Rae has since decided to reclassify as `periodic_low_amplitude` and exclude from training). All v4 keys are bit-identical. The `noise_sources_dev.csv` table must be rebuilt from the regenerated flags once the reclassification lands.
- **2026-11-10 — T20 reclass + quiet-noise periods** (`reports/2026-11-10_quiet-noise-periods/`). Rae's decision applied:
  new T20 class `periodic_low_amplitude` (quiet, P_ref < 5 d, time-split local LS S/N > 20 in both halves; reproduces
  the note review's 34 sources), excluded from training. Now 240 periodic sources / 141,056 files; 27,828 of the
  200,000 ≥800k-pool files excluded. `noise_source_flags.csv.gz` regenerated (new column `tsplit_min_snr`). Remaining
  quiet P_ref values look like red-noise/common-mode peaks of the fixed noise realization: dev_medium's 10 remaining
  quiet `noise_period` wrongs are 5× chance (null mean 2.0) but none of their sources is time-coherent or has a Gaia/FFI
  match. Weak injected signals let the pipeline lock onto the noise peak. Recommended: no further exclusions.
- **2026-11-10 — Decline operating point deferred (Rae).** The precision target / non-rotator mix for the ranker's
  decline threshold stays open until more of the pipeline is in place. Working default: match hand arbitration's OOF
  training precision; reports plot the full purity–completeness curve instead of asking Rae to pick a point (T11).
- **2026-11-11 — T27 open questions 4–7 settled (Rae took the coordinator's recommendations).** Persistent-weak-period
  "quiet" stacks leave the negatives (done in ranker v1); common-mode full-year FPs go to T29; the 125 spike stacks stay
  out; no synthetic red noise for now. Non-rotator:rotator ratio only needs to be good enough for now (Rae expects new
  negative sources, e.g. Roman and variable-star false positives); use all three coverage draws per source.
- **2026-11-11 — Ranker v1 direction (Rae).** Adopt T27's star-level decline model; no full-year training table until v1
  shows whether long-period full-year stars still lag (then build one for the 1k subset). Noise-level proposal filed (P25).
  Candidate recall by method (`reports/2026-11-11_candidate-recall-by-method/`): <1 d is limited by Stage 1 itself (all 7
  methods: 44% full-year, 63% assigned), and every method loses <1 d stars on full-year baselines; `acf_fft_highpass` is
  the best short-period addition (+0.10 full-year), `wavelet` the best long-period one (+0.12–0.19 at > 30 d).

- **2026-11-11 — Collected goal proposal** (`reports/2026-11-10_goals-proposal/`, `benchmarks/goals_proposed.json`). G13–G19 + 3 guard additions from the v4, T16c and v5 drafts, graded on the v5 baselines (4 pass, 3 fail); `goals.json` untouched, awaiting Rae's thresholds.
- **2026-11-11 — Ranker 3k learning-curve point** (`reports/2026-11-11_ranker-3k-learning-curve/`). `train_3k` built
  (drive `extracted/t10_tables/train_3k`, superset of train_1k). 1k → 3k: dev_medium success at matched precision
  0.267 → 0.273 full-year, 0.188 → 0.197 assigned; no-decline flat. The curve has plateaued: ranker v1 should train on
  train_3k, but gains must come from the decline stage, Stage 1 recall and new kinds of training data. `runs/t11_ranker`
  now lives in the main checkout.
- **2026-11-10 — T26 detectability ceiling** (`reports/2026-11-10_t26-detectability/`, branch `claude/t26-detectability-7f7523`).
  Per-star D_band·√N_cyc from the injected signal (SMARTS flux / T20 stack) vs the noise on rotation timescales, on the
  pipeline's cadences; AUC for success 0.86 FY / 0.88 assigned vs 0.81 / 0.79 for the T1 amp proxy. Proposed `detectable` ≥ 5:
  56% (FY) / 44% (assigned) of dev_medium; detectable FY success 0.43, wrong 0.10, declined 0.47. Undetectable quiet-noise stars
  are declined 97% of the time (correct); wrong answers on undetectable stars are mostly contaminated noise sources (41/47).
  The "correct P in any candidate" ceiling is mostly chance (0.76 vs 0.65 shuffled); only <1 d is informative, and there half of
  detectable stars have no correct candidate (agrees with the candidate-recall report). Scorecard flag proposed as P26.
- **2026-11-10 — Scorecard v6: `detectable` flag (T26/P26, Rae's request).** Committed side table
  `benchmarks/detectability_dev.csv` (`tools/build_detectability_table.py`; T20 stacks on the drive, read-only) keyed by
  (star_id, loaded sectors) for dev_small + dev_medium in both coverage modes. The scorecard joins it into per-star
  `detectability`/`detectable` (`--detectable-threshold`, default 5) and reports `final.detectable`, `final.undetectable`,
  `final.detectable_searchable`; all earlier metric keys are bit-identical (dev_small v5 vs v6). Rae: undetectable stars are
  **not** ranker negatives; noise-only stack light curves (T27) are the negatives.
- **2026-11-11 — T5 short-period recovery** (`reports/2026-11-11_t5-short-period-recovery/`). Full-year <1 d deficit
  = top-10 crowding (3–4× more spectral peaks, red-noise power rising to long P), not a grid/min-period limit.
  Hybrid raw-power / local-SNR peak ranking in `lombscargle`/`acf_fft` (default; `rank_by="power"` = old): dev_medium
  <1 d Stage-1 union 0.442 → 0.698 full-year, 0.634 → 0.779 assigned; fitted ceiling 0.33 → 0.65 / 0.59 → 0.69;
  final success ~flat, wrong flat, all guards pass; synth200 success 0.730 → 0.755 (G1 passes). No extra compute
  (paired 0.93×). Optional methods add no fitted <1 d / 1–3 d recall on top of the fixed pool (not pooled).
  Follow-ups P31 (rebuild ranker tables), P32 (half-lag gate), P33 (pairwise on long baselines).
- **2026-11-11 — Baselines refreshed for T5 (Rae OK).** `benchmarks/baselines/{dev_medium_fullyear,dev_medium_assigned,synth200_s0}_t5.json` (+ `_tables/`) are T5's after-fix scorecards (v5 keys), now the baselines for `check_goals.py --baseline`; `_v5` files stay as the pre-T5 reference. Reused rather than re-run: same pipeline code as main. v6 (`detectable`) baselines will come from the speed-up pass's "before" run.
- **2026-11-11 — P30 repo and docs diet** (`reports/2026-11-11_p30-repo-docs-diet/`). Deleted `claude_files/old/`
  (11 files) and the retired runners `batch_test_guesses.py`, `batch_test_acf_fft_highpass.py` (open issue 9 resolved)
  and `run_debug_baseline*.py`; the scorecard (`per_method.csv` / `per_star.csv`) and `tools/build_feature_tables.py`
  cover them. Last commit with all of them: `d3f73e6`. Lost capability: the per-smoothing-window breakdown of
  `acf_fft_highpass` and new `summary.csv` files for `analyze_batch_processing.ipynb` / `run_full_analysis.py`
  (existing outputs still load). Pruned 15 superseded baseline entries, 21 files (listed in `benchmarks/baselines/README.md`).
  Done tasks moved to `docs/TASKS_DONE.md`, changelog entries before 2026-10-22 to `docs/CHANGELOG_ARCHIVE.md`; §2 code
  map (adds `ranking_model.py`, drops `example_usage.py`) and §4 issues 2–5, 9, 12 refreshed. Coordinator's merge step
  now includes the TASKS status + changelog update (`docs/AGENT_WORKFLOW.md`); report TL;DRs capped at ~150 words.
  dev_small scorecards (assigned and full-year) bit-identical before/after; pytest passes.
- **2026-11-12 — P28 headline card + P29 Roman-season proxy** (`reports/2026-11-12_headline-card/`). New
  `tools/headline_card.py` (+ `tests/test_headline_card.py`): fixed 6-row card (raw; TESS coverage mix P ≤ 30 d;
  period-population-weighted success; correct-only; precision at prevalence 0.1/0.3/0.5; Claytor-style within 10%/20%
  at the most credible half) from any `per_star.csv` / ranker top-1, re-scored via `classify_period`, bootstrap SEs.
  dev_medium assigned hand T5 / ranker v0: 0.115 / 0.198 (cov. mix, P ≤ 30 d), 0.051 / 0.102 (`mcq14_like`); full-year
  ranker R5 0.51 / 0.64 vs Claytor+2022 0.46 / 0.69. T5 moves no row > 1 SE. P29: `consecutive-3-5`, P ≤ 30 d =
  0.205 hand / 0.284 ranker (unchanged by T5); a true 70 d window needs a per-star-sectors manifest + a
  `--max-span-days` dataset option (~14 min to score, not run). Population default, prevalences and goals H1–H5
  (proposed, `goals.json` untouched) await Rae.
