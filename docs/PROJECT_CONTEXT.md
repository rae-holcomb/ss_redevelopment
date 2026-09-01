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
