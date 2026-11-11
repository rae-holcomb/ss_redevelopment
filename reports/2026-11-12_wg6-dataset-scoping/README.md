# WG6 Roman variable-star light curves: scoping

**Date:** 2026-11-12 · **Author:** T28 scoping session (Claude) · **Branch:** `claude/t28-wg6-scoping-43bfaa` · **Status:** draft

## TL;DR
- **The WG6 download is not Roman-simulated photometry.** The README promises GP resampling to Roman cadence plus a
  Roman noise model, but the files and the code have neither. What's on the drive is **68,824 archival light curves**:
  OGLE I (plus V and UKIRT H/K) from 2001–2024, and TESS 20-s data for 390 flare stars. The files store **no truth
  parameters**.
- I joined OGLE catalog periods (Rae's OK): **99–100% of stars match** in every OGLE class. That includes
  **3,172 OGLE spotted rotators (ROT)**: periods 1–75 d (5–95%), 58% at 1–35 d. ROT agrees with our quick
  Lomb–Scargle on 88 of 89 measurable stars.
- Usable rotators: ROT, FLARE. Contaminants: ECL, ELL, T2CEP, CEP, RRLYR, HB, DSCT. Stress tests: LPV, DN, TR, BLAP.
  Not useful: VVV.
- **For Rae to decide:** how to make the light curves Roman-like (none of it can go through the pipeline as-is), and
  whether EBs are negatives or a flagged class.

## Question / goal
T28 (Roman GBTDS track), scoping step: where the RGES PIT WG6 variable-star set comes from, what it contains, which
classes are worth using, and a plan for bringing them in. This task changes no pipeline code. No goal in
`benchmarks/goals.json` applies yet; a Roman scorecard doesn't exist.

Deliverables: `docs/WG6_REFERENCE.md` (the reference, including the full README-vs-code discrepancy list) and
`docs/ROMAN_REFERENCE.md` (GBTDS design, draft, every line tagged confirmed or unconfirmed).

## What I did
1. **Read everything.** I read the shipped `README.txt` and `README.docx`. I also read the generating repo
   `rges-pit/variable_catalog_xmatch` at commit `5982d92` (2025-10-01), in a scratch clone: every `.py`, the docs
   notebook and `data/`.
2. **Extracted onto the drive** with `tools/extract_wg6.py`. It checks the mount (`WG6_DATA_ROOT`), resumes, writes
   atomically (`.part` then rename, or a `.tmp` folder then rename), handles one archive at a time, and uses
   `tarfile` `filter="data"`. Output is 22 GB under `extracted/`.
3. **Inventoried every file's header** with `tools/audit/wg6_inventory.py`: 68,824 unique stars.
4. **Downloaded the OGLE OCVS truth tables** with `tools/fetch_ogle_truth.py`: 35 files, 105.8 MB, to
   `extracted/_truth/`. Rae approved this via the coordinator. I joined them with `tools/audit/wg6_truth.py`.
5. **Profiled a sample** with `tools/audit/wg6_profile.py`: 30 stars per class plus 100 ROT, 524 stars in all. Per band
   it measures span, cadence, seasons and point-to-point noise, and runs Lomb–Scargle on the primary band. I cached
   examples and timed the pipeline on 3 ROT stars with `tools/audit/wg6_examples.py`. The notebook regenerates every
   figure from `data/`.

## Results

### Where the data came from: plain-language summary
- **Selection.** WG6 took every known variable from the OGLE-IV bulge catalogs (plus VVV, plus Gaia alerts) inside a
  4°-radius circle at l = 359.7°, b = −1.6°.
  - That circle is about 50 deg², roughly 30× the real GBTDS footprint of ~1.7 deg².
  - For each star they downloaded the OGLE I/V photometry and added UKIRT microlensing-survey H/K (2015–2017) where
    available.
  - The flare stars are a separate, **all-sky** set: TESS M dwarfs from Günther+2020 and Medina+2020, with TESS 20-s
    PDCSAP converted to magnitudes.
- **What did not happen** (README vs code vs files):
  - no GP resampling;
  - no Roman cadence;
  - no transformation to F146;
  - no Roman noise;
  - no "magnitude-shifted copies".

  The public code can't even run as committed (a missing module, a wrong call signature, a missing script). So the
  files were made with unpublished code, but the files themselves show which steps were done.
- **Quirks:**
  - Cepheid times carry a doubled +2,450,000 offset.
  - ELL files are labelled `eclipsing_binary`.
  - The `*SOURCE` header keys are constant boilerplate.

  All 10 discrepancies are listed in `docs/WG6_REFERENCE.md` §5.

### What's there: counts and profile
![counts](figures/counts_per_class.png)

ECL dominates (47,668 stars). The two rotator classes are small next to it: ROT has 3,172 and FLARE has 390.

| Class | Stars | Catalog P joined | P 5/50/95% [d] | Points (primary band, sample median) | Median spacing [min] | Seasons × length [d] | Median mag | p2p [mmag] |
|---|---|---|---|---|---|---|---|---|
| ROT | 3,172 | 100% | 1.0 / 25.7 / 74.6 | 12,014 (OGLE I) | 59 | 14 × 267 | I 16.7 | 13 |
| FLARE | 390 | 70% (lit.) | 0.32 / 1.4 / 124 | 89,023 per sector (TESS) | 0.33 | 1 × 24 per sector | T 11.1 | 5 |
| ECL | 47,668 | 99.98% | 0.31 / 0.58 / 8.2 (P_orb) | 7,113 | 57 | 6 × 271 | I 18.1 | 60* |
| ELL | 4,227 | 100% | 0.36 / 1.39 / 227 (P_orb) | 10,529 | 22 | 6 × 271 | I 16.5 | 16 |
| HB | 94 | 100% | 128 / 246 / 437 (P_orb) | 12,314 | 36 | 22 × 252 | I 13.8 | 3 |
| RRLYR | 6,809 | 100% | 0.27 / 0.51 / 0.68 | 6,516 | 67 | 8 × 269 | I 17.6 | 59* |
| DSCT | 1,160 | 99% | 0.049 / 0.075 / 0.15 | 14,309 | 24 | 10 × 269 | I 17.4 | 64* |
| CEP | 25 | 100% | 0.29 / 1.1 / 4.4 | 1,774 | 1,291 | 14 × 243 | I 16.9 | 58* |
| T2CEP | 224 | 100% | 1.2 / 8.5 / 36.7 | 2,121 | 1,118 | 14 × 243 | I 16.5 | 32* |
| LPV (Miras) | 4,373 | 100% | 171 / 319 / 490 | 3,409 | 585 | 10 × 256 | I 15.6 | 19 |
| VVV-LPV | 401 | 0 | — | 50 (all bands, UKIRT H/K) | — | — | — | — |
| BLAP | 9 | 0 | (LS ~0.02–0.05) | 13,739 | 24 | 10 × 270 | I 17.3 | 50* |
| DN | 198 | 0 | — | 6,067 | 33 | 7.5 × 262 | I 20.0 | 225 |
| TR | 74 | 0 | — | 14,135 | 24 | 10 × 269 | I 15.2 | 5 |

\* p2p includes real variability between consecutive points, which dominates for fast or large-amplitude variables.
Full table: `data/profile_table.csv`.

### Time coverage vs a Roman season
![coverage](figures/time_coverage_one_star.png)

**Top:** a typical ROT star has 14 OGLE I seasons of ~270 d each, sparse V, and a few weeks of UKIRT H.

**Bottom:** in its densest 70-d window OGLE has 1,251 points in nightly clumps; a GBTDS season at 12.1 min would have
8,331 contiguous points. This sampling difference is the central obstacle: on a regular grid only 15–18% of 15-min
cadences are filled, and `compute_acf` truncates the ACF after ~17 lags (`min_valid_frac=0.3`). **The pipeline can't
take these light curves as they are.**

### Example light curves per class
![examples](figures/example_lcs_periodograms.png)

Each row shows one star:
- **left:** the full archival light curve;
- **middle:** the densest 70 d window;
- **right:** the Lomb–Scargle periodogram, with the catalog P (solid red), P/2 and 2P (dashed) and 1 d (purple dotted).

Look at:
- **ROT**: clean ~38 d spot modulation in a single 70 d window;
- **ECL/ELL**: the main peak sits on the dashed P/2 line, not on the catalog P_orb;
- **HB, LPV, DN**: the 1 d window peak dominates the raw periodogram;
- **T2CEP**: sharp, strictly periodic peaks at 1–40 d, right in the rotation range.

### Periods vs our range
![periods](figures/period_hist_per_class.png)

Filled histograms show the catalog periods of every star; hatched ones show our top LS peak on the sample. ROT spans
the whole rotation range, but its mode (~30–50 d) sits right at the one-season R/2 = 35 d line: 63% of ROT stars are
at 0.1–35 d, 58% at 1–35 d, and 37% are above 35 d. Which contaminants fall in 1–35 d:
- all of T2CEP (94%);
- about 30% of ECL/ELL (by P_orb; their light curves show P_orb/2);
- half of CEP.

RRLYR and DSCT are entirely sub-day. LPV and HB are far beyond any single season.

### Noise vs magnitude
![noise](figures/noise_vs_mag.png)

OGLE I point-to-point scatter rises from ~3 mmag at I ≈ 13 to ~100 mmag at I ≈ 19. That is 1–2 orders of magnitude
above SMARTS-TESS (0.23–2.0 mmag, 5–95%, 30-min, green band, measured on 100 dev files). The flare stars sit on the
TESS sequence (≈5 mmag per 20 s at T ≈ 11). F146 noise at bulge magnitudes is unknown to us (no WG6 model; see
ROMAN_REFERENCE §3). Any Roman-like product will have to impose it.

### Stored vs measured periods
![check](figures/stored_vs_measured.png)

Our period is the top LS peak after skipping ground-based window peaks (0.95–1.05 d, 0.49–0.51 d, >300 d; the raw peak
for LPV). Points on the solid line agree with the catalog.

| Class | match | P/2 | 2P | other | no measurement |
|---|---|---|---|---|---|
| ROT (n=100) | **88** | 1 | 0 | 0 | 11 (1-d alias swamps it) |
| ECL | 1 | **79** | 0 | 8 | 2 |
| ELL | 1 | **28** | 0 | 1 | 0 |
| HB | 1 | 16 | 0 | 4 | 9 |
| RRLYR / DSCT / CEP / T2CEP / LPV | 28 / 28 / 23 / 29 / 30 | 0 | 0 | 2 / 0 / 2 / 1 / 0 | 0 / 2 / 0 / 0 / 0 |
| FLARE (lit. P) | 14 | 1 | 1 | 6 | 8 |

Conventions this confirms:
- **ROT** catalog periods are photometric spot periods, and reliable.
- **ECL/ELL** catalog periods are **orbital**; the light curve's dominant period is P_orb/2.
- **HB** light curves show harmonics of P_orb (P/2, P/3, P/5).
- **FLARE** literature periods agree only about two-thirds of the time on one TESS sector. Most misses have P > 25 d
  (Medina's MEarth periods).

### ROT, the Roman-like rotator truth (coordinator's question)
- **Source:** the OCVS "rotating variables toward the Galactic bulge" (Iwanek et al. 2024, 18,441 stars). WG6 holds
  3,172 of them, all with a `rot.dat` period.
- **Convention:** the "primary period" is the photometric period. 88/89 measurable stars agree within 10%; 1 is at
  P/2 and none are wrong. The 11 unmeasured stars are swamped by the daily alias in raw LS, which is a ground-based
  artefact rather than a catalog error.
- **Population caveat:** they are **mostly red giants/subgiants**, given I ≈ 15–18 at the bulge distance and I amplitudes of
  0.06–0.31 mag (5–95%). That is a much bigger spot signal than the Sun-like and M dwarfs Roman will mostly see.
- **Contamination flags:** OCVS remarks flag 52 of them as flaring and 4 as RS CVn (with EB IDs).
- **Period bins** (0–1 / 1–3 / 3–10 / 10–23 / 23–35 / 35–70 / >70 d): 159 / 111 / 545 / 649 / 540 / 959 / 209.

### Class classification for our use

| Class | Use | Why |
|---|---|---|
| **ROT** | **(a) rotator stand-in** (Roman-like truth) | Bulge targets, reliable photometric periods, 2,000+ with P < 35 d. Caveat: giant-dominated, large amplitudes |
| **FLARE** | **(a) rotator stand-in, TESS side** (+ (c) flare stress test) | Real M-dwarf rotators with literature P. Real TESS data, which gives a cross-mission check of SMARTS-trained models. Not bulge, not Roman cadence |
| **ECL** | **(b) contaminant**, but a *flagged* class (T29), not a plain negative | Light curves at P_orb/2 mimic rotators. Spotted/synchronised EBs are real rotators (Rae: flag, don't discard) |
| **ELL** | **(b) contaminant** (strongest rotation mimic) | Smooth double-humped sinusoid at P_orb/2, mostly 0.2–1 d with a 10–200 d tail |
| **T2CEP** | **(b) contaminant** | Strictly periodic, 1–37 d, right in the rotation range |
| **CEP** | (b) contaminant (only 25) | 0.3–4 d |
| **RRLYR** | (b) contaminant, sub-day bin | 0.27–0.68 d, large amplitude, sawtooth |
| **HB** | (c) stress test | P_orb 130–440 d (out of range) but harmonics land in range; only 94 |
| **DSCT** | (c) stress test | Mostly < 0.1 d, below our range; tests the short-period edge |
| **LPV / Miras** | (c) stress test | P 170–490 d. In one season they appear as a trend or a partial cycle, which tests long-trend false positives |
| **DN, TR, BLAP** | (c) stress test | Outbursts (DN), transit dips (TR, mostly flat), very short pulsators (BLAP, 9 stars) |
| **VVV-LPV** | (d) not useful | UKIRT H/K only, median 50 points, no truth joined |

The SMARTS negatives so far are noise-only stacks (T27). The (b) classes would be the **first astrophysical false
positives** the ranker sees.

### Game plan
1. **Make it Roman-like first.** This decides everything else, so Rae picks the approach (Q1 below).
   - **A. Model resampling** for the strictly periodic classes (ECL, ELL, T2CEP, CEP, RRLYR, DSCT, HB, LPV): fit a
     multi-harmonic Fourier/phase-binned template at the catalog period to the OGLE I data, then evaluate it on a
     70 d × 12.1 min grid with F146-like noise at a chosen magnitude. Cheap, deterministic, exact truth, numpy only.
   - **B. ROT** spots evolve, so a strict template is wrong. Either fit a quasi-periodic GP (e.g. a `celerite2`
     RotationTerm) per OGLE season and draw a conditional sample on the Roman grid (needs a package: Rae's OK), or
     inject ROT periods/amplitudes into our own `synthetic.spot_signal` on a Roman grid (no new dependency, but no
     longer "real").
   - **C. Ask WG6** for the GP-resampled Roman product their README says is complete, plus the noise model.
2. **Loader** `claude_files/wg6.py` (design only):
   - `load_wg6_fits(path, band="LC_I", window=None)` returns the same `(time, flux)` container as `load_smarts_fits`;
   - it applies `fix_time` (the doubled offset), converts mag to relative flux (median 1), optionally cuts one
     season or the densest 70 d window, and regrids to a regular cadence;
   - `make_roman_season(...)` implements option A or B behind one call, with the method and seed recorded in the output.
3. **Manifest**, as a new file `datasets/wg6_v0.csv` (no existing manifest edited). Columns:
   - `star_id`, and `path` relative to `WG6_DATA_ROOT`;
   - `cls`, `subtype`;
   - `P_true`, `p_convention`, `P_phot_expected` (P_orb/2 for ECL/ELL);
   - `imag`, `amp`;
   - `is_rotator`, `is_eb`, `label_source`;
   - `resample_method`, `seed`, `eval_set`.

   Splits are by star (no star in both train and eval).
4. **How the classes enter:**
   - (i) **Ranker negatives:** the (b) classes as star-level non-rotators (decline targets), every candidate at
     relevance 0, like T27. For ECL/ELL, also tag candidates within REL_TOL of P_orb or P_orb/2 as `eb_period` for the
     T29 flag.
   - (ii) **Roman scorecard dataset string**, proposed as `wg6_rot_v0[_roman70]` for the rotators and
     `wg6_contam_v0[_roman70]` for the false-positive rate. ROT/FLARE are scored with `evaluation.classify_period` against
     `P_true`. There is no `DIFFROT`, so the outcomes are correct/alias only. Stars with P > season R/2 count as
     *undetectable* (not failures), as on SMARTS.
5. **Compute** (`data/pipeline_timing.csv`): a Roman-like season (70 d at 12.1 min, 8,331 points) costs **0.57–0.83 s
   per star** on the i5 in one process (mean 0.70 s), and 2 of 3 periods were recovered. So:
   - ROT, one season each: ~37 CPU-min;
   - all 68,824 stars: ~13.3 CPU-h;
   - a v0 subset of all ROT plus ~300 per contaminant class: well under an hour at 4 workers.

   Raw OGLE seasons on a grid "run" in 0.01–0.02 s only because they fail early (ACF truncated), so those times mean nothing.

Follow-ups are proposed in `docs/TASKS.md` as P34–P37.

## What didn't work
- **Running the pipeline on raw OGLE seasons.**
  - At 15 min, `compute_acf` returns ~17 lags (coverage cut).
  - At 60 min the ACF survives, but `fit_rotation_period` found no period on 3 of 3 ROT stars at P = 2–16 d.

  This is why resampling is step 1.
- **Raw Lomb–Scargle on ground-based data.** 19 of 100 ROT stars have the 1-d window peak on top, so I added the mask.
  It stays a diagnostic only: Roman won't have the daily alias.
- **The first timing run** picked ROT stars with P > 35 d, which can't be found in 70 d. I re-ran it on P = 2–20 d.

## Open questions for Rae
1. **Resampling approach** (game plan step 1): A, template resampling for periodic classes; B, GP or our own
   spot-injection for ROT; C, ask WG6 for the Roman-resampled product their README describes. My suggestion: A plus
   C now, and B once you decide on a GP package (`celerite2` or `tinygp`, which needs your OK to install).
2. **EBs and ELLs:** star-level negatives for the ranker, or a separate flagged class (T29) kept out of the negatives?
   Your earlier guidance was "flag EBs, don't discard them".
3. **ROT is giant-dominated**, with large amplitudes. Is it acceptable as the Roman rotator truth for v0? Should
   amplitudes be scaled down to dwarf-like levels when resampling?
4. **The 37% of ROT with P > 35 d:** keep them as "undetectable in one season" truth (as with SMARTS), or drop them?
5. **The F146 noise model:** which source should we use (Roman ETC/pandeia tables, or the Wilson+2023 simulations)?
   The WG6 README mentions one, but it isn't in the repo or the files.
6. **Flare stars as a real-TESS validation set** for the SMARTS-trained ranker: 273 M dwarfs with literature P, a cheap
   over-tuning test (roadmap step 3). Worth a task?
7. **GBTDS design numbers** in `docs/ROMAN_REFERENCE.md` come from NASA/STScI web pages, not the final APT definition
   (12.1 vs 15 min cadence, season dates, off-season sampling). Who should confirm them?
8. Suspicious items:
   - zips `-002` and `-003` are absent, with `.tgz` files of those numbers in their place, which is consistent;
   - the 390 flare files are fewer than the 1,228 Günther stars (WG6 only kept stars with FAST-LC 20-s data);
   - 4 ROT stars are known RS CVn EBs.

## Reproduce
```bash
# drive mounted at /Volumes/Data_Vault (or export WG6_DATA_ROOT=...)
/opt/anaconda3/envs/ss_openblas/bin/python tools/extract_wg6.py              # resumable, ~40 min, 22 GB on drive
/opt/anaconda3/envs/ss_openblas/bin/python tools/fetch_ogle_truth.py         # 35 files, 106 MB on drive
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_inventory.py      # ~12 min, header-only
git clone https://github.com/rges-pit/variable_catalog_xmatch /path/to/scratch/wg6code   # commit 5982d92; outside the repo
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_truth.py --wg6-repo /path/to/scratch/wg6code
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_profile.py --n 30
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_profile.py --n 100 --sets Rotators_lightcurves
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_truth.py --period-check
/opt/anaconda3/envs/ss_openblas/bin/python tools/audit/wg6_examples.py --time-pipeline 3
# data/ls_spectra/ keeps only the 14 example stars (trimmed for git); the profile run regenerates the rest
/opt/anaconda3/envs/ss_openblas/bin/python tools/run_notebook.py reports/2026-11-12_wg6-dataset-scoping/notebook.ipynb
```
