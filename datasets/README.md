# Frozen SMARTS datasets (T2)

| file | rows | data root | files live in |
|---|---|---|---|
| `dev_small.csv` | 50 | `data/smarts_dev` | laptop: `data/smarts_dev/` (gitignored, ~8 MB) |
| `dev_medium.csv` | 1,000 | `data/smarts_dev` | laptop: `data/smarts_dev/` (~156 MB together with dev_small) |
| `dev_small_clean.csv` | 47 | `data/smarts_dev` | `dev_small.csv` minus its 3 `noise_exclude` stars |
| `dev_medium_clean.csv` | 874 | `data/smarts_dev` | `dev_medium.csv` minus its 126 `noise_exclude` stars |
| `holdout.csv` | 3,000 | the SMARTS data root (`$SMARTS_DATA_ROOT`, default `/Volumes/Data_Vault/smarts_data`) | drive: `extracted/holdout/` |
| `train_v0.csv` (T10) | 3,000 | `$SMARTS_DATA_ROOT` | drive: `extracted/train_v0/` |

`train_v0.csv` is the ranker v0 **training** sample (`tools/build_train_sample.py`, seed 20261110): SIM_ID < 800k,
strict-quiet noise (T20 `quiet`, not `exclude_from_training`, not the 2026-11-09 coherent-quiet list, not the one
noise TIC shared with the >= 800k pool), stratified and coverage-assigned like dev_medium. The first 1,000 rows
(`nested_1k`, `train_order` 0-999) are a balanced nested subset. Extra columns: `subset`, `nested_1k`,
`train_order`, `noise_tic`, `noise_classification`, `noise_coherent_quiet`. Summary: `train_v0_build_summary.json`.

**Use the `_clean` dev manifests** (Rae, 2026-11-16). The frozen dev sets predate the T20 noise flags; 126 of
dev_medium's 1,000 stars (3 of dev_small's 50) sit on a noise source flagged `noise_exclude` (T20
`exclude_from_training`: known variable, crowded contamination, or unexplained source-specific signal). They carry
55 of the 88 full-year `_v7` wrong answers. `tools/build_clean_manifests.py` writes the `_clean` files (frozen rows
minus those sim_ids, same columns and order; `--check` verifies them, and `tests/test_noise_exclusion.py` too). The
scorecard refuses a manifest holding such stars unless `--allow-noise-excluded` (audits only). The holdout has no
sim_id column; filter it the same way when Rae opens it. `train_v0` and the negatives were already built without them.

**Never score `holdout.csv` unless Rae says so.** These manifests are frozen once Rae
approves them. A change means a new versioned file (`dev_medium_v2.csv`), never an edit.

## Scoring

Manifest `path`s are relative to the set's data root; the scorecard joins them with
`--data-root` (or `$SMARTS_DATA_ROOT` when no `--data-root` is given; a relative root is
taken from the repo root):

```bash
python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev --outdir runs/x
python benchmarks/scorecard.py --manifest datasets/dev_small.csv --data-root data/smarts_dev --ignore-manifest-sectors --outdir runs/x_fullyear
```

- The default uses each row's assigned `sectors` pattern (dataset string
  `datasets/dev_small.csv,data_root=data/smarts_dev,sectors=manifest`).
- `--ignore-manifest-sectors` scores the same stars on every native sector, i.e. full-year
  (`...,sectors=ignored`).
- If any file is missing, the scorecard stops before it starts, with a hint (wrong
  `--data-root`, or the drive is unplugged).
- If `$SMARTS_DATA_ROOT` is exported in your shell, the dev sets still need
  `--data-root data/smarts_dev`.

## How the sets were drawn (`tools/build_dev_sets.py`, seed 20261015)

The spec is the "Rae's design decisions (2026-10-15)" block of T2 in `docs/TASKS.md`.
`build_summary.json` holds the parameters and totals from the last build.

1. **Pool:** catalog rows with SIM_ID >= 800,000 (200,000 rows: the val/test galaxy-noise
   pool; ranker training uses SIM_ID < 800k). Files are joined by catalog **row index**
   (`smarts-tess-v1.0-RRRRRR.fits` = row RRRRRR).
2. **Strata (63):** audit period bin (<1, 1-3, 3-10, 10-30, 30-60, 60-120, 120-180 d)
   × amplitude-proxy tercile (ACTIVITY·sin(INCL)·cos(mid-band latitude); cuts 0.2199 and
   1.0279 over the whole 1M corpus) × shear class (anti / none / solar).
3. **Quotas:**
   - The five bins below 60 d get equal shares. 60-120 and 120-180 d together get 1/7 of
     each set (the "should decline" stars).
   - Within a bin, the nine amplitude × shear cells get equal shares, limited by each
     cell's capacity.
   - Sets are filled in the order dev_small, dev_medium, holdout.
   - The <1 d bin is capacity-limited: 803 of its 984 pool rows were checked. So the
     holdout's <1 d cells range from 45 to 66 stars instead of an equal 57.
4. **Exclusions:**
   - Each stratum's candidates (1.3 × quota, in a seeded random order) were extracted in
     one streaming pass per bundle. Each was checked with `tools/audit/flux_outliers.scan`:
     raw max |flux − 1| > 0.1 means corrupt galaxy noise.
   - Passing rows filled the quotas in rank order. Strata that ran short got more
     candidates.
   - 455 of 5,200 checked rows (8.75%) were excluded, all for corrupt noise. They are listed
     with reasons in `exclusions_log.csv`.
5. **Native sectors:** one `load_smarts_fits` call per file records two lists:
   - `native_sectors`: what the default load keeps, i.e. every sector with at least one good
     orbit. This is what full-year scoring sees.
   - `native_whole_sectors`: sectors with **both** orbits unflagged. An explicit
     `load_smarts_fits(sectors=[...])` drops a requested sector if either of its orbits is
     flagged, so only these sectors are safe to request.
6. **Coverage:**
   - Each row gets a pattern (`sectors`, `coverage_class`) from
     `coverage/all_cycles_patterns.csv`, using only patterns that are subsets of
     `native_whole_sectors`. The assigned pattern is therefore exactly what the pipeline
     loads; `tests/test_datasets.py` checks this for every dev_small row.
   - Classes are balanced within every period bin of every set, and patterns are uniform
     within a class among the feasible ones.
   - 3 dev_small rows got a fallback class, because a bin lacked a 13-whole-sector file
     for `full`.
7. **Weights:**
   - `sampling_weight` = corpus (pool) stratum share / set stratum share. When a set has
     no rows in a stratum (dev_small: 50 rows for 63 strata), that stratum's share is
     spread over the set's other strata in the same period bin. The weights sum to the set
     size.
   - `coverage_weight` = real star-season class share (from the library) / set class share.
   - `combined_weight` = their product.
   - Weighted rates estimate the *non-corrupt* pool, since excluded files can't be scored.

## Columns

- **Identifiers:** `path`, `star_id`, `row`, `sim_id`, `bundle`, `member` (path inside
  the bundle tarball).
- **Truth:** `true_period` (PERIOD = P_eq), `shear` (DIFFROT), `minlat`, `maxlat`,
  `activity`, `incl` (radians), `cycle`, `overlap`, `tspot`, `bfly`.
- **Differential rotation:** `P_lo`, `P_hi` (from `evaluation.active_band_period_range`),
  `band_disjoint` (whole band outside P_eq ± REL_TOL), `band_extends` (band reaches beyond
  ± REL_TOL).
- **Native data:** `native_sectors`, `n_native_sectors`, `native_whole_sectors`,
  `n_whole_sectors`, `native_class` (the coverage class of `native_sectors`),
  `max_abs_dev` (raw max |flux − 1|).
- **Coverage:** `sectors`, `n_sectors`, `coverage_class`, `n_feasible_patterns` (library
  patterns of that class that fit).
- **Strata and weights:** `stratum` (`pbin|amp_tier|shear_class`), `pbin`, `eval_pbin`
  (`evaluation.period_bin`), `amp_tier`, `amp_proxy`, `shear_class`, `sampling_weight`,
  `coverage_weight`, `combined_weight`.

`strata_counts.csv` has, per stratum: pool size, rows checked and excluded, and each set's
quota and count. The full tables and plots are in `reports/2026-10-15_t2-dev-sets/`.

## Rebuilding

```bash
/opt/anaconda3/envs/ss_openblas/bin/python tools/build_dev_sets.py --n-workers 2
```

- The rebuild needs the drive. It reuses the per-bundle check cache on the drive
  (`extracted/t2_build/checks/`) and the staged candidate files
  (`extracted/t2_build/candidates/`), so a rebuild with that cache takes a few minutes.
  From scratch it takes ~30 min of drive time.
- It is resumable and writes everything atomically.
- `tests/test_datasets.py::test_build_reproducible_from_seed` checks that a rebuild
  reproduces these files exactly.

## WG6 Roman-resampled (P35)

`wg6_roman_v0.csv` (14,253 rows; one per file x season 1-3, improved noise) is built by `tools/build_wg6_manifest.py`;
`zip`/`member` are relative to `$WG6_RESAMPLED_ROOT` (default `/Volumes/Data_Vault/wg6_roman_data_resampled`). Split by FL
template and by OGLE star (`eval_set` dev/holdout; **never score the holdout without Rae**). Score with dataset strings:

```bash
python benchmarks/scorecard.py --manifest wg6r_fl_v0_improved_s1 --outdir runs/wg6r_fl_s1       # FL rotators, dev
python benchmarks/scorecard.py --manifest wg6r_contam_v0_improved_s1 --outdir runs/wg6r_contam_s1 # non-rotators (FP rate), dev
python benchmarks/scorecard.py --manifest wg6r_beyond_v0_improved_s1 --outdir runs/wg6r_beyond_s1 # stars with P_true > R/2 (35.5 d)
```
On `contam`, "correct" = lock on the variable's own period = false positive; on ECL/ELL `double` = P_orb lock.

## Real-TESS validation set (P27): `tess_validation_v1.csv`

480 real TESS stars; **v1 frozen 2026-11-19** (`tools/build_tess_validation.py`; `--check` rebuilds in memory and fails on any
difference; `tests/test_tess_lc.py` runs it). `tess_validation_v0.csv` is the superseded first freeze (never scored; still
reproducible with `--version v0`). v1 applies Rae's answers to v0's open questions:

- **Locked 30% holdout** (`split == "holdout"`, 144 stars; the other 336 are `validation`): stratified by label x period bin (by
  source for the negative blocks), seeded (20261119), rounded per stratum. **Never score the holdout without Rae's word.** No scorecard
  dataset string selects it, and `tests/test_tess_lc.py` checks that.
- **Replacements:** the 72 stars with no TESS-SPOC or QLP light curve at MAST and the 7 EBs with `contratio` >= 0.2 were replaced
  from the same pools with stars that have SPOC or QLP files, strata counts unchanged (`proposed/p27_v1_replacements.csv`, drawn by
  `tools/validation_set/draw_replacements.py`, seed 271). Every v1 star is scorable and every star (EBs now included) has
  `contratio` < 0.2.
- **Gaia class instead of Teff for pulsators:** `gaia_class` / `gaia_pulsator` from the Gaia DR3 classifier and
  main-sequence-oscillator tables (`proposed/p27_v1_gaia_classes.csv`, `tools/validation_set/gaia_classes.py`); the `hot_star` flag
  is gone.
- **QLP stays fallback-only:** the downloader fetches QLP only for stars with no SPOC file.

Rows are keyed by TIC, not file paths: light curves are downloaded per sector by `tools/download_tess_validation.py` to
`$TESS_VALIDATION_ROOT` (default `/Volumes/Data_Vault/tess_validation`), which writes `files_index.csv` (paths relative to that
root). Product: TESS-SPOC `SAP_FLUX` (QLP `SAP_FLUX` fallback); PDCSAP is scored on the same stars. Loader: `claude_files/tess_lc.py`.

| `label` | n | `truth_period` | source blocks |
|---|---|---|---|
| `rotator` | 268 | literature P_rot, 0.18-65 d | clusters (Hyades, Praesepe, Pleiades, Rup 147, NGC 6811, Blanco 1, NGC 3532, NGC 2547) + Kepler field (McQuillan+2014) |
| `nonrotator` | 100 | none | `kepler_rc_quiet` (50), `kepler_dwarf_nodetect` (50) |
| `nonrotator_candidate` | 22 | none | `white_dwarf` |
| `expected_undetectable` | 40 | none | `claytor_scvz_template` (Zach's list minus the 294 Gaia DR3 variables): scored by detection rate, never as FP |
| `contaminant_eb` | 50 | orbital P (Prsa+2022), 0.1-60 d | `prsa2022_eb` |

Columns: `star_id` (`TIC<tic>`), `tic`, `label`, `source`, `truth_period`, `truth_kind` (`spot_rotation` / `orbital` / `none`),
`period_method`, `period_dataset`, `period_ref` (literature provenance for every period), `Tmag`, `Teff`, `logg`, `ra`, `dec`,
`contratio` (TIC crowding), `gyro_flag`, `sectors` / `n_sectors` (TESSCut FFI sectors), `n_tess_spoc`, `n_qlp`, `n_tglc`,
`n_cdips` (HLSP files at MAST when drawn), then: `pbin` (truth-period bin, rotators and EBs), `star_type` (`rc_giant` / `white_dwarf`
/ `giant` = logg < 3.5 / `dwarf`), `gaia_class`, `gaia_pulsator`, `pulsation_class` (`rc_giant` = the Kepler red-clump block,
`gaia_pulsator`, or empty: picks the pulsation window of `evaluation.classify_negative`), `gaia_variable` (Zach-block audit, 0),
`product_primary` (`spoc` / `qlp`), `scorable` (all True in v1) and `split`. Nothing is drawn beyond 71 d (the 70-180 d block is
skipped, Rae 2026-11-19).

Scorecard dataset strings (validation split only): `tessval_v1_{sap,pdcsap}` (rotators), `..._eb` (EB lock rate against P_orb),
`..._neg` (per-class false-positive and `pulsation` rates for `nonrotator` / `nonrotator_candidate`), `..._eu` (detection rate of
`expected_undetectable`). The outcomes are in `metrics.json` under `tessval` (see `evaluation.classify_negative`, `eb_outcome`);
`pulsation` windows: `rc_giant` 0.2-0.5 d, `gaia_pulsator` 0.1-3 d.
