# Frozen SMARTS datasets (T2)

| file | rows | data root | files live in |
|---|---|---|---|
| `dev_small.csv` | 50 | `data/smarts_dev` | laptop: `data/smarts_dev/` (gitignored, ~8 MB) |
| `dev_medium.csv` | 1,000 | `data/smarts_dev` | laptop: `data/smarts_dev/` (~156 MB together with dev_small) |
| `holdout.csv` | 3,000 | the SMARTS data root (`$SMARTS_DATA_ROOT`, default `/Volumes/Data_Vault/smarts_data`) | drive: `extracted/holdout/` |

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
