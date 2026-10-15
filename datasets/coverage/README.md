# Coverage libraries (T16a)

Real-star TESS sector-coverage patterns, with how many stars share each one, for every
TESS observing cycle (year) 1–7. The scorecard's `--coverage` option and
`claude_files/coverage.py` use them to give synthetic stars or SMARTS files realistic
patchy coverage.

| file | content |
|---|---|
| `all_cycles_patterns.csv` | **Default library.** Cycles 1–7 pooled. `n_stars` counts *star-seasons* (a star observed in two cycles counts twice); `n_cycles` / `cycles` say where each pattern occurs. |
| `cycle<N>_patterns.csv` | One cycle (N = 1–7), selectable with `--coverage-library`. `cycle1_patterns.csv` holds the same 121 patterns as the old deduplicated `sector_configs.txt`, now with counts. |

- **Built by:** `python tools/build_coverage_library.py`, from `target_df_shortform.csv`
  (8,156 TIC stars, each with every sector it falls on, `sec`, S1–96). The first line of
  each CSV is a `#` provenance comment that includes the input file's sha256.
- **Cycles** (as in tess-point 0.9.2): 1 = S1–13, 2 = S14–26, 3 = S27–39, 4 = S40–55
  (16 sectors), 5 = S56–69 (14), 6 = S70–83 (14), 7 = S84–96. Every sector here is about 27 d;
  Year 8's longer pointings, from S97 on, aren't in the input table.
- **Pattern:** a star's sectors within one cycle, numbered from the cycle's first sector
  (1 = S1, S14, S27, ...), written as `1-2-13`. SMARTS light curves span 13 sectors, so
  sector s of a pattern is orbits 2s−1 and 2s of a SMARTS file. Cycles 4–6 are longer
  than 13 sectors, so `coverage.fit_to_year` maps their patterns into 1–13:
  - **Shifted:** a pattern that spans ≤ 13 sectors but ends after sector 13 is moved
    earlier, keeping its spacing. This affects 446 Cycle-4, 347 Cycle-5 and 404 Cycle-6
    star-seasons.
  - **Truncated** (kept as the policy, Rae 2026-10-15): a pattern spanning > 13 sectors keeps its 13-sector window with the
    most sectors. This affects 610 Cycle-4 and 25 Cycle-6 star-seasons, losing 1–2 sectors
    each (median 1).
  - `n_shifted` / `n_truncated` record this per pattern.
- **Classes:** single, consecutive-2, gapped-2, consecutive-3-5, gapped-3-5,
  consecutive-6-12, gapped-6-12 and full. The consecutive/gapped split of 3-5 and 6-12 is
  Rae's decision (2026-10-15). `--coverage 3-5` / `6-12` draw from both halves.
- **Sampling:** within a class, patterns are drawn in proportion to their star-season count
  (`--coverage-weighting frequency`, the default). Rae first chose uniform sampling to avoid
  tuning to Year 1's mix, then switched to weighted once the library pooled Years 1–7
  (2026-10-15). `uniform` is still available. With the pooled library, weighting is
  proportional to each year's star count in the input table (kept, Rae 2026-10-15) (Years 5 and 7 have ~4,000 each,
  Year 2 ~2,600).
- **Sector and year lengths:** from the TESS Science Office orbit times
  (`https://tess.mit.edu/public/files/TESS_orbit_times.csv`, fetched 2026-10-15). Sectors keep
  their usual length in every year (median start-to-start 26.8–27.7 d). The 14–16-sector years
  are simply *longer than a year*:

  | year | sectors | span |
  |---|---|---|
  | 1 | 13 | 358 d |
  | 2 | 13 | 352 d |
  | 3 | 13 | 354 d |
  | 4 | 16 | 434 d |
  | 5 | 14 | 383 d |
  | 6 | 14 | 376 d |
  | 7 | 13 | 349 d |

  A SMARTS light curve (13 × 27.4 d ≈ 356 d) can't hold Years 4–6, hence `fit_to_year`.
  Year 8 (S97–111, not in the input table) has irregular pointings: two 4-orbit ~55–58 d
  sectors (S97–98), single-orbit sectors (S102–103, 13–38 d) and 15 sectors over 450 d.

### Class fractions per cycle (fraction of star-seasons)

| cycle | stars | patterns | single | consec-2 | gapped-2 | consec-3-5 | gapped-3-5 | consec-6-12 | gapped-6-12 | full | frac north |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3438 | 121 | 0.742 | 0.159 | 0.014 | 0.037 | 0.010 | 0.000 | 0.026 | 0.011 | 0.00 |
| 2 | 2607 | 201 | 0.580 | 0.132 | 0.047 | 0.014 | 0.156 | 0.003 | 0.057 | 0.010 | 1.00 |
| 3 | 3414 | 126 | 0.731 | 0.176 | 0.011 | 0.033 | 0.012 | 0.001 | 0.026 | 0.010 | 0.00 |
| 4 | 3206 | 117 | 0.484 | 0.248 | 0.041 | 0.126 | 0.062 | 0.016 | 0.022 | 0.000 | 0.83 |
| 5 | 4019 | 116 | 0.726 | 0.164 | 0.010 | 0.051 | 0.021 | 0.016 | 0.011 | 0.000 | 0.38 |
| 6 | 3086 | 152 | 0.518 | 0.270 | 0.033 | 0.017 | 0.109 | 0.019 | 0.034 | 0.000 | 0.84 |
| 7 | 4006 | 96 | 0.752 | 0.145 | 0.023 | 0.041 | 0.011 | 0.000 | 0.027 | 0.000 | 0.36 |
| **pooled** | 23776 | 534 | 0.658 | 0.183 | 0.024 | 0.046 | 0.048 | 0.008 | 0.028 | 0.004 | |

(The `stars` column counts star-seasons for the pooled row.)

Patterns per class in the pooled library:

| single | consec-2 | gapped-2 | consec-3-5 | gapped-3-5 | consec-6-12 | gapped-6-12 | full |
|---|---|---|---|---|---|---|---|
| 13 | 12 | 39 | 24 | 203 | 11 | 231 | 1 |

413 of the 534 patterns never occur in Cycle 1, and 361 occur in only one cycle.

Columns: `pattern`, `n_stars`, `frac_stars`, `n_sectors`, `consecutive`, `class`,
`first_sector`, `last_sector`, `span_sectors`, `max_missing_run`, `n_shifted`,
`n_truncated`, then `cycle` (per-cycle files) or `n_cycles`, `cycles` (pooled file).

## Known biases

- **One cycle at a time.** Each pattern is one star-season. A star seen in S1 and S28
  contributes a single-sector pattern to both Cycle 1 and Cycle 3; revisits across years
  aren't represented, since SMARTS spans one year (see D2 below).
- **Hemisphere mix differs by cycle.** Cycles 1 and 3 are purely southern, Cycle 2
  purely northern, and Cycles 4 and 6 mostly northern. The northern-heavy years have far
  more multi-sector and gapped coverage (single 48–58% vs 73–75%). Pooling mixes these
  in proportion to this table's stars per cycle.
- **Cycles 4–6 are squeezed into 13 sectors** (shifted or truncated, above). No 13-sector
  star-season exists after Cycle 3, so `full` comes only from Cycles 1–3.
- **Target selection (Rae, 2026-10-15).** `target_df_shortform.csv` comes from a
  separate analysis by Rae: the few thousand **brightest stars in TESS**, with the sector
  coverage at each star's sky position computed with **`tess-point`**. It was intended as a
  guide to typical sector-coverage patterns. Consequences:
  - The patterns describe **geometric (FFI) coverage at the sky positions of bright stars**.
    That's the right notion for SMARTS, which emulates FFI light curves. It doesn't
    reflect any target-list or 2-min-cadence selection.
  - Brightness itself doesn't change the coverage. The sample's coverage mix follows the
    sky distribution of bright stars (concentrated toward the Galactic plane), which may
    differ somewhat from fainter or field-star populations. Ecliptic latitude, which sets
    the CVZ and overlap fraction, is the main driver.
  - `tess-point` reports *on-silicon* geometry. It doesn't model data gaps, bad orbits or
    stars falling in CCD gaps beyond its footprint model.
- **No multi-year gaps.** SMARTS spans one year, so revisit gaps between cycles
  can't be emulated (see TASKS.md D2).
- For 37 stars (0.45%), `num_s` differs from `len(sec)` in the input table. The
  builder uses `sec`.
