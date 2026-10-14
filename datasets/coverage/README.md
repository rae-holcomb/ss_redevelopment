# Coverage library (T16a)

`cycle1_patterns.csv` lists the real-star TESS Cycle-1 sector patterns, with how
many stars share each one. The scorecard's `--coverage` option and
`claude_files/coverage.py` use it to give synthetic stars or SMARTS files
realistic patchy coverage.

- **Built by:** `python tools/build_coverage_library.py`, from `target_df_shortform.csv`
  (8,156 TIC stars, each with its full observed sector list `sec`). The first line
  of the CSV is a `#` provenance comment that includes the input file's sha256.
- **Pattern:** a star's sectors within 1–13, written as `1-2-13`. SMARTS light curves
  span 13 sectors (one year), so sector s of a pattern is orbits 2s−1 and 2s of a
  SMARTS file.
- **Counts:** 3,438 stars have at least one Cycle-1 sector, and they share 121 distinct
  patterns. These are the same 121 as the old, deduplicated `sector_configs.txt`.

| class | patterns | stars | fraction |
|---|---|---|---|
| single | 13 | 2552 | 0.742 |
| consecutive-2 | 12 | 547 | 0.159 |
| gapped-2 | 10 | 49 | 0.014 |
| 3-5 | 37 | 160 | 0.047 |
| 6-12 | 48 | 92 | 0.027 |
| full | 1 | 38 | 0.011 |

Columns: `pattern`, `n_stars`, `frac_stars`, `n_sectors`, `consecutive`, `class`,
`first_sector`, `last_sector`, `span_sectors`, `max_missing_run`.

## Known biases

- **Cycle 1 only.** Sectors after 13 are ignored, so a star observed in S1 and S28
  counts as single-sector. The 4,718 stars observed only after Cycle 1 are dropped.
- **Southern ecliptic only.** Cycle 1 pointed south: every star in the library has
  ecliptic latitude between −89° and −6°. Northern-hemisphere patterns (Cycle 2)
  may differ, for example in how many CVZ stars there are.
- **Unknown target selection.** How `target_df_shortform.csv`'s 8,156 stars were
  chosen isn't recorded (Rae to confirm).
- **No multi-year gaps.** SMARTS spans one year, so revisit gaps between cycles
  can't be emulated (see TASKS.md D2).
- For 37 stars (0.45%), `num_s` differs from `len(sec)` in the input table. The
  builder uses `sec`.
