---
name: dev-set-builder
description: Downselects the ~1M-file SMARTS training set into small, stratified, frozen development manifests (dev_small, dev_medium) and a locked holdout set. Use for "make a dev set", "downselect the training set", or "build a benchmark subset".
---

You build the frozen datasets everyone else is graded on, so your
decisions affect every other agent's results. Read `CLAUDE.md`,
`docs/SMARTS_REFERENCE.md`, `docs/PROJECT_CONTEXT.md`, `docs/TASKS.md`, and the latest data-audit
report in `reports/` first.

## Deliverables (in `datasets/`)

- `dev_small.csv`: ~30–50 stars for fast iteration (one scorecard run in
  about a minute).
- `dev_medium.csv`: ~1000 stars, the default for judging method changes.
- `holdout.csv`: ~2000–5000 stars, **disjoint** from both dev sets, used
  only when Rae says so.
- `README.md`: how each set was drawn: seed, strata, exclusions, and the
  per-stratum counts table.

Each manifest has columns `path, star_id, true_period` plus the
stratification variables (amplitude, magnitude/noise, n_sectors, etc.).
Write paths relative to a `SMARTS_DATA_ROOT` placeholder so they survive
the data moving between machines.

## Design rules

- **Stratify on sector coverage** too (task T16f): assign each star a real-star
  sector pattern from the T16a coverage library (single sector, consecutive,
  non-consecutive, full year), store it as a manifest column, and load it with
  `load_smarts_fits(sectors=...)`. Development must exercise gappy data throughout.
- **Stratify** on log-period (at least the evaluation.PERIOD_BIN_LABELS
  bins), signal-to-noise, and baseline. **Oversample** the regimes we care
  about (<10 d is where recovery is weakest) but record the sampling
  weights so population-level rates can be reweighted.
- Include a deliberate share of near-undetectable stars, so "declines to
  answer" is measured too.
- Apply the data auditor's exclusion list for *corrupt* files only. Keep
  hard-but-valid stars.
- Use a fixed seed and write a script (`tools/build_dev_sets.py`) that
  regenerates the manifests exactly.
- Check disjointness and per-stratum counts in a test
  (`tests/test_datasets.py`) that skips cleanly when the manifests or data
  are absent.
- After Rae approves, the manifests are frozen. Changing them later needs
  a new versioned file (`dev_medium_v2.csv`), never an edit.

Finish with the `write-report` skill, including plots comparing each
set's distributions to the full corpus.
