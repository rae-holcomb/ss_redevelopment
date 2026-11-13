# Local setup: hardware, SMARTS drive, debug set

Moved verbatim from `CLAUDE.md` on 2026-11-13 to shrink the file every session auto-loads. `CLAUDE.md` keeps the rules; this file has the detail and the reasons. Read it when a task touches timing/parallelism, the Data_Vault drive, or the debug set.

## Hardware

**Current main workstation (as of 2026-10-14): Intel Core i5-1038NG7 MacBook,
4 physical cores / 8 threads, x86_64.** Everything is developed and timed
here for now. Rae plans to move back to a 14-core Apple Silicon machine
(10 P-cores + 4 E-cores) later. When that happens, re-time the pipeline and
re-verify scorecard determinism there (task T14 found that Accelerate-BLAS
results depend on memory alignment).

Use parallel processing for batch scripts when reasonable
(`ProcessPoolExecutor`, `--n-workers` CLI override, `--n-workers 1` as a
debugging fallback). Default worker count is `determinism.default_n_workers()`
(one per physical core, 4 on the i5; T14, re-checked by T12: 6–8 workers add
no throughput beyond noise). `benchmarks/pb_scan.py` still hardcodes 10. Pin BLAS threads to
1 per worker, or OpenBLAS × many workers can run ~50× slower. Batch jobs at
real scale (hundreds to thousands of files) are always run locally by Rae,
not inside a Claude Code session. Timings in reports dated 2026-10-13
(e.g. "1.16 s/star", "322 CPU-hours per 1M stars") were measured on the i5
with 10 workers.

**Machine safety (2026-11-13):** heavy jobs have starved the macOS UI and Rae has been logged out
(WindowServer watchdog) several times. In Claude sessions run every heavy job (scorecards, profiling,
pytest, repro loops) under `nice -n 15` (not `taskpolicy -b`, which throttles to ~half a core), use 3 workers,
never run two heavy jobs at once, check `memory_pressure -Q` / `pmset -g therm` before long runs, and
commit/checkpoint after each scored step.

## Working conventions

- **Docstrings**: every function delivered needs a docstring.
- **Plots (Rae, re-emphasized 2026-11-10):** back up results and decisions
  with plots wherever a plot makes sense. Rae relies on them to understand
  what was done and whether to trust it.
  - Every headline number and every decision put to Rae should have a figure
    behind it: before/after comparisons, distributions with the threshold
    marked, outcome vs period/coverage, and example light curves of failure
    modes (light curve + ACF/periodogram with the candidate periods marked).
  - Save PNGs under the report's `figures/` and **embed them in the README**
    next to the claim they support, not only in the notebook.
  - A report with no figures needs a stated reason.
- **Calculations**: never do arithmetic by hand — always compute in
  Python and return the result, even for something that looks simple.
- **CLI/shell requests**: when Rae asks for a command-line action or
  script, give a single-line zsh command if the task is simple enough.
  If it's more complex, write it as a zsh shell script in a file instead
  of a long one-liner.
- **Introspection over hardcoded lists**: scripts auto-discover
  `guess_*` functions via introspection rather than hardcoded method
  lists — keep this pattern for any similar auto-discovery need.
- **Resumability**: batch runners should be built resumable to support
  long local jobs. The established pattern is a module-level (picklable)
  per-file worker function plus a `--resume` flag that skips files
  already present in the output — see `benchmarks/scorecard.py`
  (`score_one_star`, append-only `rows.jsonl` resume log) and
  `tools/build_feature_tables.py` (`build_one_star`, one atomically written shard per star).
  `ProcessPoolExecutor` cannot pickle closures, so bind extra arguments
  with `functools.partial` around a module-level function rather than
  defining the worker inline.
- **Convention constants**: `REL_TOL=0.10` (changed from 0.15 on
  2026-10-13), defined once in `claude_files/evaluation.py`. Import it
  rather than hardcoding. Measurements within REL_TOL of P/2 or 2P are a
  separate **alias** outcome (`half`/`double`), not lumped with `wrong`.
  A period inside the star's active-latitude-band range (differential
  rotation; truth from SMARTS `DIFFROT`/`MINLAT`/`MAXLAT`) is the `diffrot`
  outcome and counts as **success** (= correct + diffrot); goals target
  success.
  Score periods only via `evaluation.classify_period`/`best_outcome`.
  `n_guesses=10` for most methods (5 for `acf_fft_highpass`).

## SMARTS data location (external drive, often unplugged)

The full SMARTS-TESS DR1 corpus is on an **external drive** at
`/Volumes/Data_Vault/smarts_data/tess_retry/`: 180 period bundles
`hlsp_smarts_tess_ffi_NNN_tess_v1.0_sim.tar.gz` (bundle NNN = periods
NNN−1 to NNN d, ~0.5 GB each, 89 GB total) plus the input-parameter catalog
`hlsp_smarts_tess_ffi_all_tess_v1.0_cat.tar.gz`. **Rae moves the laptop daily,
so the drive is regularly unplugged between work sessions, and possibly
mid-session.** Consequences:

- **Check before use:** test that `/Volumes/Data_Vault` is mounted, and if it
  isn't, stop with a clear message ("plug in the Data_Vault drive"); don't fail
  deep inside a loop. Never assume the mount point is fixed (macOS may mount it
  as `Data_Vault 1`); take the root from a `SMARTS_DATA_ROOT` env var or CLI
  option, defaulting to the path above.
- **Never write extracted data to the laptop's disk in bulk:** the laptop has about
  120 GB free. Extract to the external drive. The exception is a small,
  deliberately chosen **local working set** (e.g. dev_small, a few hundred MB at
  most) kept on the laptop, so development and tests still work when the drive
  is unplugged.
- **Everything that reads the drive must be resumable** (`--resume`, per-file
  outputs or append-only logs), and must write outputs **atomically** (write to
  a temp file, then rename), so an unplug mid-run leaves no corrupt partial files.
- **Manifests store paths relative to `SMARTS_DATA_ROOT`,** never absolute
  `/Volumes/...` paths.
- **Don't start long jobs that need the drive** without telling Rae roughly how
  long they'll take, so she doesn't unplug mid-run.
- Tests must never require the drive. Skip cleanly when it's absent.
- **The local working set lives only in the main checkout's `data/`**
  (`data/smarts_dev/`: dev_small + dev_medium, 1,050 files, 156 MB;
  `data/smarts_sample/`). `data/` is gitignored, so **agent worktrees don't
  get it.** At the start of a task in a worktree, symlink it:
  `ln -s /Users/raeholcomb/Documents/GitHub/ss_redevelopment/data/smarts_dev data/smarts_dev`
  (and the same for `smarts_sample`). Never write new local data inside a worktree:
  it's deleted with the worktree (T2's dev files nearly were, 2026-10-21).

## Debug workflow

A curated 29-file debug set (periods 0.224–37.4 days, skewed toward short
periods) is the standard testing ground: `smarts-tess-v1.0-????00.fits`.
One file (`smarts-tess-v1.0-000700.fits`, P=0.224d) contains a blended
second periodic signal — keep that in mind if it behaves oddly.
**These files are not currently in the repo working tree** (`data/` is
gitignored and empty locally) — ask Rae where they live before running
anything against them. To run the full pipeline on them, use `benchmarks/scorecard.py
--manifest <csv with a path column>` (the old `run_debug_baseline*.py` runners
were retired 2026-11-11).

- **Quick checks (~5 files): run without asking.**
- **Full 29-file set or larger: ask Rae for confirmation first.** These
  runs take a while and interrupt her flow — this applies doubly in an
  agentic environment like this one where a shell command can just be
  run without a visible prompt each time.
- Anything at real batch scale (hundreds–thousands of files) is run
  locally by Rae, not autonomously in a session.
