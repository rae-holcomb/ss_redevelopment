# CLAUDE.md — `ss_redevelopment`

This file is read automatically by Claude Code at the start of every
session in this repo. It covers standing conventions and working
preferences. **For project history, architecture, design decisions, and
open issues, read `docs/PROJECT_CONTEXT.md` first — it is the
authoritative, continuously-maintained source of truth for the project
itself.** This file is about *how to work*, not *what the project is*.

For how SMARTS simulations were generated (parameter distributions, FITS
header keys, differential-rotation sign convention, noise model), read
`docs/SMARTS_REFERENCE.md`. Don't re-derive it from the papers.

Also read `docs/METHODOLOGY_NOTES.md` if you need the scientific (rather
than software-engineering) motivation for a design choice, and
`claude_files/FEATURE_DOCUMENTATION.txt` for the ML feature set
reference.

## About Rae

Astrophysicist rewriting SpinSpotter (Holcomb et al. 2022) to recover
stellar rotation periods from SMARTS/TESS light curves at ~1M-star scale,
ultimately feeding a supervised ranking model. Codes primarily in Python.

## Repo layout

- `claude_files/` — the flat pipeline package (all `.py` modules plus
  `FEATURE_DOCUMENTATION.txt`). `claude_files/old/` holds superseded
  versions; don't read from it unless comparing against history.
- `docs/`: `PROJECT_CONTEXT.md`, `METHODOLOGY_NOTES.md`, `SMARTS_REFERENCE.md`, `TASKS.md`
  (task board), `AGENT_WORKFLOW.md` (how Rae runs/reviews agents).
- `tests/`: pytest suite (`python -m pytest -q`, ~5 s, no data needed;
  `-m "not slow"` skips end-to-end tests). Must pass before any commit.
- `benchmarks/`: `scorecard.py` (standard benchmark → `metrics.json` +
  figures), `check_goals.py` (grades against `goals.json` and a baseline),
  `baselines/` (committed reference metrics).
- `datasets/`: frozen dev/holdout manifests (`dev_small`, `dev_medium`, `holdout`; see
  `datasets/README.md`). Dev files are local in `data/smarts_dev/`; score them with
  `--data-root data/smarts_dev` (`--ignore-manifest-sectors` for full-year).
  `datasets/coverage/` holds the real-star TESS sector-pattern libraries with star counts
  (T16a): `cycle<N>_patterns.csv` for TESS years 1–7, plus `all_cycles_patterns.csv`
  pooled (the default). Rebuild them with `tools/build_coverage_library.py`;
  `claude_files/coverage.py` reads and samples them (scorecard `--coverage`).
- `reports/`: one folder per agent task (README + figures + notebook).
- `runs/`: scorecard outputs (gitignored, regenerable).
- `tools/`: helper scripts (`new_report.py`, `run_notebook.py`, `compare_runs.py`,
  `build_coverage_library.py`).
- `.claude/agents/`, `.claude/skills/`: agent role definitions and the
  `write-report` procedure.
- `results_products/` — batch outputs and `run_full_analysis.py`. Result
  CSVs are gitignored.
- `data/` — light curve inputs; **gitignored.** The SMARTS corpus is on an
  external drive; see "SMARTS data location" below.
- Root-level notebooks (`ss_claude_sandbox.ipynb`, `smarts_sandbox.ipynb`,
  `analyze_batch_processing.ipynb`) are Rae's working scratch space.

## Environment

Use `/opt/anaconda3/envs/ss_openblas/bin/python` (env `ss_openblas`,
Python 3.12, numpy 1.26.4 + scipy 1.16.0 **linked against OpenBLAS**;
recipe and pins in `environment.yml`). The Jupyter kernel is registered as
"Python (ss_openblas)" (kernel name `ss_openblas`). Ask Rae before
installing packages.

Why OpenBLAS (T14): with Apple Accelerate, `np.dot`/`np.linalg.norm` results
depend on memory alignment, so the comb fit wasn't bit-reproducible. Scripts
pin BLAS to 1 thread per process via `claude_files/determinism.py`; new
entry-point scripts must call `determinism.pin_blas_threads()` *before*
importing numpy and pass `initializer=determinism.init_worker` to
`ProcessPoolExecutor`. `tests/test_determinism.py` XFAILs (rather than
passes) in a non-OpenBLAS env; if you see that XFAIL, you're in the wrong env.

The old env `ss_local` (Python 3.13, numpy/scipy on Accelerate) still exists
and runs everything, but results are not bit-reproducible there. Base
anaconda Python is 3.8 and lacks lmfit, so don't use it.

## Hardware

**Current main workstation (as of 2026-10-14): Intel Core i5-1038NG7 MacBook,
4 physical cores / 8 threads, x86_64.** Everything is developed and timed
here for now. Rae plans to move back to a 14-core Apple Silicon machine
(10 P-cores + 4 E-cores) later. When that happens, re-time the pipeline and
re-verify scorecard determinism there (task T14 found that Accelerate-BLAS
results depend on memory alignment).

Use parallel processing for batch scripts when reasonable
(`ProcessPoolExecutor`, `--n-workers` CLI override, `--n-workers 1` as a
debugging fallback). Older scripts hardcode `DEFAULT_N_WORKERS=10`, which was
tuned for the Apple Silicon machine and oversubscribes the i5. T14 is
benchmarking 4/8/10 workers and will set the default. Pin BLAS threads to
1 per worker, or OpenBLAS × many workers can run ~50× slower. Batch jobs at
real scale (hundreds to thousands of files) are always run locally by Rae,
not inside a Claude Code session. Timings in reports dated 2026-10-13
(e.g. "1.16 s/star", "322 CPU-hours per 1M stars") were measured on the i5
with 10 workers.

## Working conventions

- **Docstrings**: every function delivered needs a docstring.
- **Plots**: display plots to demonstrate failure modes or changes
  whenever possible. In this environment that typically means saving a
  PNG and either opening it or embedding it in a notebook — confirm
  Rae's preferred mechanism if unclear.
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
  already present in the output — see `batch_test_acf_fft_highpass.py`
  (`process_one_file`) and `run_debug_baseline_full_methods.py`.
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

## Debug workflow

A curated 29-file debug set (periods 0.224–37.4 days, skewed toward short
periods) is the standard testing ground: `smarts-tess-v1.0-????00.fits`.
One file (`smarts-tess-v1.0-000700.fits`, P=0.224d) contains a blended
second periodic signal — keep that in mind if it behaves oddly.
**These files are not currently in the repo working tree** (`data/` is
gitignored and empty locally) — ask Rae where they live before running
anything against them.

- **Quick checks (~5 files): run without asking.**
- **Full 29-file set or larger: ask Rae for confirmation first.** These
  runs take a while and interrupt her flow — this applies doubly in an
  agentic environment like this one where a shell command can just be
  run without a visible prompt each time.
- Anything at real batch scale (hundreds–thousands of files) is run
  locally by Rae, not autonomously in a session.

## Agent sessions

Rae runs several agent sessions in parallel, each on its own worktree
branch, and reviews their work through reports. If you're working on a
task from `docs/TASKS.md`:

- Read your role file in `.claude/agents/` if you were given one.
- **Legibility is the point.** End every task with the `write-report`
  skill. Make plots of failure modes, not just aggregate numbers. Put
  decisions that belong to Rae under "Open questions for Rae" instead of
  making them silently.
- **Don't move the goalposts.** Never edit `benchmarks/goals.json`,
  `benchmarks/scorecard.py`'s metric definitions, `claude_files/evaluation.py`,
  `datasets/` manifests, or existing test assertions as part of a method
  task. Propose changes in the report. Never evaluate on
  `datasets/holdout.csv` unless Rae says so.
- Any claim of improvement needs a before/after `check_goals.py --baseline`
  on the same dataset string.
- Synthetic scorecards (≤ ~200 stars) and `dev_small` are "quick checks";
  `dev_medium` needs Rae's OK; larger is Rae's local job.
- New follow-up ideas go under "Proposed" in `docs/TASKS.md`.

## Performance changes

Before optimizing, profile. Any performance change should be benchmarked
before/after; if a reversion happens (e.g. batched per-scale FFT in
`_cwt_morlet` was tried and found slower than single `fft(data)`),
document the measured rationale in code comments so it isn't re-tried
blind later.

## Version control

GitHub repo `rae-holcomb/ss_redevelopment`. Commit and push notable
changes; keep `docs/PROJECT_CONTEXT.md`'s changelog updated per its own
"How to maintain this document" instructions when you make a notable
decision, resolve an open issue, or discover something future sessions
should know.
