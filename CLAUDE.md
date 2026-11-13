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
  `FEATURE_DOCUMENTATION.txt`). Superseded versions live only in git history.
- `docs/`: `PROJECT_CONTEXT.md` (+ `CHANGELOG_ARCHIVE.md`), `METHODOLOGY_NOTES.md`,
  `SMARTS_REFERENCE.md`, `LOCAL_SETUP.md` (hardware, drive, debug set), `AGENT_WORKFLOW.md`,
  `FEATURE_TABLE_SCHEMA.md`, `HANDOFF.md`. Task board: `TASKS.md` (roadmap + active),
  `TASKS_PROPOSED.md` (P##), `TASKS_DISCUSSIONS.md` (D##), `TASKS_DONE.md`.
  **Read only the section you need** (e.g. your own task); open the archives
  (`TASKS_DONE.md`, `CHANGELOG_ARCHIVE.md`) and old `reports/` only for history or ambiguity.
- `tests/`: pytest suite (`python -m pytest -q`, ~5 s, no data needed;
  `-m "not slow"` skips end-to-end tests). Must pass before any commit.
- `benchmarks/`: `scorecard.py` (standard benchmark → `metrics.json` +
  figures; also the batch runner for any manifest, with per-method Stage-1
  results in `per_method.csv`), `check_goals.py` (grades against `goals.json`
  and a baseline), `baselines/` (committed reference metrics; use the `_v7` files).
- `datasets/`: frozen dev/holdout manifests (`dev_small`, `dev_medium`, `holdout`; see
  `datasets/README.md`). Dev files are local in `data/smarts_dev/`; score them with
  `--data-root data/smarts_dev` (`--ignore-manifest-sectors` for full-year).
  `datasets/coverage/`: real-star TESS sector-pattern libraries (see its README).
- `reports/`: one folder per agent task (README + figures + notebook).
- `runs/`: scorecard outputs (gitignored, regenerable).
- `tools/`: helper scripts (`build_feature_tables.py` and `train_ranker.py` for the
  ranker, dataset/side-table builders `build_*.py`, `audit/` for data audits,
  `new_report.py`, `run_notebook.py`, `compare_runs.py`).
- `.claude/agents/`, `.claude/skills/`: agent role definitions and the
  `write-report` procedure.
- `results_products/` — old batch outputs and `run_full_analysis.py`, which reads
  the `summary.csv` of the retired `batch_test_guesses.py` (existing outputs still
  work; new ones can't be made). Result CSVs are gitignored and untracked (2026-11-13).
- `data/` — light curve inputs; **gitignored.** The SMARTS corpus is on an
  external drive; see "Hardware, data drive, debug set" below.
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

## Hardware, data drive, debug set (rules; detail in `docs/LOCAL_SETUP.md`)

- **Machine:** 4-core Intel i5 MacBook (since 2026-10-14). Default workers =
  `determinism.default_n_workers()`; pin BLAS to 1 thread per worker. Batch runners
  use `ProcessPoolExecutor` with a `--n-workers` override (`1` for debugging).
- **Machine safety (2026-11-13):** heavy jobs have logged Rae out by starving the UI.
  Run every heavy job (scorecards, profiling, pytest, repro loops) under `nice -n 15`
  (not `taskpolicy -b`), use 3 workers, never two heavy jobs at once, check
  `memory_pressure -Q` / `pmset -g therm` before long runs, checkpoint after each scored step.
- **SMARTS corpus** is on an often-unplugged external drive
  (`/Volumes/Data_Vault/smarts_data/tess_retry/`, 89 GB; root from `SMARTS_DATA_ROOT`).
  Check the mount first and stop with "plug in the Data_Vault drive"; never extract in bulk
  to the laptop (~120 GB free); anything reading the drive is resumable and writes atomically;
  manifests store paths relative to `SMARTS_DATA_ROOT`; tests never require the drive; tell Rae
  how long a drive job will take before starting it.
- **Local working set** (`data/smarts_dev/`, `data/smarts_sample/`) lives only in the main
  checkout and is gitignored. In a worktree, symlink it:
  `ln -s /Users/raeholcomb/Documents/GitHub/ss_redevelopment/data/smarts_dev data/smarts_dev`
  (same for `smarts_sample`). Never write new local data inside a worktree.
- **Run sizes:** ~5 files, run freely. The 29-file debug set or larger: ask Rae first.
  Hundreds to thousands of files: Rae runs them locally, not a session.

## Agent sessions

Rae runs several agent sessions in parallel, each on its own worktree
branch, and reviews their work through reports. If you're working on a
task from `docs/TASKS.md`:

- Read your role file in `.claude/agents/` if you were given one.
- **Legibility is the point.** End every task with the `write-report`
  skill. Make plots of failure modes, not just aggregate numbers (see
  "Plots" above: figures are expected, not optional). Put
  decisions that belong to Rae under "Open questions for Rae" instead of
  making them silently.
- **Don't move the goalposts.** Never edit `benchmarks/goals.json`,
  `benchmarks/scorecard.py`'s metric definitions, `claude_files/evaluation.py`,
  `datasets/` manifests, or existing test assertions as part of a method
  task. Propose changes in the report. Never evaluate on
  `datasets/holdout.csv` unless Rae says so.
- Any claim of improvement needs a before/after `check_goals.py --baseline`
  on the same dataset string.
- Synthetic scorecards (≤ ~200 stars) and `dev_small` are "quick checks".
  **dev_medium full-year is the default judging set for method changes**
  (`--manifest datasets/dev_medium.csv --data-root data/smarts_dev --ignore-manifest-sectors`,
  ~7.5 min full-year / ~5.5 min assigned at 3 workers since T31; Rae's standing OK, 2026-10-21; baseline
  `benchmarks/baselines/dev_medium_fullyear_v7.json`; `_assigned_v7` for assigned). Use assigned coverage (no
  `--ignore-manifest-sectors`) for coverage-robustness work. Never run the holdout
  without Rae; anything larger than dev_medium is Rae's local job.
- New follow-up ideas go at the end of `docs/TASKS_PROPOSED.md`.

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
