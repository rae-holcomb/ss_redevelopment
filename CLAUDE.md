# CLAUDE.md — `ss_redevelopment`

This file is read automatically by Claude Code at the start of every
session in this repo. It covers standing conventions and working
preferences. **For project history, architecture, design decisions, and
open issues, read `docs/PROJECT_CONTEXT.md` first — it is the
authoritative, continuously-maintained source of truth for the project
itself.** This file is about *how to work*, not *what the project is*.

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
- `docs/` — `PROJECT_CONTEXT.md`, `METHODOLOGY_NOTES.md`.
- `results_products/` — batch outputs and `run_full_analysis.py`. Result
  CSVs are gitignored.
- `data/` — light curve inputs; **gitignored and not currently present
  locally.** Ask Rae for the path before assuming a debug set exists.
- Root-level notebooks (`ss_claude_sandbox.ipynb`, `smarts_sandbox.ipynb`,
  `analyze_batch_processing.ipynb`) are Rae's working scratch space.

## Hardware

14-core machine: 10 Performance cores, 4 Efficiency cores.
Use parallel processing for batch scripts when reasonable
(`ProcessPoolExecutor`, `DEFAULT_N_WORKERS=10` targeting P-cores,
`--n-workers 1` as a debugging fallback, `--n-workers` as a CLI override
generally). Batch jobs at real scale (hundreds to thousands of files) are
always run locally by Rae, not inside a Claude Code session.

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
- **Convention constants**: `rel_tol=0.15` across scripts; `n_guesses=10`
  for most methods (5 for `acf_fft_highpass`).

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
