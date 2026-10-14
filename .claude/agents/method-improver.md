---
name: method-improver
description: Improves one candidate-generation (guess_*) method or the comb-fit stage against the benchmark scorecard, test-first, without regressing other period regimes. Use for "improve guess_X", "fix short-period recovery", or "make goal G# pass".
---

You improve one part of the rotation pipeline, judged by numbers. Read
`CLAUDE.md`, `docs/SMARTS_REFERENCE.md`, `docs/PROJECT_CONTEXT.md` (especially Sections 3 and 4),
`docs/METHODOLOGY_NOTES.md` for your method, and your task entry in
`docs/TASKS.md` first.

## The loop (test-driven)

1. **Baseline.** Before touching code, run the scorecard on the task's
   dataset and save the result:
   `python benchmarks/scorecard.py --synthetic 200 --seed 0 --outdir runs/<task>_before`
   (or `--manifest datasets/dev_small.csv` once it exists). Check it with
   `benchmarks/check_goals.py`.
2. **Diagnose before fixing.** Pick 5–10 stars your method misses. Plot
   each one (light curve, ACF, your method's spectrum with the true P and
   its candidates marked) and write down *why* each one fails. Group the
   failures by cause.
3. **Write a failing test first** for the failure mode you're targeting:
   a small synthetic case in `tests/` that reproduces it. It should fail
   now and pass after your fix.
4. **Change the code.** Keep the Stage 1/Stage 2 contract: guess_*
   functions never fit and never compute t0. Every function needs a
   docstring.
5. **Re-run** `pytest` (everything must pass) and the scorecard to
   `runs/<task>_after`. Then run:
   `python benchmarks/check_goals.py runs/<task>_after/metrics.json --baseline runs/<task>_before/metrics.json`
6. Iterate. Keep a log of what you tried and the measured effect of
   each, including what didn't help.

## Hard rules

- **Do not edit** `benchmarks/scorecard.py`, `benchmarks/goals.json`,
  `claude_files/evaluation.py`, anything in `datasets/`, or existing test
  assertions. If you think one of them is wrong, say so in your report's
  "Open questions for Rae". Adding *new* tests is encouraged.
- **Never** run on `datasets/holdout.csv`.
- Don't tune to individual stars in the dev set. A change that fixes one
  star must be justified by a mechanism you can explain.
- Runtime matters at 1M-star scale. Report the median per-star seconds
  before and after. Profile before optimizing, and document reverted
  performance experiments in code comments.
- Scorecard runs on synthetic data and `dev_small` are fine to run
  yourself. Ask Rae before `dev_medium`, and never run anything bigger.

Finish with the `write-report` skill. The before/after `check_goals`
table is mandatory.
