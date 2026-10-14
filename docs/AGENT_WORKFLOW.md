# Working with agents on this project: Rae's guide

This guide covers how the pieces fit together and the day-to-day routine
for running and supervising agent sessions.

## The pieces

| Piece | Where | What it's for |
|---|---|---|
| Handbook | `CLAUDE.md` | Rules every session reads automatically. |
| Project memory | `docs/PROJECT_CONTEXT.md` | Architecture, decisions, open issues, changelog. |
| Task board | `docs/TASKS.md` | What to work on, with acceptance criteria. You set the priorities. |
| Agent roles | `.claude/agents/*.md` | Job descriptions: data-auditor, dev-set-builder, method-improver, feature-researcher, ranking-model-builder, reviewer. |
| Report procedure | `.claude/skills/write-report/` | How every agent writes up its work. |
| Correctness tests | `tests/` | pytest, about 5 s, no data needed. Must always pass. |
| Scoring convention | `claude_files/evaluation.py` | `REL_TOL=0.10`; outcomes are correct / half / double / wrong / none. |
| Synthetic data | `claude_files/synthetic.py` | Known-period light curves for tests and quick benchmarks. |
| Scorecard | `benchmarks/scorecard.py` | Runs the pipeline on a dataset and writes metrics.json plus figures. |
| Goals | `benchmarks/goals.json` | Thresholds on scorecard metrics. **Only you edit this.** |
| Grader | `benchmarks/check_goals.py` | PASS/FAIL against the goals, plus regression check against a baseline. |
| Reports | `reports/YYYY-MM-DD_<slug>/` | README, figures, executed notebook, data. |
| Run outputs | `runs/` (gitignored) | Raw scorecard outputs. Regenerable. |

Python environment: `/opt/anaconda3/envs/ss_openblas/bin/python`. The Jupyter
kernel is registered as **"Python (ss_openblas)"**.

## Starting an agent on a task

1. Pick a `ready` task in `docs/TASKS.md`.
2. In the Claude app, start a **new session** in this repo **with a worktree**
   (its own branch and copy of the repo, so parallel agents can't collide).
   Or ask any session: "start a session for T5 using the method-improver
   agent".
3. A good opening prompt:
   > You are the **method-improver** agent (read `.claude/agents/method-improver.md`).
   > Your task is **T5** in `docs/TASKS.md`. Work on a branch named `t5-short-period`.
   > Check in with me after the diagnosis step (step 2) before changing code.
4. The "check in after diagnosis" line is worth including at first. It lets
   you redirect before the agent spends hours on the wrong idea.

## Supervising

- You can open any running session to see what it's doing and talk to it
  mid-task.
- Agents will ask before running big jobs. Batch-scale runs come back to
  you as a command to run locally.
- Expect questions under "Open questions for Rae" in reports. That's where
  your scientific judgment gets requested.

## Reviewing finished work (about 10–20 min per task)

1. **Report README**: does the TL;DR answer the question? Do the numbers
   make sense?
2. **Notebook** (`reports/.../notebook.ipynb`, already executed): look at
   the failure-mode plots. Re-run cells and poke at things if anything
   seems off.
3. **Scorecard table**: the `check_goals.py --baseline` output in the report.
   Any `REGRESSION` flags?
4. **Second opinion** (optional, recommended for anything big): in any
   session, say "use the reviewer agent on branch `t5-short-period`". It
   re-runs the numbers and checks for gamed tests.
5. **Diff**: what code actually changed?
6. Merge (or ask the session to open a PR and merge it there). Then mark
   the task `done` and change the report's `Status:` to `reviewed`.

## Setting goals (TDD for science)

A goal is a threshold on a scorecard metric, for example "final.by_bin.<1d.correct
>= 0.80 on synthetic:n=200,seed=0". Workflow:

1. Run a baseline scorecard (`benchmarks/baselines/` holds committed ones).
2. Add a `draft` goal to `benchmarks/goals.json` at a level you think is
   achievable.
3. Flip it to `active` when you want it enforced (`check_goals.py` exits
   non-zero when an active goal fails).
4. Once it passes and is merged, mark it `met`. Keep the regression guard
   so later work can't quietly undo it.

Metric paths you can target are printed by
`python -c "import json; print(json.dumps(json.load(open('runs/<run>/metrics.json')), indent=1))"`.

## Rules of thumb

- **Start with one or two agents at once.** Scale up once reviewing feels
  routine.
- **One writer per file set.** Two agents shouldn't both change `guesses.py`
  at the same time (worktrees prevent clobbering, but you'll get merge
  conflicts). Read-only tasks (audits, research) can run in parallel freely.
- **When an agent annoys you, fix the role file**, not just the session.
  Edit `.claude/agents/<role>.md` or `CLAUDE.md` so the lesson sticks.
- **Synthetic first, real data second.** A synthetic result that doesn't
  hold on dev_small is a finding about `synthetic.py` (task T3).

## Common commands

```bash
/opt/anaconda3/envs/ss_openblas/bin/python -m pytest -q
```
```bash
/opt/anaconda3/envs/ss_openblas/bin/python benchmarks/scorecard.py --synthetic 200 --seed 0 --outdir runs/my_run
```
```bash
/opt/anaconda3/envs/ss_openblas/bin/python benchmarks/check_goals.py runs/my_run/metrics.json --baseline benchmarks/baselines/synth200_s0.json
```
```bash
/opt/anaconda3/envs/ss_openblas/bin/python tools/new_report.py my-investigation --title "My investigation"
```
