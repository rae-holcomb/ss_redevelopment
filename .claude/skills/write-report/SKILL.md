---
name: write-report
description: Write up any investigation, experiment, data audit, or code change for Rae using the project's report contract (README + figures + executed notebook + metrics). Use at the end of every agent task in this repo, and whenever results need to be shown to Rae.
---

# Writing a report for Rae

Rae supervises several agents at once. She judges your work by reading the
report first, checking it against the notebook, then reading the diff, so
the report has to stand on its own and has to be honest.

## 1. Scaffold

```bash
/opt/anaconda3/envs/ss_local/bin/python tools/new_report.py <kebab-slug> --title "<Title>"
```

This creates `reports/YYYY-MM-DD_<slug>/` with `README.md`, `notebook.ipynb`,
`figures/`, `data/`. Never hand-roll a different layout.

## 2. Fill the README (every section, in order)

- **TL;DR**: the answer first, with the headline number. "acf_fft_highpass
  recall below 1 d went 0.62 → 0.81 on synthetic n=200 seed 0; no
  regressions." Not "I investigated...".
- **Question / goal**: name the task in `docs/TASKS.md` and the goal IDs from
  `benchmarks/goals.json`.
- **What I did**: datasets by their exact scorecard dataset string, and the
  commands you ran.
- **Results**:
  - For any code change: paste the output of
    `benchmarks/check_goals.py <new>/metrics.json --baseline <old>/metrics.json`.
  - For every figure: embed it, then add one sentence telling Rae what to
    look at.
  - Show failure modes, not just aggregates: pick 3–6 representative stars
    that still fail and plot them (light curve, ACF with candidate periods
    marked, method spectra). `claude_files/plotting.py` has helpers.
- **What didn't work**: dead ends, with the measured reason.
- **Open questions for Rae**: anything that's her decision (thresholds,
  goals, scientific judgment calls), anything that looked suspicious, and
  ideas you didn't pursue.
- **Reproduce**: exact commands from the repo root.

## 3. Notebook

- The notebook must regenerate every README figure from saved data
  (`data/` for small tables, or a named `runs/` directory). Keep heavy
  computation in scripts, not the notebook.
- Execute it so Rae sees outputs without running anything:
  ```bash
  /opt/anaconda3/envs/ss_local/bin/python tools/run_notebook.py reports/<folder>/notebook.ipynb
  ```
- Put markdown cells between figures explaining what each one shows.

## 4. Honesty rules

- Report negative and null results plainly. A clear "this didn't help, here's
  why" is a successful report.
- Never claim an improvement without a before/after scorecard on the same
  dataset string and seed.
- Quote numbers from computed output. Never do arithmetic by hand.
- If you deviated from the task brief, say so in the TL;DR.
- Mark the README `Status:` as `draft` until Rae has reviewed it.

## 5. Finish

- Add a terse dated changelog entry to `docs/PROJECT_CONTEXT.md` pointing at
  the report folder. Update its Section 4 if you resolved or found an open
  issue.
- Update the task's status line in `docs/TASKS.md`.
- Commit the report together with the code it describes, on your task branch.
