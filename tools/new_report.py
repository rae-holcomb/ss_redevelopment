#!/usr/bin/env python3
"""
new_report.py -- scaffold a report folder following the project's report contract.

Creates reports/YYYY-MM-DD_<slug>/ containing:
    README.md        -- the human-facing write-up (template sections to fill in)
    notebook.ipynb   -- regenerates every figure from saved results; Rae's
                        entry point for poking at the work herself
    figures/         -- PNGs referenced from README.md
    data/            -- small result tables the notebook reads (CSV/JSON);
                        large outputs go in runs/ (gitignored) instead

See .claude/skills/write-report/SKILL.md for what goes in each section.

Usage
-----
    python tools/new_report.py short-period-highpass --title "Improving acf_fft_highpass below 3 d"
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

README_TEMPLATE = """# {title}

**Date:** {date} · **Author:** <agent/session name or Rae> · **Branch:** <git branch> · **Status:** draft

## TL;DR
<!-- 2-4 bullets. Lead with the answer and the headline number(s). -->

## Question / goal
<!-- What was asked, and which goal(s) in benchmarks/goals.json or task in docs/TASKS.md this addresses. -->

## What I did
<!-- Steps, in order. Datasets used (exact scorecard dataset string / manifest), commands run. -->

## Results
<!-- Numbers and figures. Before/after table from check_goals.py --baseline when code changed.
     Every figure: ![caption](figures/xxx.png) followed by one sentence saying what to look at. -->

## What didn't work
<!-- Dead ends and why. This is as valuable as what worked. -->

## Open questions for Rae
<!-- Decisions that are yours to make, things that looked odd, ideas worth trying next. -->

## Reproduce
```bash
# exact commands, from repo root, using /opt/anaconda3/envs/ss_openblas/bin/python
```
"""

SETUP_CELL = """# Setup: make the flat claude_files package importable and load shared helpers.
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path.cwd()
while not (REPO / "claude_files").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "claude_files"))
REPORT = REPO / "reports" / "{folder}"
FIG = REPORT / "figures"
DATA = REPORT / "data"
FIG.mkdir(exist_ok=True)

import evaluation as ev  # project-wide scoring convention (REL_TOL, outcome categories)
"""


def make_notebook(title: str, folder: str) -> nbf.NotebookNode:
    """Build the skeleton notebook for a new report."""
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "ss_openblas", "display_name": "Python (ss_openblas)", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nCompanion notebook to `README.md` in this folder. "
                                 "Running top-to-bottom regenerates every figure in `figures/` from "
                                 "the saved results in `data/` (or the referenced `runs/` directory)."),
        nbf.v4.new_code_cell(SETUP_CELL.format(folder=folder)),
        nbf.v4.new_markdown_cell("## Load results"),
        nbf.v4.new_code_cell("# e.g. per_star = pd.read_csv(REPO / 'runs' / '<run>' / 'per_star.csv')"),
        nbf.v4.new_markdown_cell("## Figures\n\nOne section per figure in the README, same order."),
        nbf.v4.new_code_cell(""),
    ]
    return nb


def main(argv=None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slug", help="short kebab-case name, e.g. data-audit-v1")
    p.add_argument("--title", default=None)
    p.add_argument("--date", default=dt.date.today().isoformat())
    args = p.parse_args(argv)

    folder = f"{args.date}_{args.slug}"
    out = ROOT / "reports" / folder
    if out.exists():
        print(f"{out} already exists; not overwriting.", file=sys.stderr)
        return 1
    (out / "figures").mkdir(parents=True)
    (out / "data").mkdir()
    title = args.title or args.slug.replace("-", " ").capitalize()
    (out / "README.md").write_text(README_TEMPLATE.format(title=title, date=args.date))
    nbf.write(make_notebook(title, folder), out / "notebook.ipynb")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
