"""Count lines/bytes of the start-up docs and of repo areas at two git revisions (P30 before/after).

Usage (repo root): python reports/2026-11-11_p30-repo-docs-diet/scripts/line_counts.py BEFORE AFTER
Writes data/startup_docs.csv and data/repo_areas.csv in the report folder.
"""
import subprocess, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data"
STARTUP = ["CLAUDE.md", "docs/PROJECT_CONTEXT.md", "docs/TASKS.md"]
AREAS = ["claude_files", "tools", "benchmarks", "docs"]
TEXT_EXT = (".py", ".md", ".txt", ".sh", ".json", ".csv", ".yml", ".ipynb")


def git(*args):
    """Run a git command and return stdout as text."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def blob(rev, path):
    """Return the bytes of `path` at `rev` (b'' if absent)."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else b""


def area_stats(rev, area):
    """Files, total bytes and text lines (.py/.md/... only) tracked under `area` at `rev`."""
    files = [f for f in git("ls-tree", "-r", "--name-only", rev, area).splitlines()]
    nbytes, lines, code_lines = 0, 0, 0
    for f in files:
        b = blob(rev, f)
        nbytes += len(b)
        if f.endswith(TEXT_EXT):
            n = b.count(b"\n")
            lines += n
            if f.endswith((".py", ".md", ".txt", ".sh")):
                code_lines += n
    return dict(files=len(files), bytes=nbytes, text_lines=lines, code_doc_lines=code_lines)


def main(before, after):
    """Write the two CSVs comparing `before` and `after` revisions."""
    OUT.mkdir(exist_ok=True)
    rows = []
    for rev, label in ((before, "before"), (after, "after")):
        for f in STARTUP:
            rows.append(dict(file=f, when=label, lines=blob(rev, f).count(b"\n"), bytes=len(blob(rev, f))))
    pd.DataFrame(rows).to_csv(OUT / "startup_docs.csv", index=False)
    rows = []
    for rev, label in ((before, "before"), (after, "after")):
        for a in AREAS:
            rows.append(dict(area=a, when=label, **area_stats(rev, a)))
    pd.DataFrame(rows).to_csv(OUT / "repo_areas.csv", index=False)
    print(pd.read_csv(OUT / "startup_docs.csv").pivot(index="file", columns="when", values="lines"))
    print(pd.read_csv(OUT / "repo_areas.csv"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
