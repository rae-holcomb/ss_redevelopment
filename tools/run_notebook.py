#!/usr/bin/env python3
"""
run_notebook.py -- execute a notebook in place so its outputs are saved.

Uses nbclient directly rather than `jupyter nbconvert --execute`, because a
stale system-wide config (/usr/local/etc/jupyter/jupyter_nbconvert_config.json
referencing the uninstalled jupyter_contrib_nbextensions) breaks nbconvert
on Rae's machine.

Usage
-----
    python tools/run_notebook.py reports/<folder>/notebook.ipynb [--timeout 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main(argv=None) -> int:
    """Execute the notebook in place with the ss_local kernel, cwd = notebook's folder."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", type=Path)
    p.add_argument("--timeout", type=int, default=600, help="per-cell timeout, seconds")
    p.add_argument("--kernel", default="ss_local")
    args = p.parse_args(argv)

    nb = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(args.notebook.parent.resolve())}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, args.notebook)
    print(f"executed {args.notebook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
