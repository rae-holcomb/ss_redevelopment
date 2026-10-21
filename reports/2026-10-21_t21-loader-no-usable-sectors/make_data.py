"""Save the T21 example (smarts-tess-v1.0-828115, sectors=[1, 11]) to data/ for the notebook.

Writes data/example_828115.csv (raw time/flux every 3rd point, with each point's orbit,
sector and flagged status) and data/example_828115_error.txt (the NoUsableSectorsError text).
Run from the repo root: python reports/2026-10-21_t21-loader-no-usable-sectors/make_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
from preprocessing import NoUsableSectorsError, load_smarts_fits  # noqa: E402

PATH = ROOT / "data" / "smarts_sample" / "smarts-tess-v1.0-828115.fits"
OUT = Path(__file__).resolve().parent / "data"


def main():
    """Load the file, record per-point orbit/flag info, and capture the error for sectors=[1, 11]."""
    pre = load_smarts_fits(PATH)
    flagged = pre.diagnostics["segment_flagged"]
    rows = []
    for i, (t, f) in enumerate(pre.segments):
        rows.append(pd.DataFrame(dict(time=t, flux=f, orbit=i + 1, sector=i // 2 + 1,
                                      flagged=bool(flagged[i]))))
    df = pd.concat(rows, ignore_index=True).iloc[::3]
    df.to_csv(OUT / "example_828115.csv", index=False, float_format="%.6f")
    try:
        load_smarts_fits(PATH, sectors=[1, 11])
    except NoUsableSectorsError as exc:
        msg = f"{type(exc).__name__}: {exc}".replace(str(ROOT) + "/", "")
    (OUT / "example_828115_error.txt").write_text(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()
