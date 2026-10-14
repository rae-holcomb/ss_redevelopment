#!/usr/bin/env python3
"""
make_data.py -- regenerate the small data tables the T14 notebook plots.

Subcommands (run from the repo root):

    # 1. memory-alignment sensitivity of BLAS primitives, run once PER ENV:
    /opt/anaconda3/envs/ss_local/bin/python    reports/2026-10-14_t14-bit-reproducible-scorecards/make_data.py alignment
    /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-10-14_t14-bit-reproducible-scorecards/make_data.py alignment

    # 2. run-to-run final-period differences + runtime table from runs/ dirs:
    /opt/anaconda3/envs/ss_openblas/bin/python reports/2026-10-14_t14-bit-reproducible-scorecards/make_data.py runs

Outputs go to data/ next to this script:
    alignment_<backend>.csv   one row per (primitive, vector length): fraction
                              of random inputs whose result changes with the
                              buffer's 8-byte offset inside a 64-byte line
    run_pairs.csv             one row per (pair of runs, star): both final
                              periods and the relative difference
    timing.csv                one row per timing run: env, workers, wall time,
                              median/p90 per-star seconds
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
sys.path.insert(0, str(ROOT / "claude_files"))

import determinism  # noqa: E402
determinism.pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _at_offset(a: np.ndarray, off: int) -> np.ndarray:
    """Copy `a` into a fresh buffer starting `off` bytes past a 64-byte boundary."""
    buf = np.empty(a.nbytes + 128, dtype=np.uint8)
    s = (-buf.ctypes.data) % 64 + off
    out = buf[s:s + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    return out


def _h(x) -> str:
    """Bitwise hash of an array or tuple of arrays."""
    xs = x if isinstance(x, tuple) else (x,)
    return hashlib.md5(b"".join(np.ascontiguousarray(np.asarray(v)).tobytes() for v in xs)).hexdigest()


def alignment(n_trials: int = 40) -> Path:
    """Measure how often each primitive's result depends on memory alignment.

    For each primitive and size, draw `n_trials` random inputs; an input
    counts as sensitive if computing the primitive on bitwise-identical
    copies at the 8 possible 8-byte offsets gives more than one distinct
    result.
    """
    import scipy.linalg as sl
    rng = np.random.default_rng(0)
    prims = {
        "np.dot(v, v)  [BLAS ddot]": ("vec", lambda v: np.dot(v, v)),
        "np.linalg.norm(v)": ("vec", lambda v: np.linalg.norm(v)),
        "np.sum(v * v)  [numpy, no BLAS]": ("vec", lambda v: np.sum(v * v)),
        "J.T @ f  [BLAS gemv]": ("mat", lambda J: J.T @ np.linspace(0, 1, len(J))),
        "scipy.linalg.svd(J)": ("mat", lambda J: sl.svd(J, full_matrices=False)),
    }
    rows = []
    for name, (kind, f) in prims.items():
        for n in (8, 25, 100, 1000):
            sens = 0
            for _ in range(n_trials):
                x = rng.normal(size=n) if kind == "vec" else rng.normal(size=(n, 26))
                sens += len({_h(f(_at_offset(x, o))) for o in range(0, 64, 8)}) > 1
            rows.append(dict(primitive=name, n=n, frac_sensitive=sens / n_trials, n_trials=n_trials))
    backend = determinism.blas_backend()
    out = DATA / f"alignment_{backend}.csv"
    pd.DataFrame(rows).assign(backend=backend).to_csv(out, index=False)
    print(f"wrote {out}")
    return out


# (label, env/backend, runs/ directory, n_workers) for every scorecard run used.
RUNS_200 = [
    (f"{side}_w{nw}{suffix}", backend, f"runs/t14_{side}_w{nw}{suffix}", nw)
    for side, backend in (("before", "accelerate"), ("after", "openblas"))
    for suffix in ("", "_pass2")
    for nw in (4, 8, 10)
] + [("after_w4_repeat", "openblas", "runs/t14_after_w4_repeat", 4)]
# Timing of after_w4_repeat is contaminated (another session's scorecard started
# during it, see runs/t14_timing.log); it is kept for the determinism comparison only.
TIMING_CONTAMINATED = {"after_w4_repeat"}


def runs() -> None:
    """Write run_pairs.csv (all within-backend pairs of 200-star runs) and timing.csv."""
    have = [r for r in RUNS_200 if (ROOT / r[2] / "per_star.csv").exists()]
    star = {lab: pd.read_csv(ROOT / d / "per_star.csv").set_index("star_id").sort_index()
            for lab, _, d, _ in have}
    rows = []
    for (la, ba, _, _), (lb, bb, _, _) in itertools.combinations(have, 2):
        if ba != bb:
            continue
        a, b = star[la], star[lb].loc[star[la].index]
        for sid in a.index:
            pa, pb = a.at[sid, "final_period"], b.at[sid, "final_period"]
            both_nan = pd.isna(pa) and pd.isna(pb)
            rel = 0.0 if both_nan or pa == pb else (abs(pa - pb) / abs(pa) if not (pd.isna(pa) or pd.isna(pb)) else np.inf)
            rows.append(dict(backend=ba, pair=f"{la} vs {lb}", star_id=sid, true_period=a.at[sid, "true_period"],
                             P_a=pa, P_b=pb, rel_diff=rel,
                             outcome_a=a.at[sid, "final_outcome"], outcome_b=b.at[sid, "final_outcome"]))
    pd.DataFrame(rows).to_csv(DATA / "run_pairs.csv", index=False)
    t = []
    for lab, backend, d, nw in have:
        m = json.loads((ROOT / d / "metrics.json").read_text())
        t.append(dict(run=lab, backend=backend, n_workers=nw, wall_seconds=m["config"]["wall_seconds"],
                      median_star_seconds=m["runtime"]["median_seconds"], p90_star_seconds=m["runtime"]["p90_seconds"],
                      git_commit=m["config"]["git_commit"],
                      timing_pass=2 if "pass2" in lab else 1, timing_clean=lab not in TIMING_CONTAMINATED))
    pd.DataFrame(t).to_csv(DATA / "timing.csv", index=False)
    print(f"wrote {DATA/'run_pairs.csv'} ({len(rows)} rows) and {DATA/'timing.csv'} ({len(t)} runs)")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    {"alignment": alignment, "runs": runs}[sys.argv[1]]()
