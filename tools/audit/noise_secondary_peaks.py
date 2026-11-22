"""P65: top-N Lomb-Scargle peaks of every noise-source stack, each with its time-split S/N.

The T20 flags keep one period per noise source (``P_ref``, the stack's top LS peak) and the
2026-11-10 time-split test (``tsplit_min_snr``) is evaluated only there. A coherent signal that is
*not* the top peak (for example a ~1 d line under a stronger 26 d red-noise bump) is invisible to
both. This script lists the top ``--n-peaks`` local maxima of each stack's full-year LS (0.1-60 d)
and, at each, the local LS S/N in each half of the year, using the same statistic as
``galaxy_phase2_analysis.time_split_snr`` (power at f0 over the median power at 3-25% fractional
frequency offset; halves split at t = 182.5 d). Each stack is first written to a temporary
SMARTS-layout FITS file and read back with the production loader (``load_smarts_fits``: gap
re-detection, bad-orbit flagging), exactly as ``galaxy_periods.py ls`` does for ``P_ref``, so rank 1
here reproduces ``P1_all`` and the S/N values describe the light curve the pipeline actually sees.

Reads a *local copy* of the drive's ``extracted/galaxy_stack/{stacks.npy,stack_index.csv}`` (copy them
once, sequentially; never read the drive in parallel). Single process, ~2-4 min for 1,719 stacks.

    $PY tools/audit/noise_secondary_peaks.py --stack-dir <copy> --out reports/<folder>/data/stack_peaks.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))
import determinism  # noqa: E402

if __name__ == "__main__":
    determinism.pin_blas_threads(1)

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

from galaxy_periods import write_fits  # noqa: E402

T_SPLIT = 182.5
F_MIN, F_MAX = 1 / 60, 1 / 0.1
# Same frequency grids as time_split_snr (half: 182 d baseline x 5 oversampling); full year 2x denser.
F_HALF = np.linspace(F_MIN, F_MAX, int((F_MAX - F_MIN) * 182 * 5))
F_FULL = np.linspace(F_MIN, F_MAX, int((F_MAX - F_MIN) * 365 * 5))
SEP = 0.02   # peaks closer than 2% in frequency to a higher one are the same peak


def load_stack(g: np.ndarray, tic: int, tmp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Round-trip one stack through a temporary FITS file and the production loader."""
    import warnings

    from preprocessing import load_smarts_fits
    write_fits(tmp, g, tic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = load_smarts_fits(str(tmp))
    ok = np.isfinite(lc.flux)
    return lc.time[ok], lc.flux[ok]


def local_snr(ls: LombScargle, p: np.ndarray, f: np.ndarray, f0: float) -> float:
    """LS power at f0 over the median of ``p`` at 3-25% fractional offset from f0."""
    band = (np.abs(f / f0 - 1) < 0.25) & (np.abs(f / f0 - 1) > 0.03)
    return float(ls.power(f0) / np.median(p[band])) if band.sum() > 5 else np.nan


def top_peaks(p: np.ndarray, f: np.ndarray, n: int) -> list[int]:
    """Indices of the ``n`` highest local maxima of ``p``, at least SEP apart in fractional frequency."""
    loc = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    out: list[int] = []
    for j in loc[np.argsort(p[loc])[::-1]]:
        if all(abs(f[j] / f[i] - 1) > SEP for i in out):
            out.append(int(j))
        if len(out) == n:
            break
    return out


def continuum(p: np.ndarray, f: np.ndarray, step: float = 0.01) -> np.ndarray:
    """The ``local_snr`` denominator on a coarse log-frequency grid (median of ``p`` at 3-25% offset), interpolated.

    A narrower running median (e.g. +-5%) is wrong here: some noise-source signals are broad quasi-periodic
    humps ~+-3% wide (TIC 10001638867 at 1.39 d), which then raise their own continuum and vanish.
    """
    g = np.exp(np.arange(np.log(f[0]), np.log(f[-1]), step))
    med = []
    for f0 in g:   # f is sorted: slice the +-25% window first (40x faster than masking the whole grid)
        lo, hi = np.searchsorted(f, [0.75 * f0, 1.25 * f0])
        x = np.abs(f[lo:hi] / f0 - 1)
        band = x > 0.03
        med.append(np.median(p[lo:hi][band]) if band.sum() > 5 else np.nan)
    med = np.asarray(med)
    ok = np.isfinite(med)
    return np.interp(np.log(f), np.log(g[ok]), med[ok])


def line_peaks(p: np.ndarray, f: np.ndarray, n: int, p_max: float) -> list[int]:
    """The ``n`` most line-like local maxima at P <= ``p_max`` d: ranked by power over the local continuum."""
    r = p / continuum(p, f)
    r = np.where(f >= 1 / p_max, r, 0.0)
    return top_peaks(r, f, n)


class StackLS:
    """Full-year and per-half LS of one stack, with S/N lookups at arbitrary periods."""

    def __init__(self, t: np.ndarray, y: np.ndarray):
        """``t``, ``y``: the loaded light curve's finite cadences."""
        t, y = np.asarray(t, np.float64), np.asarray(y, np.float64)
        self.full = LombScargle(t, y - np.median(y))
        self.p_full = self.full.power(F_FULL, method="fast")
        self.halves = []
        for m in (t < T_SPLIT, t >= T_SPLIT):
            mm = m
            if mm.sum() < 500:  # same minimum as time_split_snr
                self.halves.append(None)
                continue
            ls = LombScargle(t[mm], y[mm] - np.median(y[mm]))
            self.halves.append((ls, ls.power(F_HALF, method="fast")))

    def snr(self, P: float) -> dict:
        """Full-year and per-half local S/N at period P (d), and the full-year LS power there."""
        f0 = 1 / P
        r = dict(power=float(self.full.power(f0)), snr_full=local_snr(self.full, self.p_full, F_FULL, f0))
        for k, h in zip(("snr_h1", "snr_h2"), self.halves):
            r[k] = np.nan if h is None else local_snr(h[0], h[1], F_HALF, f0)
        r["min_half_snr"] = float(np.nanmin([r["snr_h1"], r["snr_h2"]])) if np.isfinite(
            [r["snr_h1"], r["snr_h2"]]).all() else np.nan
        return r


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack-dir", required=True, help="local copy holding stacks.npy and stack_index.csv")
    ap.add_argument("--n-peaks", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp-dir", required=True, help="scratch folder for the temporary FITS file")
    ap.add_argument("--n-lines", type=int, default=5, help="most line-like peaks at P <= --line-pmax")
    ap.add_argument("--line-pmax", type=float, default=5.0)
    ap.add_argument("--query", help="CSV with columns key, tic, P: S/N of each tic's stack at P, and at P/2 and 2P")
    ap.add_argument("--query-out")
    ap.add_argument("--spectra-tics", type=int, nargs="*", default=[],
                    help="save these sources' full-year and half-year LS power to <out>_spectra.npz")
    a = ap.parse_args(argv)
    sd = Path(a.stack_dir)
    idx = pd.read_csv(sd / "stack_index.csv")
    S = np.load(sd / "stacks.npy", mmap_mode="r")
    tmp = Path(a.tmp_dir) / "stack_tmp.fits"
    q = pd.read_csv(a.query) if a.query else pd.DataFrame(columns=["key", "tic", "P"])
    rows, qrows, spec = [], [], {}
    for i, r in idx.iterrows():
        tmp.unlink(missing_ok=True)
        try:
            st = StackLS(*load_stack(S[i, 2], r.tic, tmp))
        except Exception as e:  # noqa: BLE001 -- e.g. every orbit flagged
            print(r.galaxy, type(e).__name__, e)
            continue
        for k, j in enumerate(top_peaks(st.p_full, F_FULL, a.n_peaks), 1):
            P = 1 / F_FULL[j]
            rows.append(dict(galaxy=r.galaxy, tic=r.tic, kind="power", rank=k, P=P, **st.snr(P)))
        for k, j in enumerate(line_peaks(st.p_full, F_FULL, a.n_lines, a.line_pmax), 1):
            P = 1 / F_FULL[j]
            rows.append(dict(galaxy=r.galaxy, tic=r.tic, kind="line", rank=k, P=P, **st.snr(P)))
        for _, qq in q[q.tic == r.tic].iterrows():
            for mult in (1.0, 0.5, 2.0):
                P = qq.P * mult
                if 0.1 <= P <= 60:
                    qrows.append(dict(key=qq.key, tic=r.tic, P_query=qq.P, mult=mult, P=P, **st.snr(P)))
        if r.tic in a.spectra_tics:
            spec[int(r.tic)] = (st.p_full.astype(np.float32),
                                np.array([np.full(len(F_HALF), np.nan) if h is None else h[1] for h in st.halves],
                                         np.float32))
        if i % 200 == 0:
            print(i, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    if a.query:
        pd.DataFrame(qrows).to_csv(a.query_out, index=False)
    if spec:
        tics = sorted(spec)
        np.savez_compressed(str(a.out).replace(".csv", "_spectra.npz"), tic=np.array(tics), f_full=F_FULL,
                            f_half=F_HALF, p_full=np.array([spec[k][0] for k in tics]),
                            p_half=np.array([spec[k][1] for k in tics]))


if __name__ == "__main__":
    main()
