"""T20 follow-up (f): do independent FFI light curves (QLP / TASOC / ELEANOR-LITE) show the SMARTS noise-source periods?

Reads ``<ffi_dir>/ffi_index_<selection stem>.csv`` and ``lc/*.csv`` from ``ffi_lc_check.py`` and
the selection (with each source's reference period ``P_ref`` from the SMARTS
stack). Each sector is median-normalized, detrended with a running median of
``4 P_ref`` (0.5-5 d) and 5-sigma clipped; then, per sector and stitched,
Lomb-Scargle over 0.1 d to the detrending window: the top peak, the best match
of the top three peaks to P_ref, P_ref/2 or 2 P_ref (within 3%), the
semi-amplitude at P_ref. ``confirmed`` = a stitched top-3 peak matches and
P_ref < 5 d (longer periods are untestable in single 27-d sectors with
pipeline trends: the quiet controls "match" at 13-15 d by chance).

Writes ``ffi_compare.csv`` (per TIC), ``ffi_compare_sectors.csv`` (per light
curve) and ``ffi_compare_power.npz`` (stitched periodograms and 100-bin phase
curves folded at P_ref, for figures) to ``--outdir``.

Usage::

    $PY tools/audit/ffi_lc_compare.py --make-selection reports/<folder>/data/sources.csv \
        --select reports/<folder>/data/ffi_selection.csv
    $PY tools/audit/ffi_lc_compare.py --ffi-dir /Volumes/Data_Vault/smarts_data/extracted/ffi_check \\
        --select reports/<folder>/data/ffi_selection.csv --outdir reports/<folder>/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

PMIN, PMAX = 0.1, 30.0
MATCH_TOL = 0.03
FREQ = np.linspace(1 / PMAX, 1 / PMIN, 20000)   # df = 5e-4 /d: > 4x oversampled for 5 sectors (~130 d)


TESTABLE_PMAX = 5.0   # single 27-d sectors with pipeline trends can't confirm longer periods


def detrend_window(P: float) -> float:
    """Running-median window (d) used to remove slow trends before testing period P."""
    return float(np.clip(4 * P, 0.5, 5.0))


def clean(t: np.ndarray, f: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Median-normalize, subtract a running median of ``window`` days, then 5-sigma (MAD) clip one sector."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok] / np.median(f[ok])
    o = np.argsort(t)
    t, f = t[o], f[o]
    n = max(3, int(window / np.median(np.diff(t))) | 1)
    f = f - median_filter(f, size=n, mode="nearest") + 1
    s = 1.4826 * np.median(np.abs(f - 1))
    keep = np.abs(f - 1) < 5 * s if s > 0 else np.ones(len(f), bool)
    return t[keep], f[keep] - 1


def top_peaks(p: np.ndarray, n: int = 3, pmax: float = PMAX) -> list:
    """Periods of the ``n`` highest local maxima (periods <= ``pmax``) of a periodogram on FREQ, > 2% apart."""
    loc = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    loc = loc[1 / FREQ[loc] <= pmax]
    out = []
    for k in loc[np.argsort(-p[loc])]:
        if all(abs(1 / FREQ[k] - q) / q > 0.02 for q in out):
            out.append(1 / FREQ[k])
        if len(out) == n:
            break
    return out


def match(peaks: list, P: float) -> str:
    """'P', 'P/2' or '2P' if any peak is within MATCH_TOL of that multiple of P, else ''."""
    for lab, q in (("P", P), ("P/2", P / 2), ("2P", 2 * P)):
        if any(abs(x - q) / q < MATCH_TOL for x in peaks):
            return lab
    return ""


def analyse(t: np.ndarray, y: np.ndarray, P: float, pmax: float) -> dict:
    """LS on (t, y): power array, top-3 peaks below ``pmax``, match to P, semi-amplitude at P (ppm)."""
    ls = LombScargle(t, y)
    p = ls.power(FREQ, method="fast")
    peaks = top_peaks(p, pmax=pmax)
    m = ls.model(t, 1 / P) if PMIN <= P <= PMAX else np.zeros(1)
    return dict(power=p, peaks=peaks, match=match(peaks, P), amp_at_P_ppm=float(np.ptp(m) / 2 * 1e6),
                pow_at_P=float(ls.power(1 / P)) if PMIN <= P <= PMAX else np.nan)


def make_selection(sources_csv: Path, out_csv: Path) -> pd.DataFrame:
    """The 19-source FFI check list: top periodic sources per class, two 0.83-d family members, 4 quiet controls."""
    src = pd.read_csv(sources_csv)
    pp = src[src.periodic]
    top = lambda c, n, key="pow1_all": pp[pp.classification == c].nlargest(n, key)
    pick = [top("source_specific_unexplained", 5), top("contamination_known_variable", 3, "amp1_ppm_all"),
            top("contamination_candidate_crowded", 2), top("systematic_common_mode", 2),
            top("misclassified_star", 1), src[src.tic.isin([10002480867, 10002480866])],
            src[src.classification == "quiet"].sample(4, random_state=0)]
    sel = pd.concat(pick).drop_duplicates("tic")[["tic", "galaxy", "classification", "P_ref", "pipe_P",
                                                  "P1_all", "amp1_ppm_all", "pow1_all"]]
    sel.to_csv(out_csv, index=False)
    return sel


def make_selection_all(sources_csv: Path, out_csv: Path, n_null: int = 40, seed: int = 1) -> pd.DataFrame:
    """Every periodic source, plus ``n_null`` quiet sources tested at periods drawn from the periodic ones.

    The quiet controls get ``P_ref`` = a random periodic source's ``P_ref``
    (their own value is kept as ``P_ref_own``), so the fraction of them that
    "confirm" is the chance-match rate of the test at the same period mix.
    """
    src = pd.read_csv(sources_csv)
    cols = ["tic", "galaxy", "classification", "P_ref", "pipe_P", "P1_all", "amp1_ppm_all", "pow1_all"]
    pp = src.loc[src.periodic, cols].assign(control=False)
    rng = np.random.default_rng(seed)
    q = src[src.classification == "quiet"].sample(n_null, random_state=seed)[cols].copy()
    q["P_ref_own"] = q.P_ref
    q["P_ref"] = rng.choice(pp.P_ref.values, n_null)
    sel = pd.concat([pp, q.assign(control=True)], ignore_index=True)
    sel.to_csv(out_csv, index=False)
    return sel


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ffi-dir")
    ap.add_argument("--select", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--make-selection", metavar="SOURCES_CSV",
                    help="only write the 19-source selection (--select) from galaxy_phase2_analysis's sources.csv")
    ap.add_argument("--make-selection-all", metavar="SOURCES_CSV",
                    help="only write the all-periodic + 40 quiet-control selection (--select)")
    args = ap.parse_args()
    if args.make_selection:
        print(make_selection(Path(args.make_selection), Path(args.select)).to_string())
        return
    if args.make_selection_all:
        sel = make_selection_all(Path(args.make_selection_all), Path(args.select))
        print(f"{int((~sel.control).sum())} periodic sources + {int(sel.control.sum())} quiet controls")
        return
    ffi = Path(args.ffi_dir)
    stem = Path(args.select).stem
    idx = pd.read_csv(ffi / f"ffi_index_{stem}.csv")
    suffix = stem.replace("ffi_selection", "")   # "" for the 19-source check, "_all" for the full run
    sel = pd.read_csv(args.select).set_index("tic")
    per_lc, per_tic, powers, phases = [], [], [], []
    for tic, d in idx[idx.status == "ok"].groupby("tic", sort=False):
        P = float(sel.loc[tic, "P_ref"])
        w = detrend_window(P)
        T, Y = [], []
        for r in d.itertuples():
            lc = pd.read_csv(Path(ffi) / "lc" / Path(r.path).name)
            t, y = clean(lc.time.values, lc.flux.values, w)
            a = analyse(t, y, P, w)
            per_lc.append(dict(tic=tic, author=r.author, sector=r.sector, n=len(t),
                               P_top=a["peaks"][0] if a["peaks"] else np.nan, match=a["match"],
                               amp_at_P_ppm=a["amp_at_P_ppm"], pow_at_P=a["pow_at_P"]))
            T.append(t)
            Y.append(y)
        t, y = np.concatenate(T), np.concatenate(Y)
        a = analyse(t, y, P, w)
        ph = ((t / P) % 1.0) if P > 0 else np.zeros_like(t)
        prof = np.array([np.median(y[(ph >= k / 100) & (ph < (k + 1) / 100)]) if np.any((ph >= k / 100) & (ph < (k + 1) / 100))
                         else np.nan for k in range(100)])
        secm = [x["match"] for x in per_lc if x["tic"] == tic]
        per_tic.append(dict(tic=tic, classification=sel.loc[tic, "classification"], P_ref=P,
                            control=bool(sel.loc[tic, "control"]) if "control" in sel else False,
                            smarts_amp_ppm=float(sel.loc[tic, "amp1_ppm_all"]),
                            authors=";".join(sorted(set(d.author))), sectors=";".join(map(str, d.sector)),
                            P_top=a["peaks"][0] if a["peaks"] else np.nan,
                            peaks=";".join(f"{x:.4g}" for x in a["peaks"]), match=a["match"],
                            testable=P < TESTABLE_PMAX, detrend_window_d=w,
                            confirmed=(a["match"] != "") and P < TESTABLE_PMAX, ffi_amp_at_P_ppm=a["amp_at_P_ppm"],
                            pow_at_P=a["pow_at_P"], n_sectors_matching=int(sum(m != "" for m in secm)),
                            n_sectors=len(secm)))
        powers.append(a["power"].astype(np.float16))
        phases.append(prof.astype(np.float32))
    out = Path(args.outdir)
    res = pd.DataFrame(per_tic)
    res.to_csv(out / f"ffi_compare{suffix}.csv", index=False)
    pd.DataFrame(per_lc).to_csv(out / f"ffi_compare_sectors{suffix}.csv", index=False)
    np.savez_compressed(out / f"ffi_compare_power{suffix}.npz", tic=res.tic.values, freq=FREQ.astype(np.float32),
                        power=np.stack(powers), phase_profile=np.stack(phases))
    print(res[["tic", "classification", "P_ref", "P_top", "match", "n_sectors_matching", "n_sectors",
               "smarts_amp_ppm", "ffi_amp_at_P_ppm", "pow_at_P"]].round(3).to_string())


if __name__ == "__main__":
    main()
