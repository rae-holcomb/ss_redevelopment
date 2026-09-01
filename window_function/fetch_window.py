"""Download real TESS FFI light curves for one high-sector-count star and
cache the good-cadence time stamps, i.e. the star's observational window
function.

The target is a southern-CVZ star observed in every sector its camera has
covered, so it supplies both (a) a long run of *consecutive* sectors and
(b) the realistic *clumped* multi-cycle sector pattern that stars actually
have in the current extended mission.

FFI products (QLP, with TESS-SPOC filling sectors QLP skipped) are used
rather than 2-min SPOC because the SMARTS light curves this project is
built around are themselves FFI-cadence.

Output: an .npz holding, per sector, the array of good-quality cadence
times (BTJD) and the median cadence spacing.
"""

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import lightkurve as lk  # noqa: E402

DEFAULT_TIC = "TIC 167814656"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_cadences.npz")


def fetch_sector_times(tic=DEFAULT_TIC, authors=("QLP", "TESS-SPOC")):
    """Retrieve good-cadence time stamps for every sector available for `tic`.

    Searches each author in `authors` in priority order; a sector is taken
    from the first author that provides it, so later authors only fill gaps
    left by earlier ones.

    Parameters
    ----------
    tic : str
        Target identifier passed to `lightkurve.search_lightcurve`.
    authors : sequence of str
        Pipeline names, highest priority first.

    Returns
    -------
    dict
        Maps sector number (int) to a dict with keys 'time' (ndarray of
        BTJD values for good cadences), 'dt' (median cadence spacing in
        days) and 'author' (which pipeline supplied it).
    """
    out = {}
    for author in authors:
        sr = lk.search_lightcurve(tic, author=author)
        if len(sr) == 0:
            continue
        sectors = np.array([int(str(m).split()[-1]) for m in sr.table["mission"]])
        for sector in sorted(set(sectors)):
            if sector in out:
                continue
            # If a pipeline supplies several products for one sector, take
            # the first; cadence sampling is what matters, not flux column.
            idx = int(np.flatnonzero(sectors == sector)[0])
            try:
                lc = sr[idx].download()
            except Exception as exc:  # network / missing file
                print(f"  sector {sector:3d} ({author}): FAILED {type(exc).__name__}: {exc}")
                continue
            lc = lc.remove_nans("flux")
            t = np.asarray(lc.time.value, dtype=float)
            t = np.sort(t[np.isfinite(t)])
            if t.size < 2:
                print(f"  sector {sector:3d} ({author}): no good cadences")
                continue
            dt = float(np.median(np.diff(t)))
            out[sector] = {"time": t, "dt": dt, "author": author}
            print(
                f"  sector {sector:3d} ({author:9s}): N={t.size:6d}  "
                f"dt={dt * 86400:6.1f}s  span={t[-1] - t[0]:6.2f}d  "
                f"duty={t.size * dt / (t[-1] - t[0]):.3f}"
            )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tic", default=DEFAULT_TIC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Fetching FFI window function for {args.tic} ...")
    data = fetch_sector_times(args.tic)
    if not data:
        raise SystemExit("no sectors retrieved")

    sectors = np.array(sorted(data))
    payload = {"sectors": sectors, "tic": args.tic}
    for s in sectors:
        payload[f"t_{s}"] = data[s]["time"]
        payload[f"dt_{s}"] = data[s]["dt"]
        payload[f"author_{s}"] = data[s]["author"]
    np.savez_compressed(args.out, **payload)
    print(f"\n{len(sectors)} sectors -> {args.out}")
    print(f"sectors: {list(map(int, sectors))}")


if __name__ == "__main__":
    main()
