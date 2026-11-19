"""T32 step 1: free per-window vertex residuals vs harmonic n for three groups of clean dev_medium stars.

For each star in a group, re-run Stage 2 under comb variant `local` on the
cached Stage-1 inputs (tools/stage2_harness.py cache), take the selected
candidate, and for every window of its comb (all n_peaks windows built from
the seed, not just the survivors) fit the free weighted quadratic that
_fit_comb_local uses. Recorded per window:
  - resid_line  : (free vertex - (t0 + n P)) / P, about local's fitted comb line;
  - off_seed    : (free vertex - (t0_seed + n P0)) / (window_frac * P0), the
                  offset from the frozen window's centre in half-widths
                  (|off_seed| > 1 = outside the window);
  - accepted    : vertex is a maximum (a2 < 0) inside its window;
  - lsq_in_win  : whether the lsq fit of the same candidate puts that tooth's
                  vertex in its window (NaN if lsq dropped/trimmed it).
Groups: gain (v7 not success -> local success), none2wrong (v7 none -> local
wrong), shared (success under both). Writes data/vertex_residuals.csv.

Usage: python diag_vertex.py   (from the repo root; needs runs/t32_clean/cache_{fy,as} and s2/local_{fy,as}.csv)
"""
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402
determinism.pin_blas_threads()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import comb_fit as cf  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "t32_clean"
S = {"correct": "succ", "diffrot": "succ", "half": "alias", "double": "alias", "wrong": "wrong", "none": "none"}
BASE = {"fy": "fullyear", "as": "assigned"}


def free_vertex(acf_lags, acf, w, weights):
    """Free weighted quadratic vertex of window w (as in _fit_comb_local) -> (vertex or nan, accepted, a2)."""
    lag, y = acf_lags[w.mask], acf[w.mask]
    wt = np.ones(len(lag)) if weights is None else weights[w.mask]
    mid, hw = 0.5 * (w.lag_hi + w.lag_lo), max(0.5 * (w.lag_hi - w.lag_lo), 1e-12)
    z = (lag - mid) / hw
    X = np.column_stack([np.ones_like(z), z, z * z]) * wt[:, None]
    a0, a1, a2 = np.linalg.lstsq(X, y * wt, rcond=None)[0]
    if a2 >= 0:
        return np.nan, False, a2
    zv = -a1 / (2 * a2)
    return mid + zv * hw, bool(abs(zv) <= 1), a2


def groups(mode):
    """star_id -> group label for one mode."""
    b = pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_clean_{BASE[mode]}_v7_tables/per_star.csv")
    h = pd.read_csv(RUNS / f"s2/local_{mode}.csv")
    m = b[["star_id", "final_outcome"]].merge(h[["star_id", "final_outcome"]], on="star_id", suffixes=("_v7", "_loc"))
    a, c = m.final_outcome_v7.map(S), m.final_outcome_loc.map(S)
    out = {}
    out.update({s: "gain" for s in m.star_id[(a != "succ") & (c == "succ")]})
    out.update({s: "none2wrong" for s in m.star_id[(a == "none") & (c == "wrong")]})
    out.update({s: "shared" for s in m.star_id[(a == "succ") & (c == "succ")]})
    return out


def star_rows(path, group, mode):
    """Per-window residual rows for one cached star."""
    with open(path, "rb") as fh:
        rec = pickle.load(fh)
    lags, acf, counts = rec["lags"], rec["acf"], rec["counts"]
    res = cf.fit_rotation_period(lags, acf, rec["pooled"], acf_counts=counts, comb_variant="local")
    if not res.success:
        return []
    sel = res.candidates[0]
    weights = np.sqrt(np.clip(counts, 0, None) / max(counts[0], 1.0))
    P0, t0s, P, t0 = sel.period, sel.t0, sel.fit.P, sel.fit.t0
    wins = cf._build_windows(lags, P0, t0s, 8, window_frac=0.25, max_points_per_window=100)
    # lsq fit of the same candidate, for comparison
    try:
        lfit = cf._fit_single_candidate(lags, acf, P0, t0s, n_peaks=8, weights=weights,
                                        min_peaks_required=sel.n_peaks_required if sel.few_peaks else 2,
                                        min_keep_after_trim=sel.n_peaks_required, max_points_per_window=100,
                                        comb_spec=cf.COMB_VARIANTS["lsq"])
        lsq_in = {n: p["vertex_in_window"] for n, p in lfit.per_peak.items()}
    except Exception:  # noqa: BLE001
        lsq_in = {}
    used = set(sel.fit.per_peak)
    rows = []
    for w in wins:
        v, acc, a2 = free_vertex(lags, acf, w, weights)
        rows.append(dict(star_id=rec["star_id"], mode=mode, group=group, n=w.n, P_true=rec["true_period"],
                         P0=P0, P_fit=P, resid_line=(v - (t0 + w.n * P)) / P,
                         off_seed=(v - (t0s + w.n * P0)) / (0.25 * P0), accepted=acc, in_local_fit=w.n in used,
                         lsq_in_win=float(lsq_in[w.n]) if w.n in lsq_in else np.nan))
    return rows


def main():
    """Build data/vertex_residuals.csv for both modes."""
    warnings.simplefilter("ignore")
    rows = []
    for mode in ("fy", "as"):
        for sid, g in groups(mode).items():
            rows.extend(star_rows(RUNS / f"cache_{mode}/{sid}.pkl", g, mode))
    df = pd.DataFrame(rows)
    (HERE / "data").mkdir(exist_ok=True)
    df.to_csv(HERE / "data/vertex_residuals.csv", index=False)
    print(df.groupby(["mode", "group"]).star_id.nunique())


if __name__ == "__main__":
    main()
