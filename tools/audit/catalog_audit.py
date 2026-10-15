"""Audit the SMARTS-TESS DR1 input-parameter catalog (1M rows) against the reference.

Reads the converted catalog (``catalog_to_table.py``) from the drive and the
10k batch1 FITS headers (``results_products/batch1/summary.csv``, local), and
writes *small* summary tables to ``--outdir`` (the report's ``data/`` folder)
so the report notebook can make every figure without the drive:

- ``inventory.json``: row count, IDs, duplicates, NaNs, ranges, rounding,
  distribution-check statistics (KS D vs the reference distribution), shear
  sign mix, BFLY fraction, batch1 header-vs-catalog agreement.
- ``param_hists.csv``: per-parameter histograms (catalog, batch1, reference
  expectation) in the space where the reference is uniform.
- ``spearman.csv``: rank-correlation matrix of the parameters (200k subsample).
- ``bundle_counts.csv``: rows per 1-d period bundle (= files per tarball).
- ``reach.csv``: corpus fraction reachable under period-vs-baseline limits,
  per period bin and coverage scenario.
- ``diffrot_bins.csv``, ``diffrot_ratio_hist.csv``: active-band period range
  (``evaluation.active_band_period_range``) relative to P_eq.
- ``detectability.csv``: 2D counts of PERIOD bin x amplitude-proxy bin, plus
  marginal proxy fractions.
- ``strata.csv``: per-stratum availability for T2 (period bin x proxy tercile
  x shear class), with bundles spanned.

Also writes per-row derived columns (P_lo, P_hi, amplitude proxy) to
``<root>/extracted/catalog/catalog_derived.npz`` on the drive.

Runtime ~1 min (the 1M active_band_period_range calls dominate).

Usage::

    python tools/audit/catalog_audit.py --outdir reports/2026-10-15_t1-smarts-catalog-audit/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "claude_files"))
import smarts_data as sd  # noqa: E402
from catalog_to_table import load_catalog  # noqa: E402
import evaluation  # noqa: E402

BATCH1 = REPO / "results_products" / "batch1" / "summary.csv"
PARAMS = ["PERIOD", "ACTIVITY", "CYCLE", "OVERLAP", "INCL", "MINLAT", "MAXLAT",
          "DIFFROT", "TSPOT", "BFLY"]
#: Fine period bins (days) for the audit; the scorecard's bins stop at ">30d".
AUDIT_PBIN_EDGES = (0.0, 1.0, 3.0, 10.0, 30.0, 60.0, 120.0, 180.0)
AUDIT_PBIN_LABELS = ("<1d", "1-3d", "3-10d", "10-30d", "30-60d", "60-120d", "120-180d")
#: 13-sector SMARTS baseline (header TELAPSE, days).
BASELINE_13 = 357.04151926726604
#: Coverage scenarios for the reach table: name -> baseline (days). One TESS
#: sector is ~27.4 d; consecutive sectors scale linearly. These are spans, not
#: duty cycles; gaps are ignored.
COVERAGE_BASELINES = {"13 sectors (full SMARTS)": BASELINE_13,
                      "6 consecutive": 6 * 27.4, "3 consecutive": 3 * 27.4,
                      "2 consecutive": 2 * 27.4, "1 sector": 27.4}
#: Pipeline-reach limits, as fractions of the baseline: classic comb fit
#: (>= 12 cycles), current few-peak comb limit (~B/3), hard ACF edge (B/2).
REACH_FRACS = {"P < B/12": 1 / 12, "P < B/3": 1 / 3, "P < B/2": 1 / 2}


def audit_pbin(p) -> np.ndarray:
    """Audit period-bin label for each period in ``p``."""
    idx = np.clip(np.searchsorted(AUDIT_PBIN_EDGES, p, side="right") - 1, 0,
                  len(AUDIT_PBIN_LABELS) - 1)
    return np.asarray(AUDIT_PBIN_LABELS)[idx]


def uniformized(df: pd.DataFrame) -> dict:
    """Map each parameter into the variable that the reference says is U(0, 1).

    Reference (docs/SMARTS_REFERENCE.md Table 1): PERIOD U(0.1,180);
    ACTIVITY logU(0.1,10); CYCLE logU(1,40); OVERLAP logU(0.1, CYCLE);
    INCL uniform in sin^2 i; MINLAT U(0,40); MAXLAT U(MINLAT+5, 80);
    TSPOT logU(1,10); |DIFFROT| logU(0.1,1) for nonzero shear.
    """
    nz = df.DIFFROT.values != 0
    return {
        "PERIOD": (df.PERIOD.values - 0.1) / (180 - 0.1),
        "ACTIVITY": np.log(df.ACTIVITY.values / 0.1) / np.log(100.0),
        "CYCLE": np.log(df.CYCLE.values) / np.log(40.0),
        "OVERLAP": np.log(df.OVERLAP.values / 0.1) / np.log(df.CYCLE.values / 0.1),
        "INCL": np.sin(df.INCL.values) ** 2,
        "MINLAT": df.MINLAT.values / 40.0,
        "MAXLAT": (df.MAXLAT.values - df.MINLAT.values - 5) / (75 - df.MINLAT.values),
        "TSPOT": np.log(df.TSPOT.values) / np.log(10.0),
        "|DIFFROT|": np.log(np.abs(df.DIFFROT.values[nz]) / 0.1) / np.log(10.0),
    }


def amplitude_proxy(df: pd.DataFrame) -> np.ndarray:
    """Rotational-modulation amplitude proxy: ACTIVITY * sin(INCL) * cos(mean band latitude).

    From Claytor+2022 Eq. 2-8: a spot's flux deficit scales with the activity
    level A, and the part of cos(beta) that varies with rotation phase is
    cos(lat) * sin(i). Ignores spot-size spread, cycle phase and evolution;
    meant only for ranking/stratifying, calibrated against measured LC
    amplitudes in the sample spot-check.
    """
    mid = np.radians(0.5 * (df.MINLAT.values + df.MAXLAT.values))
    return df.ACTIVITY.values * np.sin(df.INCL.values) * np.cos(mid)


def shear_class(d) -> np.ndarray:
    """'anti' (DIFFROT<0), 'none' (==0) or 'solar' (>0)."""
    d = np.asarray(d)
    return np.where(d < 0, "anti", np.where(d > 0, "solar", "none"))


def band_ranges(df: pd.DataFrame) -> tuple:
    """(P_lo, P_hi) per row via evaluation.active_band_period_range (the scoring definition)."""
    out = np.array([evaluation.active_band_period_range(p, s, lo, hi) for p, s, lo, hi in
                    zip(df.PERIOD.values, df.DIFFROT.values, df.MINLAT.values, df.MAXLAT.values)])
    return out[:, 0], out[:, 1]


def inventory(df: pd.DataFrame, b1: pd.DataFrame) -> dict:
    """Row/ID/NaN/range/rounding checks, distribution tests and batch1 cross-check."""
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    inv = {
        "n_rows": int(len(df)),
        "sim_id_min": int(df.SIM_ID.min()), "sim_id_max": int(df.SIM_ID.max()),
        "sim_id_unique": int(df.SIM_ID.nunique()),
        "sim_id_is_permutation_of_range": bool(np.array_equal(np.sort(df.SIM_ID.values),
                                                              np.arange(len(df)))),
        "sim_id_equals_row": bool(np.array_equal(df.SIM_ID.values, np.arange(len(df)))),
        "rows_sorted_by_period": bool(np.all(np.diff(df.PERIOD.values) >= 0)),
        "n_duplicate_param_rows": int(df.drop(columns="SIM_ID").duplicated().sum()),
        "n_nan": {c: int(df[c].isna().sum()) for c in df.columns},
        "n_inf": {c: int(np.isinf(df[c].values).sum()) for c in num},
        "n_period_nonpositive": int((df.PERIOD <= 0).sum()),
        "range": {c: [float(df[c].min()), float(df[c].max())] for c in num},
        "median": {c: float(df[c].median()) for c in num},
        "mean": {c: float(df[c].mean()) for c in num},
        "all_on_1e-4_grid": {c: bool(np.allclose(np.round(df[c].values * 1e4) / 1e4,
                                                 df[c].values, atol=1e-9, rtol=0)) for c in num},
        "omega_vs_2pi_over_P_max_rel_dev": float(np.max(np.abs(
            df.OMEGA.values * df.PERIOD.values / (2 * np.pi) - 1))),
        "n_exact_integer_period": int(np.sum(df.PERIOD.values == np.round(df.PERIOD.values))),
    }
    d = df.DIFFROT.values
    inv["shear_mix"] = {"solar(>0)": float(np.mean(d > 0)), "zero": float(np.mean(d == 0)),
                        "anti(<0)": float(np.mean(d < 0)),
                        "0<|shear|<0.1": float(np.mean((d != 0) & (np.abs(d) < 0.1)))}
    inv["bfly_true_frac"] = float(df.BFLY.mean())
    u = uniformized(df)
    inv["ks_vs_reference"] = {k: {"D": float(stats.kstest(v, "uniform").statistic),
                                  "n": int(len(v)),
                                  "D_crit_1pct": float(1.63 / np.sqrt(len(v)))}
                              for k, v in u.items()}
    # Alternative inclination hypothesis: isotropic (uniform in cos i).
    inv["ks_incl_uniform_cos_i"] = float(stats.kstest(np.cos(df.INCL.values), "uniform").statistic)
    inv["ks_incl_uniform_i"] = float(stats.kstest(df.INCL.values / (np.pi / 2), "uniform").statistic)
    # batch1 cross-check: file number = catalog row index.
    j = b1.join(df, on="ROW", rsuffix="_cat")
    inv["batch1"] = {"n_files": int(len(b1)),
                     "max_abs_diff_row_join": {c: float(np.max(np.abs(j[c] - j[c + "_cat"])))
                                               for c in PARAMS if c != "BFLY"},
                     "bfly_mismatch_row_join": int((j.BFLY.astype(bool) != j.BFLY_cat).sum())}
    js = b1.join(df.set_index("SIM_ID"), on="ROW", rsuffix="_cat")
    inv["batch1"]["frac_period_mismatch_if_joined_on_sim_id"] = float(
        np.mean(np.abs(js.PERIOD - js.PERIOD_cat) > 1e-3))
    return inv


def param_hists(df: pd.DataFrame, b1: pd.DataFrame, nbins: int = 40) -> pd.DataFrame:
    """Histograms of each uniformized parameter for catalog and batch1, with the flat reference."""
    uc, ub = uniformized(df), uniformized(b1)
    edges = np.linspace(0, 1, nbins + 1)
    rows = []
    for k in uc:
        hc, _ = np.histogram(uc[k], edges)
        hb, _ = np.histogram(ub[k], edges)
        for i in range(nbins):
            rows.append(dict(param=k, lo=edges[i], hi=edges[i + 1],
                             cat_frac=hc[i] / len(uc[k]), batch1_frac=hb[i] / len(ub[k]),
                             ref_frac=1 / nbins))
    return pd.DataFrame(rows)


def reach_table(p: np.ndarray, p_hi: np.ndarray) -> pd.DataFrame:
    """Fraction of stars below each reach limit, per coverage scenario and audit period bin.

    ``frac_eq`` uses P_eq; ``frac_band_all`` requires the whole active band
    (P_hi) to be below the limit too.
    """
    bins = audit_pbin(p)
    rows = []
    for cov, B in COVERAGE_BASELINES.items():
        for lim_name, f in REACH_FRACS.items():
            lim = f * B
            for b in ("all",) + AUDIT_PBIN_LABELS:
                m = np.ones_like(p, bool) if b == "all" else bins == b
                rows.append(dict(coverage=cov, baseline_d=B, limit=lim_name, limit_d=lim,
                                 pbin=b, n=int(m.sum()),
                                 frac_eq=float(np.mean(p[m] < lim)),
                                 frac_band_all=float(np.mean(p_hi[m] < lim))))
    return pd.DataFrame(rows)


def diffrot_tables(df, p_lo, p_hi, rel_tol=evaluation.REL_TOL) -> tuple:
    """Per-bin diffrot-impact fractions and histograms of P_lo/P_eq, P_hi/P_eq."""
    p = df.PERIOD.values
    lo_r, hi_r = p_lo / p, p_hi / p
    extends = (hi_r > 1 + rel_tol) | (lo_r < 1 - rel_tol)
    # Whole band outside +/-REL_TOL of P_eq: P_eq itself is never a spot period.
    disjoint = (lo_r > 1 + rel_tol) | (hi_r < 1 - rel_tol)
    # Period at the band's mid latitude (same sin^2 law as evaluation.latitude_period).
    s2mid = np.sin(np.radians(0.5 * (df.MINLAT.values + df.MAXLAT.values))) ** 2
    mid_r = 1.0 / (1.0 - df.DIFFROT.values * s2mid)
    mid_out = np.abs(mid_r - 1) > rel_tol
    sc = shear_class(df.DIFFROT.values)
    bins = audit_pbin(p)
    rows = []
    for b in ("all",) + AUDIT_PBIN_LABELS:
        for s in ("all", "solar", "none", "anti"):
            m = (np.ones_like(p, bool) if b == "all" else bins == b) & \
                (np.ones_like(p, bool) if s == "all" else sc == s)
            rows.append(dict(pbin=b, shear=s, n=int(m.sum()),
                             frac_band_extends_beyond_tol=float(extends[m].mean()),
                             frac_band_disjoint_from_peq=float(disjoint[m].mean()),
                             frac_midlat_period_outside_tol=float(mid_out[m].mean()),
                             frac_band_width_gt_2tol=float(((hi_r - lo_r) > 2 * rel_tol)[m].mean()),
                             median_hi_over_eq=float(np.median(hi_r[m])),
                             median_lo_over_eq=float(np.median(lo_r[m]))))
    edges = np.concatenate([[0.4], np.geomspace(0.5, 40, 120)])
    h_lo, _ = np.histogram(lo_r, edges)
    h_hi, _ = np.histogram(hi_r, edges)
    h_mid, _ = np.histogram(mid_r, edges)
    hist = pd.DataFrame(dict(lo=edges[:-1], hi=edges[1:], n_lo=h_lo, n_hi=h_hi, n_mid=h_mid))
    return pd.DataFrame(rows), hist, extends, disjoint


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_catalog(root)
    b1 = pd.read_csv(BATCH1).drop_duplicates("star_id")
    b1["ROW"] = b1.star_id.str[-6:].astype(int)
    b1 = b1[["ROW"] + PARAMS].reset_index(drop=True)

    inv = inventory(df, b1)
    with sd.atomic_path(out / "inventory.json") as tmp:
        tmp.write_text(json.dumps(inv, indent=1))
    param_hists(df, b1).to_csv(out / "param_hists.csv", index=False)

    # OVERLAP vs CYCLE (the one parameter that deviates from the reference).
    ce, oe = np.geomspace(1, 40, 41), np.geomspace(0.1, 40, 61)
    h2, _, _ = np.histogram2d(df.CYCLE.values, df.OVERLAP.values, [ce, oe])
    pd.DataFrame([dict(c_lo=ce[i], c_hi=ce[i + 1], o_lo=oe[k], o_hi=oe[k + 1], n=int(h2[i, k]))
                  for i in range(len(ce) - 1) for k in range(len(oe) - 1)]).to_csv(
        out / "overlap_cycle_hist2d.csv", index=False)
    o, c = df.OVERLAP.values, df.CYCLE.values
    inv["overlap"] = {
        "frac_overlap_gt_cycle": float(np.mean(o > c)),
        "max_overlap_where_cycle_lt_sqrt10": float(o[c < np.sqrt(10)].max()),
        "ks_logU_0.1_to_max_cycle_sqrt10": float(stats.kstest(
            np.log(o / 0.1) / np.log(np.maximum(c, np.sqrt(10)) / 0.1), "uniform").statistic)}
    with sd.atomic_path(out / "inventory.json") as tmp:
        tmp.write_text(json.dumps(inv, indent=1))

    sub = df.drop(columns=["SIM_ID"]).sample(200_000, random_state=0).astype(float)
    sub["SIM_ID"] = df.SIM_ID.loc[sub.index].values
    sub.corr(method="spearman").to_csv(out / "spearman.csv")

    p = df.PERIOD.values
    bundle = np.clip(np.ceil(p).astype(int), 1, 180)
    pd.Series(np.bincount(bundle, minlength=181)[1:], index=pd.Index(range(1, 181), name="bundle"),
              name="n_files").to_csv(out / "bundle_counts.csv")

    p_lo, p_hi = band_ranges(df)
    reach_table(p, p_hi).to_csv(out / "reach.csv", index=False)
    dr, drh, extends, disjoint = diffrot_tables(df, p_lo, p_hi)
    dr.to_csv(out / "diffrot_bins.csv", index=False)
    drh.to_csv(out / "diffrot_ratio_hist.csv", index=False)

    prox = amplitude_proxy(df)
    pedges = np.geomspace(1e-4, 10, 41)
    bins = audit_pbin(p)
    rows = []
    for b in AUDIT_PBIN_LABELS:
        h, _ = np.histogram(prox[bins == b], pedges)
        rows += [dict(pbin=b, lo=pedges[i], hi=pedges[i + 1], n=int(h[i])) for i in range(len(h))]
    pd.DataFrame(rows).to_csv(out / "detectability.csv", index=False)

    # T2 stratum availability: audit period bin x proxy tercile x shear class.
    terc = np.quantile(prox, [1 / 3, 2 / 3])
    ptier = np.where(prox < terc[0], "low", np.where(prox < terc[1], "mid", "high"))
    st = pd.DataFrame(dict(pbin=bins, amp_tier=ptier, shear=shear_class(df.DIFFROT.values),
                           bundle=bundle, row=np.arange(len(df))))
    g = st.groupby(["pbin", "amp_tier", "shear"])
    strata = g.agg(n=("row", "size"), bundle_min=("bundle", "min"),
                   bundle_max=("bundle", "max")).reset_index()
    strata["proxy_tercile_edges"] = f"{terc[0]:.4g},{terc[1]:.4g}"
    strata.to_csv(out / "strata.csv", index=False)

    derived = Path(root, sd.EXTRACT_DIR, "catalog", "catalog_derived.npz")
    with sd.atomic_path(derived) as tmp:
        np.savez_compressed(tmp, row=np.arange(len(df)), P_LO=p_lo, P_HI=p_hi, AMP_PROXY=prox,
                            BAND_EXTENDS=extends, BAND_DISJOINT=disjoint)
    print(json.dumps({k: inv[k] for k in ("n_rows", "sim_id_unique", "shear_mix",
                                          "bfly_true_frac", "ks_vs_reference")}, indent=1))
    print(f"wrote summaries to {out}; per-row derived columns to {derived}")


if __name__ == "__main__":
    main()
