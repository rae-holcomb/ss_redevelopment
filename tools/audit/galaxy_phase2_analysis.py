"""T20 phase 2 analysis: classify the periodic SMARTS noise sources and test dev_medium answers against them.

Reads the drive products of ``galaxy_tic_map.py``, ``tic_lookup.py``,
``galaxy_pass.py``, ``galaxy_stack.py`` and ``galaxy_periods.py`` and writes
small report tables to ``--outdir``:

- ``sources.csv``: one row per noise source (TIC): pool, members, corrupt
  fraction, noise level, TIC/Gaia properties (central point source and its
  star/galaxy probabilities and astrometry; neighbours within 63"), Lomb-Scargle
  peaks of the A/B/all stacks, A/B persistence, the pipeline's period on the
  stack, the common-mode test and the final classification;
- ``common_modes.npy``: the top principal components across sources;
- ``period_clusters.csv``: 1%-wide period bins holding >= 10 sources' LS peaks (informational:
  the 5-30 d red-noise continuum fills them, so they are not used to classify);
- ``devmedium_vs_source.csv``: dev_medium answers vs their noise source's own periods;
- on the drive, ``extracted/galaxy_ids/noise_source_flags.csv.gz``: every file (row) with its
  noise source's classification, reference period and amplitude.

Usage::

    $PY tools/audit/galaxy_phase2_analysis.py --outdir reports/<folder>/data
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
    determinism.pin_blas_threads(4)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluation as ev  # noqa: E402
import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

TESS_ORBIT_D = 13.7
LMC_RA, LMC_DEC = 80.89, -69.76   # LMC centre (deg); the southern CVZ overlaps its outskirts
LMC_NEAR_DEG = 4.0
CENTRAL_ARCSEC = 3.0
# "Periodic" noise source: the production pipeline returns a period for the stack, or the
# stack's LS top peak (0.1-60 d) has power >= LS_POWER_MIN and agrees between the disjoint
# halves A and B. The population's LS power: median 0.015, 90th percentile 0.067 (see report).
LS_POWER_MIN = 0.1
N_PCS = 8


def gaia_summary(gx: pd.DataFrame, tg: pd.DataFrame, var: pd.DataFrame) -> pd.DataFrame:
    """Per TIC: brightest Gaia source within 3" (central) and neighbours 3-63" (brighter ones, variables)."""
    gx = gx.merge(tg[["ID", "Tmag"]].rename(columns={"ID": "tic"}), on="tic", how="left")
    gx = gx.merge(var.rename(columns={"Class": "var_class", "ClassSc": "var_score"}), on="Source", how="left")
    cen = gx[gx.sep_arcsec < CENTRAL_ARCSEC].sort_values("Gmag").groupby("tic").head(1).set_index("tic")
    cen = cen[["Gmag", "sep_arcsec", "PSS", "PGal", "PQSO", "Plx", "RPlx", "pmRA", "pmDE", "epsi", "sepsi",
               "RUWE", "VarFlag", "var_class"]].add_prefix("cen_")
    nb = gx[gx.sep_arcsec >= CENTRAL_ARCSEC]
    brighter = nb[nb.Gmag < nb.Tmag + 1.0]   # TESS T ~ G - 0.4 for typical stars: within ~1.4 mag
    agg = pd.DataFrame(dict(
        n_nb=nb.groupby("tic").size(),
        n_nb_brighter=brighter.groupby("tic").size(),
        nb_brightest_G=nb.groupby("tic").Gmag.min(),
        n_nb_var=nb[nb.VarFlag == "VARIABLE"].groupby("tic").size(),
        nb_var_classes=nb[nb.var_class.notna()].groupby("tic").var_class.agg(lambda s: ";".join(sorted(set(s)))),
        nb_var_brightest_G=nb[nb.VarFlag == "VARIABLE"].groupby("tic").Gmag.min(),
    ))
    out = cen.join(agg, how="outer")
    for c in ("n_nb", "n_nb_brighter", "n_nb_var"):
        out[c] = out[c].fillna(0).astype(int)
    return out


def pass_summary(root: Path, gmap: pd.DataFrame) -> pd.DataFrame:
    """Per-galaxy medians of the full-corpus pass stats (corrupt fraction, p2p, slow rms)."""
    st = gp.load_pass(Path(root, sd.EXTRACT_DIR, gp.PASS_DIR))
    st = st.merge(gmap[["row", "galaxy"]], on="row")
    g = st.groupby("galaxy")
    return pd.DataFrame(dict(n_files_pass=g.size(),
                             frac_corrupt=g.max_abs_dev.apply(lambda s: float((s > gp.OUTLIER_THRESH).mean())),
                             p2p_ppm=g.p2p_ppm.median(), slow_rms_ppm_median_member=g.slow_rms_ppm.median()))


def common_modes(S: np.ndarray, n_pcs: int) -> tuple[np.ndarray, np.ndarray]:
    """Top principal components of the all-member stacks on the common grid.

    Each stack: cadences with |g - 1| > 0.1 (corrupt) or exact-linear fill set to
    0 after subtracting the median and dividing by a robust scale. Returns (PCs
    (n_pcs, n_cad), per-source unit-normalized standardized matrix X).
    """
    X = np.zeros(S.shape, np.float32)
    for i, g in enumerate(S):
        g = g.astype(np.float64)
        bad = ~np.isfinite(g) | (np.abs(g - 1) > gp.OUTLIER_THRESH)
        flat = np.zeros(len(g), bool)
        flat[1:-1] = gp.collinear(np.where(bad, 1.0, g))
        bad |= flat
        if bad.all():
            continue
        x = g - np.median(g[~bad])
        x[bad] = 0
        X[i] = x / (np.linalg.norm(x) or 1)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:n_pcs], X


def ls_after_projection(X: np.ndarray, V: np.ndarray, t: np.ndarray, P_ref: np.ndarray) -> pd.DataFrame:
    """LS power at each source's reference period before and after projecting out the common modes ``V``.

    ``cm_retained`` = power_after / power_before at ``P_ref`` (the period the
    pipeline reports for the stack, else the stack's LS top peak). A signal
    carried by the spacecraft-wide common modes loses most of its power; a
    source-specific one keeps it. Also returns the LS top peak of the
    projected stack (``P_cm``).
    """
    from astropy.timeseries import LombScargle
    import galaxy_periods as gper
    f = gper.freq_grid()
    recs = []
    for x, P in zip(X, P_ref):
        m = x != 0
        if m.sum() < 1000 or not np.isfinite(P):
            recs.append(dict(pow_ref_raw=np.nan, pow_ref_cm=np.nan, cm_retained=np.nan, P_cm=np.nan))
            continue
        y = x - (x @ V.T) @ V
        r0 = float(LombScargle(t[m], x[m]).power(1 / P))
        r1 = float(LombScargle(t[m], y[m]).power(1 / P))
        p = LombScargle(t[m], y[m]).power(f, method="fast")
        recs.append(dict(pow_ref_raw=r0, pow_ref_cm=r1, cm_retained=r1 / r0 if r0 > 0 else np.nan,
                         P_cm=float(1 / f[int(np.argmax(p))])))
    return pd.DataFrame(recs)


def period_clusters(P: pd.Series, tol: float = 0.01, min_n: int = 10) -> pd.DataFrame:
    """Periods (log-binned at ``tol``) shared by >= ``min_n`` sources."""
    lp = np.log(P.dropna().values)
    bins = np.arange(lp.min(), lp.max() + tol, tol)
    h, e = np.histogram(lp, bins)
    rows = [dict(P_center=float(np.exp(0.5 * (e[i] + e[i + 1]))), n_sources=int(h[i]))
            for i in np.flatnonzero(h >= min_n)]
    return pd.DataFrame(rows)


def matches(p_meas: float, refs, tol: float = ev.REL_TOL) -> str:
    """'match', 'half', 'double' or '' comparing a measured period with any reference period."""
    for q in refs:
        if not np.isfinite(q) or not np.isfinite(p_meas):
            continue
        if abs(p_meas - q) / q < tol:
            return "match"
        if abs(p_meas - q / 2) / (q / 2) < tol:
            return "half"
        if abs(p_meas - 2 * q) / (2 * q) < tol:
            return "double"
    return ""


def devmedium_test(src: pd.DataFrame, gmap: pd.DataFrame, per_star: Path, seed: int = 0,
                   n_null: int = 200) -> pd.DataFrame:
    """For each dev_medium star: does its pipeline answer match its noise source's own periods?

    Reference periods per source: LS P1-P3 of the all-member stack and the pipeline's
    period on the stack. A shuffled-source null (each star paired with a random
    other source of the same pool) gives the chance-match rate. ``strict_match``
    uses only the pipeline's period on the stack; ``attrs['strict_null_wrong']``
    holds ``n_null`` shuffled-source counts of strict matches among wrong answers.
    """
    d = pd.read_csv(per_star)
    d["row"] = d.star_id.str[-6:].astype(int)
    d = d.merge(gmap[["row", "galaxy"]], on="row")
    refs = src.set_index("galaxy")[["P1_all", "P2_all", "P3_all", "pipe_P"]]
    rng = np.random.default_rng(seed)
    pool = src.loc[src.pool == "ge800k", "galaxy"].values
    shuf = rng.choice(pool, len(d))
    d["source_match"] = [matches(p, refs.loc[g].values) if pd.notna(p) else ""
                         for p, g in zip(d.final_period, d.galaxy)]
    d["null_match"] = [matches(p, refs.loc[g].values) if pd.notna(p) else ""
                       for p, g in zip(d.final_period, shuf)]
    # strict: the answer equals (within REL_TOL) the pipeline's own period on the source's stack
    pipe = refs.pipe_P
    strict = lambda p, g: bool(pd.notna(p) and pd.notna(pipe[g]) and abs(p - pipe[g]) / pipe[g] < ev.REL_TOL)
    d["strict_match"] = [strict(p, g) for p, g in zip(d.final_period, d.galaxy)]
    wrong = (d.final_outcome == ev.WRONG).values
    d.attrs["strict_null_wrong"] = [int(sum(strict(p, g) for p, g in zip(d.final_period[wrong],
                                                                       rng.choice(pool, wrong.sum()))))
                                    for _ in range(n_null)]
    return d[["star_id", "row", "galaxy", "true_period", "final_period", "final_outcome",
              "source_match", "null_match", "strict_match"]]


def export_examples(src: pd.DataFrame, sdir: Path, out: Path, seed: int = 0) -> pd.DataFrame:
    """Save A/B/all stacks and LS power of a few example sources per class (for report figures)."""
    rng = np.random.default_rng(seed)
    pick = []
    for cls, d in src.groupby("classification"):
        d = d.sort_values("pow1_all", ascending=False)
        if cls == "quiet":
            d = d.iloc[rng.choice(len(d), 2, replace=False)]
        pick.append(d.head(2))
    ex = pd.concat(pick)
    idx = pd.read_csv(sdir / "stack_index.csv")
    i = idx.reset_index().set_index("galaxy").loc[ex.galaxy, "index"].values
    S = np.load(sdir / "stacks.npy", mmap_mode="r")
    P = np.load(sdir / "ls_power.npy", mmap_mode="r")
    np.savez_compressed(out / "example_stacks.npz", galaxy=ex.galaxy.values, stacks=np.asarray(S[i]),
                        ls_power=np.asarray(P[i]), ls_freq=np.load(sdir / "ls_freq.npy"))
    ex.to_csv(out / "example_sources.csv", index=False)
    return ex


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-star", default=str(REPO / "benchmarks/baselines/dev_medium_fullyear_tables/per_star.csv"))
    args = ap.parse_args()
    root = sd.require_drive(sd.data_root(args.data_root))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ex = Path(root, sd.EXTRACT_DIR)
    gmap = pd.read_csv(ex / "galaxy_ids/galaxy_map.csv.gz")
    tt = pd.read_csv(ex / "galaxy_ids/tic_table.csv")
    tg = pd.read_csv(ex / "tic_lookup/tic_targets.csv")
    gx = pd.read_csv(ex / "tic_lookup/gaia_xmatch.csv")
    var = pd.read_csv(ex / "tic_lookup/gaia_vari.csv")
    sdir = ex / "galaxy_stack"
    idx = pd.read_csv(sdir / "stack_index.csv")
    ls = pd.read_csv(sdir / "ls_peaks.csv")
    pipe = pd.read_csv(sdir / "pipeline_on_stacks.csv")

    src = tt.merge(idx[["galaxy", "pool", "n_A", "n_B", "E_max_stack"]], on="galaxy")
    src = src.merge(tg[["ID", "Tmag", "objType", "ra", "dec", "contratio", "numcont"]].rename(
        columns={"ID": "tic"}), on="tic", how="left")
    src = src.join(gaia_summary(gx, tg, var), on="tic")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    src["d_lmc_deg"] = SkyCoord(src.ra.values * u.deg, src.dec.values * u.deg).separation(
        SkyCoord(LMC_RA * u.deg, LMC_DEC * u.deg)).deg
    src["near_lmc"] = src.d_lmc_deg < LMC_NEAR_DEG
    src = src.join(pass_summary(root, gmap), on="galaxy")
    w = ls.pivot(index="galaxy", columns="stack",
                 values=["P1", "pow1", "amp1_ppm", "P2", "P3", "rms_ppm", "n_orbits_flagged"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    src = src.join(w, on="galaxy").join(ls.groupby("galaxy").persist.first(), on="galaxy")
    src = src.join(pipe.set_index("galaxy")[["final_period", "success"]].rename(
        columns={"final_period": "pipe_P", "success": "pipe_success"}), on="galaxy")

    S = np.load(sdir / "stacks.npy", mmap_mode="r")[:, 2]
    V, X = common_modes(np.asarray(S), N_PCS)
    np.save(out / "common_modes.npy", V)
    t = np.arange(gp.N_CAD) / 48.0
    src = src.reset_index(drop=True)
    src["P_ref"] = np.where(src.pipe_success.fillna(False).astype(bool), src.pipe_P, src.P1_all)
    cm = ls_after_projection(X, V, t, src.P_ref.values)
    src = pd.concat([src, cm], axis=1)
    src["cm_survives"] = src.cm_retained >= 0.5

    clus = period_clusters(src.P1_all)
    clus.to_csv(out / "period_clusters.csv", index=False)
    in_cluster = np.zeros(len(src), bool)
    for pc in clus.P_center:
        in_cluster |= (np.abs(src.P1_all - pc) / pc < 0.02).values

    ls_periodic = src.persist.fillna(False).astype(bool) & (src.pow1_all >= LS_POWER_MIN)
    periodic = ls_periodic | src.pipe_success.fillna(False).astype(bool)
    src["ls_periodic"] = ls_periodic
    # A star-like Gaia source at the target position that carries most of the target's light
    # (G within 1.5 mag of Tmag). Parallax/proper motion can't be required: LMC stars have neither.
    star_like = (src.cen_PSS > 0.9) & (src.cen_Gmag < src.Tmag + 1.5)
    var_nb = src.n_nb_var > 0
    crowded = src.near_lmc | (src.n_nb_brighter > 0)
    src["periodic"] = periodic
    src["shared_period_bin"] = in_cluster   # informational only: the 5-30 d red-noise continuum fills these
    src["star_like_center"] = star_like.fillna(False)
    src["tess_timescale"] = [any(abs(p - q) / q < 0.03 for q in (TESS_ORBIT_D, 2 * TESS_ORBIT_D, 1.0))
                             if pd.notna(p) else False for p in src.P_ref]
    cond = [~periodic,
            ~src.cm_survives.fillna(False).astype(bool),
            src.star_like_center,
            var_nb.fillna(False),
            crowded.fillna(False)]
    names = ["quiet", "systematic_common_mode", "misclassified_star", "contamination_known_variable",
             "contamination_candidate_crowded"]
    src["classification"] = np.select(cond, names, default="source_specific_unexplained")
    src.to_csv(out / "sources.csv", index=False)
    flag_cols = ["galaxy", "tic", "classification", "periodic", "P_ref", "amp1_ppm_all", "pow1_all", "near_lmc"]
    flags = gmap[["row", "sim_id", "galaxy", "pool"]].merge(src[flag_cols], on="galaxy")
    with sd.atomic_path(ex / "galaxy_ids" / "noise_source_flags.csv.gz") as tmp:
        flags.sort_values("row").to_csv(tmp, index=False, compression="gzip")

    export_examples(src, sdir, out)
    dm = devmedium_test(src, gmap, Path(args.per_star))
    null = dm.attrs["strict_null_wrong"]
    dm.to_csv(out / "devmedium_vs_source.csv", index=False)
    dm = dm.merge(src[["galaxy", "classification", "periodic"]], on="galaxy")
    summ = dict(n_sources=len(src), classification=src.classification.value_counts().to_dict(),
                n_periodic=int(periodic.sum()),
                files_in_periodic_sources=int(src.loc[periodic, "n_files"].sum()),
                period_clusters=clus.to_dict("records"),
                devmedium_strict=dict(wrong=int((dm.strict_match & (dm.final_outcome == ev.WRONG)).sum()),
                                      null_mean=float(np.mean(null)), null_max=int(np.max(null))),
                devmedium_wrong_rate=dict(periodic_source=float((dm[dm.periodic].final_outcome == ev.WRONG).mean()),
                                          quiet_source=float((dm[~dm.periodic].final_outcome == ev.WRONG).mean()),
                                          n_periodic=int(dm.periodic.sum())),
                devmedium_by_class=pd.crosstab(dm.classification, dm.final_outcome).to_dict("index"),
                devmedium={o: dict(n=int((dm.final_outcome == o).sum()),
                                   source_match=int(((dm.final_outcome == o) & (dm.source_match != "")).sum()),
                                   null_match=int(((dm.final_outcome == o) & (dm.null_match != "")).sum()))
                           for o in ("wrong", "half", "double", "diffrot", "correct")})
    (out / "phase2_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
