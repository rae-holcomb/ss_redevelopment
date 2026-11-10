"""Quiet-class noise periods: are their P_ref real signals or red-noise peaks, and why do wrong answers land on them?

Inputs: the regenerated T20 ``sources.csv`` (with time-split S/N columns, from
``galaxy_phase2_analysis.py``), a dev_medium full-year scorecard ``per_star.csv`` with the v5
noise columns, the drive's stacks / catalog, Gaia DR3 neighbour periods (``quiet_gaia_fetch.py``)
and the T20 FFI comparison table. Writes small tables to ``--outdir``:

- ``null_test.json``: observed vs shuffled-source matches of quiet-class wrong answers to P_ref;
- ``quiet_sources.csv``: every quiet / periodic_low_amplitude source with its diagnostics
  (time-split S/N at P_ref, pipeline on the stack, common-mode retention, near LMC, Gaia neighbour
  period match, FFI check, injected-period match of the stack's members, matched-star flag);
- ``matched_stars.csv``: the dev_medium quiet-class wrong answers within REL_TOL of P_ref;
- ``wrong_stars.csv``: all quiet-class wrong answers, with injected-amplitude and noise context;
- ``common_mode_peaks.csv``: the LS top peaks of T20's common modes;
- ``matched_power.npz``: per matched source, the LS power of each time half (for figures).

Usage::

    $PY tools/audit/quiet_noise_periods.py --sources <sources.csv> --per-star <per_star.csv> --outdir <dir>
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
import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

import evaluation as ev  # noqa: E402
import galaxy_pass as gp  # noqa: E402
import smarts_data as sd  # noqa: E402

QUIET = ("quiet", "periodic_low_amplitude")
GAIA_CONV = {"vari_eclipsing_binary": ("frequency", True), "vari_cepheid": ("pf", False),
             "vari_rrlyrae": ("pf", False), "vari_short_timescale": ("frequency", True),
             "vari_ms_oscillator": ("frequency1", True), "vari_rotation_modulation": ("best_rotation_period", False),
             "vari_long_period_variable": ("frequency", True)}
TESS_PERIODS = {"orbit 13.7 d": 13.7, "2 orbits 27.4 d": 27.4, "1 d": 1.0}


def within(p, q, tol=ev.REL_TOL) -> bool:
    """True when ``p`` is within fractional ``tol`` of ``q`` (both finite)."""
    return bool(np.isfinite(p) and np.isfinite(q) and abs(p - q) / q < tol)


def null_test(w: pd.DataFrame, pref_pool: np.ndarray, n: int = 5000, seed: int = 0) -> dict:
    """Shuffled-source null: each wrong answer paired with a random quiet source's P_ref.

    ``w`` has ``final_period`` and ``noise_P_ref``. Returns the observed count of matches
    (within REL_TOL), the null mean/95th/99th/max over ``n`` draws, and the one-sided p-value.
    """
    rng = np.random.default_rng(seed)
    obs = int(sum(within(p, q) for p, q in zip(w.final_period, w.noise_P_ref)))
    P = w.final_period.values[:, None]
    Q = rng.choice(pref_pool, (len(w), n))
    null = (np.abs(P - Q) / Q < ev.REL_TOL).sum(axis=0)
    return dict(n_wrong=len(w), observed=obs, null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)), null_p99=float(np.percentile(null, 99)),
                null_max=int(null.max()), p_value=float((null >= obs).mean()), null_counts=null.tolist())


def gaia_neighbour_match(src: pd.DataFrame, gaia_dirs: list[Path], gx: pd.DataFrame) -> pd.DataFrame:
    """Per TIC: number of variable neighbours with a Gaia DR3 period, and whether one matches P_ref.

    Match (as in the review's c05c): neighbour period within 3% of P_ref, P_ref/2 or 2 P_ref.
    Tables from several fetch directories are pooled (deduplicated on source and table).
    """
    rows = []
    for t, (col, isf) in GAIA_CONV.items():
        for d in gaia_dirs:
            f = Path(d, f"esa_{t}.csv")
            if not f.exists():
                continue
            x = pd.read_csv(f)
            x.columns = [c.lower() for c in x.columns]
            x = x.dropna(subset=[col])
            rows.append(pd.DataFrame(dict(Source=x.source_id, P_nb=(1 / x[col]) if isf else x[col], table=t)))
    nb = pd.concat(rows).drop_duplicates(["Source", "table"])
    nb = nb.merge(gx[["Source", "tic", "sep_arcsec", "Gmag"]], on="Source")
    nb = nb[nb.sep_arcsec >= 3]
    nb["P_ref"] = nb.tic.map(src.set_index("tic").P_ref)
    r = nb.P_nb / nb.P_ref
    nb["match"] = (np.abs(r - 1) < .03) | (np.abs(r - .5) < .015) | (np.abs(r - 2) < .06)
    g = nb.groupby("tic")
    out = pd.DataFrame(dict(n_nb_gaia_period=g.size(), gaia_nb_match=g.match.any(),
                            gaia_nb_match_table=nb[nb.match].groupby("tic").table.agg(lambda s: ";".join(sorted(set(s))))))
    return out, nb


def member_injection_match(src: pd.DataFrame, sel: pd.DataFrame, cat_period: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Does any stack member's injected PERIOD fall within 2% of the source's P_ref? (stacking-artefact test)

    The stack is a per-cadence median over the lowest-amplitude members; a residue of one member's
    injected signal would put P_ref on that member's PERIOD. Null: P_ref shuffled among sources.
    Returns per galaxy ``n_members``, ``frac_member_P_match`` and the shuffled expectation.
    """
    rng = np.random.default_rng(seed)
    m = sel[sel.role.isin(["A", "B"])].copy()
    m["P_inj"] = cat_period.reindex(m.sim_id.values).values
    gP = src.set_index("galaxy").P_ref
    shuf = pd.Series(rng.permutation(gP.values), index=gP.index)
    m["hit"] = np.abs(m.P_inj / m.galaxy.map(gP) - 1) < 0.02
    m["hit_null"] = np.abs(m.P_inj / m.galaxy.map(shuf) - 1) < 0.02
    g = m.groupby("galaxy")
    return pd.DataFrame(dict(n_members=g.size(), frac_member_P_match=g.hit.mean(), frac_member_P_match_null=g.hit_null.mean()))


def half_power(g: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """LS power on frequency grid ``f`` of the two time halves of one cleaned stack (2, len(f))."""
    g = np.asarray(g, np.float64)
    bad = ~np.isfinite(g) | (np.abs(g - 1) > gp.OUTLIER_THRESH)
    flat = np.zeros(len(g), bool)
    flat[1:-1] = gp.collinear(np.where(bad, 1.0, g))
    ok = ~(bad | flat)
    out = []
    for m in (t < 182.5, t >= 182.5):
        mm = m & ok
        out.append(LombScargle(t[mm], g[mm] - np.median(g[mm])).power(f, method="fast"))
    return np.array(out)


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--per-star", required=True)
    ap.add_argument("--gaia-dirs", nargs="+", required=True)
    ap.add_argument("--ffi", default=str(REPO / "reports/2026-10-21_t20-noise-sources-phase2/data/ffi_compare_all.csv"))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    root = sd.require_drive(sd.data_root(a.data_root))
    ex = Path(root, sd.EXTRACT_DIR)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(a.sources)
    sdir = ex / "galaxy_stack"
    idx = pd.read_csv(sdir / "stack_index.csv").reset_index().set_index("galaxy")["index"]
    pipe = pd.read_csv(sdir / "pipeline_on_stacks.csv").set_index("galaxy")
    sel = pd.read_csv(sdir / "selection.csv")
    cat = np.load(ex / "catalog/smarts_catalog.npz")
    cat_period = pd.Series(cat["PERIOD"], index=cat["SIM_ID"])
    gx = pd.read_csv(ex / "tic_lookup/gaia_xmatch.csv")

    # --- 1. dev_medium quiet-class wrong answers and the shuffled-source null
    d = pd.read_csv(a.per_star)
    d = d.merge(src[["tic", "classification", "P_ref", "amp1_ppm_all", "rms_ppm_all"]].rename(
        columns={"tic": "noise_tic", "classification": "class_new"}), on="noise_tic", how="left")
    q = src[src.classification == "quiet"]
    res = {}
    for lab, cls in (("quiet_v5_def", QUIET), ("quiet_after_reclass", ("quiet",))):
        w = d[d.class_new.isin(cls) & (d.final_outcome == ev.WRONG)]
        pool = src[src.classification.isin(cls) & (src.pool == "ge800k")].P_ref.dropna().values
        res[lab] = null_test(w, pool)
        # binned null: random P_ref from quiet sources whose P_ref is in the same factor-2 bin as the own P_ref
        w5 = w[w.noise_P_ref >= 5]
        res[lab + "_Pref_ge5"] = null_test(w5, pool[pool >= 5])
    w = d[d.class_new.isin(QUIET) & (d.final_outcome == ev.WRONG)].copy()
    w["noise_match"] = [within(p, q_) for p, q_ in zip(w.final_period, w.noise_P_ref)]
    w["P_over_Pref"] = w.final_period / w.noise_P_ref
    w.to_csv(out / "wrong_stars.csv", index=False)
    mstars = w[w.noise_match]
    mstars.to_csv(out / "matched_stars.csv", index=False)
    # What makes the remaining quiet wrongs? also: answers matching the source's 2nd/3rd LS peak
    for k in ("P2_all", "P3_all"):
        w[k] = w.noise_tic.map(src.set_index("tic")[k])
    res["wrong_match_P1_P2_P3"] = int(sum(any(within(p, w[k].iloc[i]) for k in ("noise_P_ref", "P2_all", "P3_all"))
                                          for i, p in enumerate(w.final_period)))

    # --- 2. per-source diagnostics for all quiet(-ish) sources
    s = src[src.classification.isin(QUIET)].copy()
    s["pipe_stack_P"] = s.galaxy.map(pipe.final_period)
    s["pipe_stack_top1_ls"] = s.galaxy.map(pipe.top1_lombscargle)
    s["pipe_stack_P_matches_Pref"] = [within(p, q_) for p, q_ in zip(s.pipe_stack_P, s.P_ref)]
    s["pipe_stack_ls_matches_Pref"] = [within(p, q_) for p, q_ in zip(s.pipe_stack_top1_ls, s.P_ref)]
    gm, nb = gaia_neighbour_match(src, [Path(x) for x in a.gaia_dirs], gx)
    nb.to_csv(out / "gaia_neighbour_periods.csv", index=False)
    s = s.join(gm, on="tic")
    s["n_nb_gaia_period"] = s.n_nb_gaia_period.fillna(0).astype(int)
    s["gaia_nb_match"] = s.gaia_nb_match.fillna(False).astype(bool)
    ffi = pd.read_csv(a.ffi).drop_duplicates("tic").set_index("tic")
    s["ffi_tested"] = s.tic.isin(ffi.index[ffi.testable.fillna(False).astype(bool)])
    s["ffi_confirmed"] = s.tic.map(ffi.confirmed).fillna(False).astype(bool)
    s["ffi_P_top"] = s.tic.map(ffi.P_top)
    s = s.join(member_injection_match(s, sel, cat_period), on="galaxy")
    s["matched_star_source"] = s.tic.isin(mstars.noise_tic)
    s["n_dev_wrong_matched"] = s.tic.map(mstars.noise_tic.value_counts()).fillna(0).astype(int)
    s["near_tess"] = [any(within(p, v, 0.03) for v in TESS_PERIODS.values()) for p in s.P_ref]
    s.to_csv(out / "quiet_sources.csv", index=False)

    # --- 3. common modes: their own LS top peaks (long-period systematics)
    V = np.load(Path(a.sources).parent / "common_modes.npy")
    t = np.arange(gp.N_CAD) / 48.0
    f = np.linspace(1 / 60, 10, 60000)
    cmp = []
    for k, v in enumerate(V):
        m = v != 0
        p = LombScargle(t[m], v[m]).power(f, method="fast")
        o = np.argsort(p)[::-1]
        tops = []
        for j in o:
            if all(abs(f[j] - f[i]) > 0.005 for i in tops):
                tops.append(j)
            if len(tops) == 3:
                break
        cmp.append(dict(pc=k, P1=1 / f[tops[0]], P2=1 / f[tops[1]], P3=1 / f[tops[2]]))
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(out / "common_mode_peaks.csv", index=False)

    # --- 4. per matched source: half-year periodograms
    S = np.load(sdir / "stacks.npy", mmap_mode="r")
    mg = s[s.matched_star_source].sort_values("P_ref")
    fgrid = np.linspace(1 / 60, 10, 30000)
    hp = np.array([half_power(S[idx[g], 2], t, fgrid) for g in mg.galaxy])
    full = np.array([LombScargle(t, np.nan_to_num(np.asarray(S[idx[g], 2], float), nan=1.0)).power(fgrid, method="fast")
                     for g in mg.galaxy])
    np.savez_compressed(out / "matched_power.npz", tic=mg.tic.values, P_ref=mg.P_ref.values, freq=fgrid,
                        half_power=hp.astype(np.float32), full_power=full.astype(np.float32))

    # --- summary
    sq = s[s.classification == "quiet"]
    summ = dict(null=res, n_matched_stars=len(mstars), n_matched_sources=int(s.matched_star_source.sum()),
                quiet_n=len(sq),
                quiet_frac_Pref_ge5=float((sq.P_ref >= 5).mean()),
                quiet_near_tess=int(sq.near_tess.sum()),
                matched_sources=mg[["tic", "classification", "P_ref", "pow1_all", "amp1_ppm_all", "snr_h1", "snr_h2",
                                    "tsplit_min_snr", "top_h1", "top_h2", "pipe_stack_P", "pipe_stack_top1_ls",
                                    "cm_retained", "near_lmc", "n_nb_gaia_period", "gaia_nb_match", "ffi_tested",
                                    "ffi_confirmed", "frac_member_P_match", "near_tess"]].to_dict("records"),
                pop_gaia=dict(with_period=int((sq.n_nb_gaia_period > 0).sum()), matched=int(sq.gaia_nb_match.sum())),
                pop_member_inj=dict(mean=float(sq.frac_member_P_match.mean()), null=float(sq.frac_member_P_match_null.mean())),
                pop_cm_retained_median=dict(lt5=float(sq[sq.P_ref < 5].cm_retained.median()),
                                            ge5=float(sq[sq.P_ref >= 5].cm_retained.median())),
                common_mode_peaks=cmp.round(3).to_dict("records"))
    np.save(out / "null_counts.npy", np.array(res["quiet_after_reclass"]["null_counts"]))
    for r_ in summ["null"].values():
        if isinstance(r_, dict):
            r_.pop("null_counts", None)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print(json.dumps(summ, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    main()
