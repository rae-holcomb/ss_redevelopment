"""Figure helpers for the WG6 scoping report. The notebook calls these on the saved CSVs in ``data/``.

Only the example-light-curve and coverage figures read FITS files (one star per class, from the drive);
every other figure uses the committed CSVs, so it regenerates with the drive unplugged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

# set folder -> short class label (ordering used in every figure)
CLASS_OF = {
    "Rotators_lightcurves": "ROT", "flare_lightcurves-003": "FLARE", "ECL_lightcurves1": "ECL",
    "ECL_lightcurves2-002": "ECL", "ECL_lightcurves3": "ECL", "ELL_lightcurves": "ELL",
    "HB_lightcurves": "HB", "RRLYR_lightcurves": "RRLYR", "DSCT_lightcurves": "DSCT",
    "CEP_lightcurves": "CEP", "T2CEP_lightcurves": "T2CEP", "LPV_lightcurves": "LPV",
    "VVV_lightcurves": "VVV-LPV", "BLAP_lightcurves": "BLAP", "DwarfNovae_lightcurves": "DN",
    "Transits_lightcurves": "TR",
}
ORDER = ["ROT", "FLARE", "ECL", "ELL", "HB", "RRLYR", "DSCT", "CEP", "T2CEP", "LPV", "VVV-LPV", "BLAP", "DN", "TR"]
SMARTS_PRANGE = (0.1, 180.0)


def load_inventory() -> pd.DataFrame:
    """All inventory CSVs concatenated, de-duplicated on relpath, with a ``cls`` column."""
    inv = pd.concat([pd.read_csv(p) for p in sorted((DATA / "inventory").glob("*.csv*"))], ignore_index=True)
    inv = inv.drop_duplicates("relpath", keep="last")
    inv["cls"] = inv["set"].map(CLASS_OF)
    # the root RRLYR_lightcurves/ is a duplicate of extracted/RRLYR_lightcurves/: keep extracted/ copies only
    inv["NAME"] = inv["NAME"].astype(str)
    inv = (inv.assign(_ext=inv.relpath.str.startswith("extracted/"))
              .sort_values("_ext").drop_duplicates(["cls", "NAME"], keep="last").drop(columns="_ext"))
    return inv


def load_profile() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-band rows, per-star LS rows), de-duplicated, with ``cls`` and the joined ``P_true`` column."""
    b = pd.read_csv(DATA / "profile_bands.csv").drop_duplicates(["relpath", "band"], keep="last")
    s = pd.read_csv(DATA / "profile_ls.csv").drop_duplicates("relpath", keep="last")
    for d in (b, s):
        d["cls"] = d["set"].map(CLASS_OF)
    t = DATA / "truth_joined.csv.gz"
    if t.exists():
        tr = pd.read_csv(t)[["NAME", "P_true", "truth_src"]].drop_duplicates("NAME")
        s = s.merge(tr, on="NAME", how="left")
    else:
        s["P_true"], s["truth_src"] = np.nan, ""
    return b, s


def fig_counts(inv: pd.DataFrame, path=FIG / "counts_per_class.png"):
    """Bar chart of the number of stars (files) per class, log y."""
    c = inv.groupby("cls").size().reindex(ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(c.index, c.values, color="#4c72b0")
    for i, v in enumerate(c.values):
        ax.text(i, v * 1.1, f"{int(v):,}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("stars (files)"); ax.set_ylim(1, c.max() * 4)
    ax.set_title(f"WG6 download: {int(c.sum()):,} light curves in {len(c)} classes")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); return fig


def fig_periods(s: pd.DataFrame, truth: pd.DataFrame, path=FIG / "period_hist_per_class.png"):
    """Histogram of catalog truth periods per class (all stars; filled) and the top LS peak of the profiled sample
    (hatched, for classes without a catalog) vs the SMARTS period range."""
    cls = [c for c in ORDER if c in set(s.cls)]
    fig, axes = plt.subplots(len(cls), 1, figsize=(8, 1.0 * len(cls) + 1), sharex=True)
    bins = np.logspace(-2.3, 3.3, 57)
    for ax, c in zip(np.atleast_1d(axes), cls):
        d = s[s.cls == c]
        pt = truth.loc[truth.cls == c, "P_true"].dropna()
        ax.hist(d.P1.dropna(), bins=bins, histtype="step", color="0.4", hatch="///", label="top LS peak (sample)")
        if len(pt):
            ax.hist(pt, bins=bins, color="#dd8452", alpha=0.8, label="catalog period (all stars)")
        ax.axvspan(*SMARTS_PRANGE, color="#55a868", alpha=0.12)
        ax.axvline(35, color="k", ls=":", lw=0.8)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9); ax.set_yticks([])
    ax.set_xscale("log"); ax.set_xlabel("period [d]")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left", ncol=2)
    np.atleast_1d(axes)[0].set_title("Periods per class. Green: SMARTS range 0.1–180 d; dotted: one-season R/2 = 35 d")
    fig.tight_layout(); fig.savefig(path, dpi=130); return fig


def fig_noise(b: pd.DataFrame, path=FIG / "noise_vs_mag.png"):
    """Point-to-point scatter vs median magnitude, per class, for the primary band (OGLE I; TESS for flares)."""
    smarts = np.load(DATA / "smarts_p2p_mmag.npy")
    d = b[(b.band == "LC_I") | (b.cls == "FLARE")].copy()
    if (d.cls == "FLARE").any():  # one TESS sector per flare star: take the first extension
        d = d[(d.cls != "FLARE") | (d.band == "0")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cm = plt.get_cmap("tab20")
    for i, c in enumerate([c for c in ORDER if c in set(d.cls)]):
        x = d[d.cls == c]
        ax.scatter(x.med_y, x.p2p * 1000, s=10, color=cm(i), label=c)
    lo, hi = np.percentile(smarts, [5, 95])
    ax.axhspan(lo, hi, color="#55a868", alpha=0.15, label="SMARTS-TESS 30-min (5–95%)")
    ax.set_yscale("log"); ax.set_xlabel("median magnitude (OGLE I; TESS T for flares)")
    ax.set_ylabel("point-to-point scatter [mmag]")
    ax.set_title("Noise vs brightness: per-point scatter (includes intrinsic variability for fast variables)")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout(); fig.savefig(path, dpi=130); return fig


def fig_period_check(s: pd.DataFrame, path=FIG / "stored_vs_measured.png"):
    """Catalog truth period vs our top LS peak, with 1:1, 1:2 and 2:1 lines."""
    d = s.dropna(subset=["P_true", "P1"])
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    cm = plt.get_cmap("tab20")
    for i, c in enumerate([c for c in ORDER if c in set(d.cls)]):
        x = d[d.cls == c]
        ax.scatter(x.P_true, x.P1, s=14, color=cm(i), label=f"{c} (n={len(x)})")
    lim = [0.01, 3000]
    for k, ls in ((1, "-"), (0.5, "--"), (2, "--")):
        ax.plot(lim, np.multiply(lim, k), color="k", ls=ls, lw=0.7)
    ax.text(1000, 450, "P/2", fontsize=8); ax.text(400, 1000, "2P", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("catalog period [d]"); ax.set_ylabel("top Lomb–Scargle peak (our quick check) [d]")
    ax.legend(fontsize=7); ax.set_title("Stored vs measured period")
    fig.tight_layout(); fig.savefig(path, dpi=130); return fig


def _ls_spectrum(cls: str, name: str):
    """Load the saved (max-pooled) LS spectrum for one star, or None."""
    p = DATA / "ls_spectra" / f"{[k for k, v in CLASS_OF.items() if v == cls][0]}__{name}.npz"
    cands = [p] + sorted((DATA / "ls_spectra").glob(f"*__{name}.npz"))
    for c in cands:
        if c.exists():
            z = np.load(c)
            return z["freq"], z["power"]
    return None


def fig_examples(classes=None, path=FIG / "example_lcs_periodograms.png"):
    """One star per class: full light curve, a 70-d zoom (one Roman-season-length window), and the LS periodogram
    with the catalog period (solid) and its P/2, 2P (dashed) marked."""
    classes = classes or [c for c in ORDER if (DATA / "examples" / f"{c}.npz").exists()]
    fig, axes = plt.subplots(len(classes), 3, figsize=(13, 1.7 * len(classes)),
                             gridspec_kw={"width_ratios": [2.2, 1.4, 1.6]})
    for row, c in zip(np.atleast_2d(axes), classes):
        z = np.load(DATA / "examples" / f"{c}.npz", allow_pickle=True)
        t, y, P = z["t"], z["y"], float(z["P_true"])
        name = str(z["name"])
        row[0].plot(t - t.min(), y, ",", color="k", alpha=0.5)
        row[0].invert_yaxis(); row[0].set_ylabel(c, rotation=0, ha="right", fontsize=10)
        row[0].text(0.01, 0.9, f"{name}  ({z['band']}, n={len(t):,})", transform=row[0].transAxes, fontsize=7)
        # densest 70-d window
        ts = np.sort(t); j = np.searchsorted(ts, ts + 70); i = int(np.argmax(j - np.arange(len(ts))))
        m = (t >= ts[i]) & (t < ts[i] + 70)
        row[1].plot(t[m] - ts[i], y[m], ".", ms=1.5, color="#4c72b0")
        row[1].invert_yaxis(); row[1].set_xlim(0, 70)
        row[1].text(0.01, 0.9, f"densest 70 d: {m.sum()} pts", transform=row[1].transAxes, fontsize=7)
        sp = _ls_spectrum(c, name)
        if sp is not None:
            f, pw = sp
            row[2].plot(1 / f, pw, color="0.3", lw=0.6); row[2].set_xscale("log")
        if np.isfinite(P):
            for k, ls in ((1, "-"), (0.5, "--"), (2, "--")):
                row[2].axvline(P * k, color="#c44e52", ls=ls, lw=0.9)
            row[2].text(0.02, 0.85, f"catalog P = {P:.4g} d", transform=row[2].transAxes, fontsize=7, color="#c44e52")
        else:
            row[2].text(0.02, 0.85, "no catalog period", transform=row[2].transAxes, fontsize=7)
        row[2].axvline(1.0, color="#8172b2", lw=0.6, ls=":")
        row[2].set_yticks([])
    np.atleast_2d(axes)[0, 0].set_title("full archival light curve (mag; days from first point)", fontsize=9)
    np.atleast_2d(axes)[0, 1].set_title("densest 70-d window (one GBTDS season length)", fontsize=9)
    np.atleast_2d(axes)[0, 2].set_title("Lomb–Scargle (red: catalog P, P/2, 2P; purple: 1 d)", fontsize=9)
    for ax in np.atleast_2d(axes)[-1]:
        ax.set_xlabel("days")
    np.atleast_2d(axes)[-1, 2].set_xlabel("period [d]")
    fig.tight_layout(); fig.savefig(path, dpi=110); return fig


def fig_coverage(cls="ROT", path=FIG / "time_coverage_one_star.png"):
    """Time coverage per band for one star, with the GBTDS-like 70-d season length and 12.1-min cadence for scale."""
    z = np.load(DATA / "examples" / f"{cls}.npz", allow_pickle=True)
    bands = [str(b) for b in z["bands"]]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), gridspec_kw={"height_ratios": [1.3, 1]})
    for i, b in enumerate(bands):
        t = z[f"t_{b}"]
        ax.plot(t - 2450000, np.full(len(t), i), "|", ms=8, color=plt.get_cmap("tab10")(i))
    ax.set_yticks(range(len(bands))); ax.set_yticklabels(bands)
    ax.set_xlabel("HJD − 2450000"); ax.set_title(f"{z['name']}: archival time coverage per band")
    t = np.sort(z[f"t_{bands[0]}"])
    ts = t; j = np.searchsorted(ts, ts + 70); i0 = int(np.argmax(j - np.arange(len(ts))))
    a = ts[i0]
    m = (t >= a - 15) & (t < a + 85)
    ax2.plot(t[m] - a, np.zeros(m.sum()), "|", ms=14, color="k", label=f"OGLE {bands[0]} ({((t >= a) & (t < a + 70)).sum()} pts in 70 d)")
    rg = np.arange(0, 70, 12.1 / 1440)
    ax2.plot(rg, np.full(len(rg), 1), "|", ms=14, color="#55a868", alpha=0.3,
             label=f"GBTDS-like F146, 12.1 min ({len(rg):,} pts)")
    ax2.axvspan(0, 70, color="#55a868", alpha=0.07)
    ax2.set_ylim(-0.7, 1.7); ax2.set_yticks([]); ax2.set_xlabel("days from start of densest 70-d window")
    ax2.legend(fontsize=8, loc="upper right"); ax2.set_title("one season: archival sampling vs a Roman-like season", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); return fig


# ---------------------------------------------------------------- resampled (Roman-cadence) set, added 2026-11-12
RD = DATA / "resampled"
VAR_COLORS = {"simple": "#4c72b0", "simple_shifted": "#8172b2", "improved": "#dd8452",
              "improved_folder_nonfinal": "#55a868"}
VAR_LABEL = {"simple": "simple (original brightness)", "simple_shifted": "simple, shifted +8 mag",
             "improved": "improved (_final)", "improved_folder_nonfinal": "FL non-final (simple-style, in improved folder)"}


def fig_r_same_star(path=FIG / "r_same_star_variants.png"):
    """One star: raw OGLE I (whole archive, phase-folded) vs F146 season 1 in simple, shifted and improved."""
    z = np.load(RD / "examples/same_star.npz", allow_pickle=True)
    P = float(z["P_true"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    ax = axes[0, 0]
    ph = (z["raw_t"] / P) % 1
    ax.plot(ph, z["raw_y"], ".", ms=1.5, color="k", alpha=0.4); ax.invert_yaxis()
    ax.set_title(f"{z['star_id']} raw OGLE I, folded at P = {P:.4f} d ({len(ph):,} pts, 2010–2024)", fontsize=9)
    ax.set_xlabel("phase"); ax.set_ylabel("I [mag]")
    for ax, v in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["simple", "simple_shifted", "improved"]):
        t, y, e = z[f"{v}_t"], z[f"{v}_y"], z[f"{v}_e"]
        ax.plot(t - t[0], y, ".", ms=1, color=VAR_COLORS[v]); ax.invert_yaxis()
        ax.set_title(f"F146 season 1, {VAR_LABEL[v]}: median {np.nanmedian(y):.2f} mag, "
                     f"σ_quoted {np.nanmedian(e)*1e3:.1f} mmag", fontsize=9)
        ax.set_xlabel("days from season start"); ax.set_ylabel("F146 [mag]")
        ins = ax.inset_axes([0.62, 0.62, 0.36, 0.34])
        m = (t - t[0]) < 2
        ins.plot(t[m] - t[0], y[m], ".", ms=1.5, color=VAR_COLORS[v]); ins.invert_yaxis()
        ins.set_title("first 2 d", fontsize=7); ins.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=120); return fig


def fig_r_noise(path=FIG / "r_noise_vs_mag.png"):
    """Quoted F146 error (and measured p2p) vs median magnitude per noise variant; FL objects drawn as tracks across
    their magnitude copies; SMARTS-TESS 30-min p2p band for scale."""
    p = pd.read_csv(RD / "profile.csv")
    f = p[p.band == "F146"].dropna(subset=["med_mag"])
    smarts = np.load(DATA / "smarts_p2p_mmag.npy")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for v, c in VAR_COLORS.items():
        x = f[f.variant == v]
        ogle, fl = x[x.cls != "FL"], x[x.cls == "FL"]
        a1.scatter(ogle.med_mag, ogle.med_err * 1e3, s=12, color=c, label=f"{VAR_LABEL[v]} (OGLE classes)" if len(ogle) else None)
        for _, g in fl.groupby("star_id"):
            g = g.sort_values("wfmag")
            a1.plot(g.med_mag, g.med_err * 1e3, "-", color=c, lw=0.5, alpha=0.5)
        if len(fl):
            a1.scatter(fl.med_mag, fl.med_err * 1e3, s=6, marker="x", color=c, label=f"{VAR_LABEL[v]} (FL)")
        a2.scatter(x.med_mag, x.p2p * 1e3, s=8, color=c, alpha=0.7)
    lo, hi = np.percentile(smarts, [5, 95])
    for a in (a1, a2):
        a.axhspan(lo, hi, color="0.5", alpha=0.15)
        a.set_yscale("log"); a.set_xlabel("median F146 magnitude")
        a.text(9.9, hi * 1.1, "SMARTS-TESS 30-min p2p (5–95%)", fontsize=7, color="0.3")
    a1.set_ylabel("quoted error per 12.1-min point [mmag]"); a2.set_ylabel("measured point-to-point scatter [mmag]")
    a1.set_title("Quoted F146 error vs magnitude (lines: one FL object at each magnitude copy)", fontsize=9)
    a2.set_title("Measured p2p (includes intrinsic variability of fast variables)", fontsize=9)
    a1.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=120); return fig


def fig_r_coverage(path=FIG / "r_cadence_coverage.png"):
    """F146/F087/F213 time stamps for one star: all 6 seasons, then a 1-day zoom."""
    z = np.load(RD / "examples/same_star.npz", allow_pickle=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 4.8))
    col = {"F146": "#4c72b0", "F087": "#55a868", "F213": "#c44e52"}
    t0 = z["times_F146"][0]
    for i, b in enumerate(["F146", "F087", "F213"]):
        t = z[f"times_{b}"]
        a1.plot(t - t0, np.full(len(t), i), "|", ms=10, color=col[b])
        m = (t - t0) < 1.0
        a2.plot((t[m] - t0) * 24, np.full(m.sum(), i), "|", ms=16, color=col[b])
    for a in (a1, a2):
        a.set_yticks([0, 1, 2]); a.set_yticklabels(["F146 (12.1 min)", "F087 (6 h)", "F213 (6 h, +3 h)"])
    cut = np.where(np.diff(z["times_F146"]) > 1)[0]
    a1.set_xlabel(f"days from JD {t0:.1f} (2027-08-14)"); a1.set_title(
        f"6 seasons × 71.0 d, gaps {', '.join(f'{g:.0f}' for g in np.diff(z['times_F146'])[cut])} d; "
        f"{len(z['times_F146']):,} F146 + {len(z['times_F087'])} + {len(z['times_F213'])} points", fontsize=9)
    a2.set_xlabel("hours from season start"); a2.set_title("first day", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); return fig


def fig_r_examples(path=FIG / "r_class_examples.png"):
    """Improved-noise F146 season 1 per class + all-season LS periodogram with the catalog P, P/2 and 2P marked."""
    classes = [c for c in ["FL", "ECL", "ELL", "T2CEP", "RRLYR", "DSCT", "HB", "LPV", "CV"] if (RD / f"examples/class__{c}.npz").exists()]
    fig, axes = plt.subplots(len(classes), 2, figsize=(12, 1.8 * len(classes)), gridspec_kw={"width_ratios": [2, 1.3]})
    for (a, b), c in zip(axes, classes):
        z = np.load(RD / f"examples/class__{c}.npz", allow_pickle=True)
        t, y = z["t"], z["y"]
        a.plot(t - t[0], y, ".", ms=0.8, color="k"); a.invert_yaxis()
        lab = f"{z['star_id']}" + (f" (TIC {z['template']})" if c == "FL" else "")
        a.text(0.01, 0.88, lab, transform=a.transAxes, fontsize=7); a.set_ylabel(c, rotation=0, ha="right")
        b.plot(1 / z["freq"], z["power"], color="0.3", lw=0.6); b.set_xscale("log"); b.set_yticks([])
        P = float(z["P_true"])
        if np.isfinite(P):
            for k, ls in ((1, "-"), (0.5, "--"), (2, "--")):
                b.axvline(P * k, color="#c44e52", ls=ls, lw=0.9)
            b.text(0.02, 0.85, f"catalog P = {P:.4g} d", transform=b.transAxes, fontsize=7, color="#c44e52")
        else:
            b.text(0.02, 0.85, "no catalog period", transform=b.transAxes, fontsize=7)
    axes[0, 0].set_title("improved noise, F146, season 1 (71 d)", fontsize=9)
    axes[0, 1].set_title("LS over all 6 seasons (red: catalog P solid, P/2 & 2P dashed)", fontsize=9)
    axes[-1, 0].set_xlabel("days"); axes[-1, 1].set_xlabel("period [d]")
    fig.tight_layout(); fig.savefig(path, dpi=110); return fig


def fig_r_period_check(path=FIG / "r_stored_vs_measured.png"):
    """Catalog period vs top LS peak (all 6 seasons, F146) for the improved variant."""
    f = pd.read_csv(RD / "period_check.csv")
    f = f[(f.variant == "improved")].dropna(subset=["P_true", "P1"])
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    cm = plt.get_cmap("tab10")
    for i, c in enumerate(["FL", "ECL", "ELL", "T2CEP", "RRLYR", "DSCT", "HB", "LPV"]):
        x = f[f.cls == c]
        if len(x):
            ax.scatter(x.P_true, x.P1, s=16, color=cm(i), label=f"{c} (n={len(x)})")
    lim = [0.02, 600]
    for k, ls in ((1, "-"), (0.5, "--"), (2, "--")):
        ax.plot(lim, np.multiply(lim, k), color="k", ls=ls, lw=0.7)
    ax.axhline(300, color="0.6", ls=":", lw=0.8); ax.text(0.03, 330, "LS grid limit 300 d", fontsize=7, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("catalog period [d] (FL: template literature P)"); ax.set_ylabel("top LS peak [d]")
    ax.legend(fontsize=7); ax.set_title("Resampled (improved): stored vs measured", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); return fig


def fig_r_pipeline(path=FIG / "r_pipeline_test.png"):
    """Pipeline answer vs truth on the 20 test stars (one season, improved noise), and wall time per star."""
    d = pd.read_csv(RD / "pipeline_test.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    cm = plt.get_cmap("tab10")
    lim = [0.1, 200]
    for i, c in enumerate(["FL", "ECL", "ELL", "T2CEP", "RRLYR"]):
        x = d[d.cls == c]
        ok = x.P_found.notna()
        a1.scatter(x.P_true[ok], x.P_found[ok], s=30, color=cm(i), label=c)
        a1.scatter(x.P_true[~ok], np.full((~ok).sum(), lim[0] * 1.15), marker="v", s=40, color=cm(i))
        a2.bar(np.arange(len(x)) + 5 * i, x.seconds, color=cm(i))
    for k, ls in ((1, "-"), (0.5, "--"), (2, "--")):
        a1.plot(lim, np.multiply(lim, k), color="k", ls=ls, lw=0.7)
    a1.axvline(35.5, color="0.5", ls=":", lw=0.8); a1.text(37, 0.2, "R/2 of one season", fontsize=7, color="0.4")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("catalog period [d] (ECL/ELL: P_orb)"); a1.set_ylabel("pipeline answer [d] (▼ = declined)")
    a1.legend(fontsize=8); a1.set_title("Pipeline on 20 resampled stars (F146, season 1, improved)", fontsize=9)
    a2.set_xticks([1.5 + 5 * i for i in range(5)]); a2.set_xticklabels(["FL", "ECL", "ELL", "T2CEP", "RRLYR"])
    a2.set_ylabel("seconds per star (i5, 1 process)"); a2.set_title(f"mean {d.seconds.mean():.2f} s / star-season", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); return fig
