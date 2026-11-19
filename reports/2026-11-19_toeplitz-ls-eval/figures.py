#!/usr/bin/env python3
"""figures.py -- figures for the Toeplitz-LS evaluation (P50) from data/*.csv / data/periodograms.npz.

    python reports/2026-11-19_toeplitz-ls-eval/figures.py            # all figures whose inputs exist
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK2 = "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": "#0b0b0b", "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9,
                     "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

PANELS = [("smarts-tess-v1.0-000732", "assigned", "lombscargle"),
          ("smarts-tess-v1.0-004513", "assigned", "lombscargle"),
          ("smarts-tess-v1.0-017292", "fullyear", "lombscargle"),
          ("smarts-tess-v1.0-066625", "assigned", "lombscargle"),
          ("smarts-tess-v1.0-201481", "fullyear", "lombscargle"),
          ("smarts-tess-v1.0-201481", "fullyear", "lombscargle_short")]


def fig_overlays():
    """Periodogram overlays (astropy fast vs tls) with residuals against the exact sum, one column per case."""
    z = np.load(DATA / "periodograms.npz")
    acc = pd.read_csv(DATA / "accuracy.csv")
    cand = pd.read_csv(DATA / "candidates_top10.csv")
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.0 * len(PANELS), 4.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    for j, (sid, mode, grid) in enumerate(PANELS):
        k = f"{sid}|{mode}|{grid}"
        f = z[f"{k}|freq"]
        per = 1 / f
        pa, pe, pt, pf = (z[f"{k}|{c}"] for c in ("astropy_fast", "astropy_exact", "tls", "tlsf"))
        a = axes[0, j]
        a.plot(per, pa, color=BLUE, lw=1.6, label="astropy fast (today)")
        a.plot(per, pt, color=ORANGE, lw=0.8, ls="--", label="Toeplitz-LS tls")
        ptrue = acc.query("star_id == @sid").true_period.iloc[0]
        if per.min() <= ptrue <= per.max():
            a.axvline(ptrue, color=INK2, lw=0.8, ls=":")
        for meth, mk, col in (("astropy_fast", "v", BLUE), ("tls", "^", ORANGE)):
            ps = json.loads(cand.query("star_id==@sid and mode==@mode and grid==@grid and method==@meth").periods.iloc[0])
            yv = np.interp(ps, per[::-1], pt[::-1])
            a.scatter(ps, yv + (0.04 if mk == "^" else 0.08) * np.nanmax(pt), marker=mk, s=18, color=col,
                      zorder=5, edgecolor="#fcfcfb", linewidth=0.6)
        a.set_xscale("log")
        a.set_title(f"{sid[-6:]}  P={ptrue:.2f} d\n{mode}, {grid.replace('lombscargle', 'LS')}", fontsize=8.5)
        if j == 0:
            a.set_ylabel("LS power (standard)")
            a.legend(loc="upper left", fontsize=7)
        b = axes[1, j]
        b.semilogy(per, np.abs(pa - pe) + 1e-18, color=BLUE, lw=0.8, label="|astropy fast − exact|")
        b.semilogy(per, np.abs(pt - pe) + 1e-18, color=ORANGE, lw=0.8, label="|tls − exact|")
        b.semilogy(per, np.abs(pf - pe) + 1e-18, color=AQUA, lw=0.6, alpha=0.8, label="|tlsf − exact|")
        b.set_ylim(1e-17, 1e-2)
        b.set_xlabel("period (d)")
        if j == 0:
            b.set_ylabel("|Δ power| vs exact")
            handles = b.get_lines()
    fig.suptitle("Periodograms on guess_lombscargle's grids: Toeplitz-LS overlays astropy; its error vs the exact "
                 "direct sum is ~1e-14, astropy fast's is 1e-7–1e-4.  Markers: top-10 candidates (▼ astropy, ▲ tls); "
                 "dotted line: true period.", fontsize=8.5, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=3, fontsize=8)
    fig.savefig(FIG / "fig_periodogram_overlays.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_error_summary():
    """Max |Δpower| vs the exact sum per case, for each backend."""
    acc = pd.read_csv(DATA / "accuracy.csv")
    acc["case"] = acc.star_id.str[-6:] + " " + acc["mode"].str[:4] + " " + acc.grid.str.replace("lombscargle", "LS")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(acc))
    for name, col, lab in (("astropy_fast", BLUE, "astropy fast (today)"), ("tls", ORANGE, "Toeplitz-LS tls (float64)"),
                           ("tlsf", AQUA, "Toeplitz-LS tlsf (float32)")):
        ax.semilogy(x, acc[f"{name}_vs_exact_maxabs"], "o", ms=5, color=col, label=lab, mec="#fcfcfb")
    flip = ~acc.astropy_fast_top10_same_as_exact
    for i in np.where(flip)[0]:
        ax.annotate("astropy fast top-10\n≠ exact (tls = exact)", (i, acc.astropy_fast_vs_exact_maxabs[i]),
                    xytext=(-150, -30), textcoords="offset points", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xticks(x, acc.case, rotation=70, ha="right", fontsize=6.5)
    ax.set_ylabel("max |Δ power| vs exact")
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.3))
    ax.set_title("Max power error against astropy's exact (cython) sum, 5 stars × 2 coverages × 2 grids", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_error_summary.png", dpi=150)
    plt.close(fig)


def fig_timing():
    """Log-log time per periodogram vs N and vs Nf, plus the real-star grids."""
    tm = pd.read_csv(DATA / "timing.csv")
    sty = {"astropy_fast": (BLUE, "astropy fast (today)"), "tls_generic": (VIOLET, "tls, make generic"),
           "tls_native": (ORANGE, "tls, make native"), "tlsf_native": (AQUA, "tlsf (float32), native")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, exp, xcol, xl in ((axes[0], "scan_N", "N", "N points (Nf = 4000)"),
                              (axes[1], "scan_Nf", "Nf", "Nf frequencies (N = 14,000)")):
        for b, (col, lab) in sty.items():
            d = tm.query("exp == @exp and backend == @b").sort_values(xcol)
            ax.loglog(d[xcol], d.seconds * 1e3, "o-", color=col, lw=1.6, ms=4, label=lab, mec="#fcfcfb")
        ax.set_xlabel(xl)
        ax.set_ylabel("ms per periodogram (1 thread)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("time vs N", fontsize=9)
    axes[1].set_title("time vs grid size", fontsize=9)
    real = tm.query("exp == 'real'")
    ax = axes[2]
    for b, (col, lab) in sty.items():
        d = real.query("backend == @b")
        ax.loglog(d.N * d.Nf, d.seconds * 1e3, "o", color=col, ms=4.5, label=lab, mec="#fcfcfb")
    ax.set_xlabel("N × Nf (real-star grids)")
    ax.set_ylabel("ms per periodogram")
    ax.set_title("our 20 real grids (5 stars × 2 coverages × 2 grids)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_timing.png", dpi=150)
    plt.close(fig)


def fig_end_to_end():
    """Share of per-star time spent in LS today and the projected saving if LS calls used Toeplitz-LS."""
    p = DATA / "end_to_end.csv"
    e = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    y = np.arange(len(e))
    ax.barh(y, e.total_ms, color="#e6e5e0", height=0.55, label="rest of the per-star pipeline")
    ax.barh(y, e.ls_ms, color=BLUE, height=0.55, label="Lomb-Scargle today")
    ax.barh(y + 0.14, e.ls_ms_tls, color=ORANGE, height=0.2, label="LS with tls (float64, native)")
    ax.barh(y - 0.14, e.ls_ms_tlsf, color=AQUA, height=0.2, label="LS with tlsf (float32, native)")
    for i, r in e.iterrows():
        ax.text(r.total_ms * 1.01, i, f"LS = {100 * r.ls_share:.1f}%;  saved: tls {100 * r.saved_frac_tls:+.2f}%, "
                f"tlsf {100 * r.saved_frac_tlsf:+.2f}%", va="center", fontsize=7.5, color="#0b0b0b")
    ax.set_yticks(y, e.label, fontsize=8)
    ax.set_xlabel("serial ms per star")
    ax.set_xlim(0, e.total_ms.max() * 1.75)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("End-to-end: Lomb-Scargle is a small slice of per-star time", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_end_to_end.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for fn, need in ((fig_overlays, "periodograms.npz"), (fig_error_summary, "accuracy.csv"),
                     (fig_timing, "timing.csv"), (fig_end_to_end, "end_to_end.csv")):
        if (DATA / need).exists():
            fn()
            print("made", fn.__name__)
