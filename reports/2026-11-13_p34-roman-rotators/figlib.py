"""Figures for the P34 report, from data/ only (no drive). Called by notebook.ipynb; also `python figlib.py`."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
sys.path.insert(0, str(ROOT / "tools"))
import make_roman_rotators as mr  # noqa: E402

D, F = HERE / "data", HERE / "figures"
R2, R3 = 35.5, 71.0 / 3
OUTC = {"correct": "#1b9e77", "diffrot": "#66a61e", "half": "#e6ab02", "double": "#e6ab02", "wrong": "#d95f02",
        "none": "#999999"}

# Schematic literature-informed period distributions (log-normal in P; medians/widths in the report's citation table).
# NOT fitted to catalog data (no catalog downloaded); they summarise the cited papers for comparison only.
LIT_PERIOD = {
    "F (old bulge MS)": [(1.0, 8.0, 0.25)],
    "G (old bulge MS)": [(1.0, 25.0, 0.15)],
    "K (old bulge MS)": [(1.0, 38.0, 0.15)],
    "M (disk field)":   [(0.25, 1.0, 0.45), (0.75, 70.0, 0.2)],
}
LIT_COLORS = {"F (old bulge MS)": "#4575b4", "G (old bulge MS)": "#91bfdb", "K (old bulge MS)": "#fc8d59",
              "M (disk field)": "#d73027"}


def ogle_f146(r):
    """Rough F146 for OGLE ROT stars: F146 ~ I - 0.6 (V - I) (+-0.5 mag; reddened K giants, see README)."""
    vi = (r.V - r.I).fillna((r.V - r.I).median())
    return r.I - 0.6 * vi


def fig_population():
    r = pd.read_csv(D / "ogle_rot_catalog.csv.gz")
    fl = pd.read_csv(D / "fl_templates.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.logspace(-1, np.log10(250), 45)
    a = ax[0]
    a.hist(r.P, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color="k",
           label=f"OGLE ROT, all {len(r):,} (Iwanek+2024)")
    a.hist(fl.P_true, bins=bins, density=True, histtype="step", lw=2, color="m",
           label=f"WG6 FL templates ({len(fl)})")
    x = np.logspace(-1, np.log10(250), 400)
    for k, comps in LIT_PERIOD.items():
        y = sum(w * np.exp(-0.5 * ((np.log10(x) - np.log10(m)) / s) ** 2) / (s * np.sqrt(2 * np.pi)) for w, m, s in comps)
        a.plot(x, y / (x * np.log(10)), color=LIT_COLORS[k], lw=2, label=f"{k}: literature (schematic)")
    for v, ls, lab in ((R3, ":", "R/3 = 23.7 d"), (R2, "--", "R/2 = 35.5 d")):
        a.axvline(v, color="k", ls=ls, lw=1, label=lab)
    a.set(xscale="log", yscale="log", ylim=(1e-4, 3), xlabel="period [d]", ylabel="density per day",
          title="Periods: OGLE ROT vs FL vs expected Roman dwarfs")
    a.legend(fontsize=7, loc="lower left")
    a = ax[1]
    f146 = ogle_f146(r)
    a.hist(f146, bins=np.arange(9, 22, 0.25), histtype="stepfilled", alpha=0.35, color="k", label="OGLE ROT (est. F146)")
    a.axvspan(9, 16, color="r", alpha=0.08)
    a.axvline(16, color="r", lw=1.5, label="F146 saturation ~16 (Claytor & Tayar 2026)")
    a.axvline(18, color="orange", lw=1.5, label="semi-saturation ~18")
    a.axvspan(18, 23, color="g", alpha=0.08, label="bulk of forecast Roman rotators")
    a.axvspan(16, 21, fill=False, hatch="//", ec="m", lw=0, label="FL WFmag 16-21")
    sat = float((f146 < 16).mean())
    a.set(xlabel="F146 [mag] (OGLE: rough I - 0.6(V-I), +-0.5)", ylabel="stars",
          title=f"Brightness: {sat:.0%} of OGLE ROT est. brighter than F146 16")
    a.legend(fontsize=7, loc="upper right")
    a = ax[2]
    a.scatter(r.P, 1000 * r.Iamp, s=1, alpha=0.15, color="k", label="OGLE ROT: I amplitude")
    for st, pr in mr.TYPE_PRIORS.items():
        lo, hi = pr["P"]
        a.plot([lo, hi], [pr["rvar_kep"][0] * mr.F146_AMP_FACTOR] * 2, lw=3, label=f"prior Rvar(F146): {st}")
    for m, c in ((16, "0.6"), (18, "0.4"), (20, "0.2"), (22, "0.0")):
        a.axhline(float(mr.sigma_f146_mmag(m)), color=c, ls="--", lw=1)
        a.text(0.12, float(mr.sigma_f146_mmag(m)) * 1.08, f"sigma/point F146={m}", fontsize=7, color=c)
    a.set(xscale="log", yscale="log", xlabel="period [d]", ylabel="amplitude [mmag]", ylim=(0.1, 3000),
          title="Amplitudes: OGLE giants vs dwarf priors vs Roman noise")
    a.legend(fontsize=6, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(F / "population.png", dpi=130)
    plt.close(fig)
    return dict(n_rot=len(r), frac_sat=sat, frac_f146_lt18=float((f146 < 18).mean()), frac_beyond=float((r.P > R2).mean()),
                frac_short=float((r.P < 2).mean()), n_fl=len(fl), fl_frac_lt5=float((fl.P_true < 5).mean()))


def _lcs():
    z = np.load(D / "prototype_lcs.npz")
    return z["time"].astype(float), list(z["ids"]), z["flux"].astype(float)


def fig_examples(stars=("proto04_K_12d", "proto06_M_3.5d", "proto02_G_30d")):
    t, ids, flux = _lcs()
    res = pd.read_csv(D / "prototype_results.csv").set_index("star_id")
    fig, ax = plt.subplots(len(stars), 3, figsize=(16, 3.0 * len(stars)), sharex=True)
    nb = 10  # 10 x 12.1 min ~ 2 h bins
    for i, s in enumerate(stars):
        for j, m in enumerate((16, 18, 20)):
            sid = f"{s}_m{m}"
            f = 1000 * flux[ids.index(sid)]
            row = res.loc[sid]
            a = ax[i, j]
            a.plot(t, f, ",", color="0.6", alpha=0.6)
            n = (len(t) // nb) * nb
            a.plot(t[:n].reshape(-1, nb).mean(1), f[:n].reshape(-1, nb).mean(1), ".", ms=2, color="k")
            for k in np.arange(0, 71, row.P_eq):
                a.axvline(k, color="tab:blue", lw=0.6, alpha=0.5)
            pf = "-" if pd.isna(row.P_found) else f"{row.P_found:.2f} d"
            a.set_title(f"{s}  F146={m}: Rvar/sigma={row.snr_rvar:.2f}, found {pf} ({row.outcome})", fontsize=8,
                        color=OUTC.get(row.outcome, "k"))
            lim = np.percentile(np.abs(f), 99.5)
            a.set_ylim(lim, -lim)  # magnitude-like: dimming down
            if j == 0:
                a.set_ylabel("rel. flux [ppt]")
    for a in ax[-1]:
        a.set_xlabel("days since season-1 start (WG6 grid); blue lines every P_eq")
    fig.tight_layout()
    fig.savefig(F / "examples_f146_16_18_20.png", dpi=120)
    plt.close(fig)


def fig_acf(stars=("proto04_K_12d", "proto02_G_30d", "proto09_giant_22d")):
    z = np.load(D / "prototype_acf.npz")
    res = pd.read_csv(D / "prototype_results.csv").set_index("star_id")
    fig, ax = plt.subplots(1, len(stars), figsize=(16, 3.8))
    for a, s in zip(ax, stars):
        lags, acf, cands, Pfit = z[s + "_lags"], z[s + "_acf"], z[s + "_cands"], float(z[s + "_Pfit"][0])
        row = res.loc[f"{s}_m18"]
        a.plot(lags, acf, color="k", lw=1)
        for c in cands:
            a.axvline(c, ymin=0, ymax=0.08, color="tab:orange", lw=1.2)
        for k in range(1, 6):
            a.axvline(k * row.P_eq, color="tab:blue", ls="--", lw=0.8, alpha=0.7)
        a.axvspan(row.P_lo, row.P_hi, color="tab:blue", alpha=0.15)
        if np.isfinite(Pfit):
            a.axvline(Pfit, color="tab:red", lw=1.5)
        a.axvline(R2, color="0.5", ls=":")
        a.set(xlim=(0, 50), xlabel="lag [d]", ylabel="ACF",
              title=f"{s} F146=18: P_eq={row.P_eq:g} d, tau_spot={row.tau_spot_days:.0f} d -> {row.outcome}")
        a.title.set_fontsize(8)
    ax[0].plot([], [], color="tab:orange", label="pooled candidates (rug)")
    ax[0].plot([], [], color="tab:blue", ls="--", label="k * P_eq (band shaded)")
    ax[0].plot([], [], color="tab:red", label="comb-fit P")
    ax[0].plot([], [], color="0.5", ls=":", label="R/2")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "acf_candidates.png", dpi=130)
    plt.close(fig)


def fig_outcomes():
    res = pd.read_csv(D / "prototype_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for o, g in res.groupby("outcome"):
        ax.scatter(g.P_eq * (1 + 0.04 * (g.f146 - 18)), g.snr_rvar, s=40, color=OUTC.get(o, "k"), label=f"{o} ({len(g)})",
                   edgecolor="k", lw=0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.axvline(R3, color="k", ls=":", lw=1)
    ax.axvline(R2, color="k", ls="--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="P_eq [d] (jittered by magnitude)", ylabel="Rvar(F146) / sigma per point",
           title="Prototype: 10 stars x F146 16/18/20, WG6 season 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(F / "prototype_outcomes.png", dpi=130)
    plt.close(fig)
    return res.outcome.value_counts().to_dict()


def strata_table(n_per_cell=100, n_holdout=25, mags=(17.0, 19.0, 21.0)):
    """Proposed v0 strata: type x period bin cells allowed by TYPE_PRIORS; each star rendered at every magnitude."""
    rows = []
    for st, pr in mr.TYPE_PRIORS.items():
        for lo, hi in mr.PERIOD_BINS:
            ok = max(lo, pr["P"][0]) < min(hi, pr["P"][1])
            rows.append(dict(stype=st, p_bin=f"{lo:g}-{hi:g}", beyond_reach=lo >= R2, allowed=ok,
                             n_dev=n_per_cell if ok else 0, n_holdout=n_holdout if ok else 0))
    t = pd.DataFrame(rows)
    t["n_star_seasons"] = (t.n_dev + t.n_holdout) * len(mags)
    return t


def fig_strata(sec_per_star_season=1.0):
    t = strata_table()
    t.to_csv(D / "proposed_strata.csv", index=False)
    piv = t.pivot(index="stype", columns="p_bin", values="n_dev").loc[list(mr.TYPE_PRIORS)]
    piv = piv[[f"{lo:g}-{hi:g}" for lo, hi in mr.PERIOD_BINS]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(np.where(piv.values > 0, 1, 0), cmap="Greens", vmin=0, vmax=2.5, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v} + 25" if v else "-", ha="center", va="center", fontsize=9)
    ax.axvline(4.5, color="r", lw=2)
    ax.text(4.55, -0.6, "beyond R/2 ->", color="r", fontsize=8)
    ax.set_xticks(range(piv.shape[1]), [f"{c} d" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    n_stars = int((t.n_dev + t.n_holdout).sum())
    n_ss = int(t.n_star_seasons.sum())
    ax.set_title(f"Proposed romrot_v0: dev + holdout stars per cell, each at F146 17/19/21\n"
                 f"{n_stars:,} stars, {n_ss:,} star-seasons, ~{n_ss * sec_per_star_season / 3600:.1f} CPU-h",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(F / "proposed_strata.png", dpi=130)
    plt.close(fig)
    return dict(n_cells=int(t.allowed.sum()), n_stars=n_stars, n_star_seasons=n_ss)


def main():
    F.mkdir(exist_ok=True)
    out = dict(population=fig_population(), outcomes=fig_outcomes(), strata=fig_strata())
    fig_examples()
    fig_acf()
    print(out)
    return out


if __name__ == "__main__":
    main()
