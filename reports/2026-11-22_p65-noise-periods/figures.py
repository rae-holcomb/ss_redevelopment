"""P65 figures, from data/ only (run analysis.py first).

The offender stack light curves come from data/offender_stacks.npz; if it is missing, pass --stack-dir (a
local copy of the drive's extracted/galaxy_stack with stacks.npy and stack_index.csv) to build it once.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, FIG = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY, INK = "#8a8a85", "#2b2b28"
OFFENDERS = [10001742265, 10001638867, 10001514155, 10002408980, 10001658496, 10001478100, 10001742277]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def stacks(stack_dir):
    """Binned (2 h) offender stacks, cached in data/offender_stacks.npz."""
    path = os.path.join(DATA, "offender_stacks.npz")
    if not os.path.exists(path):
        idx = pd.read_csv(os.path.join(stack_dir, "stack_index.csv"))
        S = np.load(os.path.join(stack_dir, "stacks.npy"), mmap_mode="r")
        out = []
        for t in OFFENDERS:
            g = np.asarray(S[int(np.flatnonzero(idx.tic == t)[0]), 2], float)
            g[~np.isfinite(g) | (np.abs(g - 1) > 0.1)] = np.nan
            with np.errstate(all="ignore"):
                out.append(np.nanmedian(g.reshape(-1, 4), axis=1))
        np.savez_compressed(path, tic=np.array(OFFENDERS), t=np.arange(len(out[0])) * 4 / 48.0,
                            flux=np.array(out, np.float32))
    return np.load(path)


def fig_rates():
    """Part (b): success and wrong rates with and without the 22 periodic_low_amplitude stars."""
    r = pd.read_csv(os.path.join(DATA, "rates.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, metric in zip(axes, ("success", "wrong")):
        x = 0
        ticks, labels = [], []
        for mode in ("fullyear", "assigned"):
            for meth in ("hand_v9", "ranker_v1"):
                for j, sub in enumerate(("all 874", "without the 22")):
                    v = r[(r["mode"] == mode) & (r.method == meth) & (r.subset == sub)][metric].iloc[0]
                    ax.bar(x + 0.38 * j, v, 0.36, color=C[j], label=sub if x == 0 else None)
                    ax.text(x + 0.38 * j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
                ticks.append(x + 0.19)
                labels.append(f"{meth.replace('_', ' ')}\n{mode}")
                x += 1
        ax.set_xticks(ticks, labels)
        ax.set_title(f"{metric} rate on dev_medium_clean", color=INK)
        ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rates_with_without_22.png"), dpi=150)


def fig_wrong_snr():
    """Noise-stack S/N at each answered-wrong top-1 period vs the shuffled-source null; and the breakdown."""
    ex = pd.read_csv(os.path.join(DATA, "wrong_explained.csv"))
    w = ex[ex.category == "answered_wrong"]
    null = np.load(os.path.join(DATA, "null_snr.npy")).ravel()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    x = np.logspace(-1, 2.5, 200)
    nl = null[np.isfinite(null)]
    ws = w.noise_snr_at_top.dropna().values
    a.plot(x, [(nl >= v).mean() for v in x], color=GREY, lw=2,
           label=f"same period in 50 random stacks each (n={len(nl)})")
    a.plot(x, [(ws >= v).mean() for v in x], color=C[1], lw=2,
           label=f"answered-wrong top-1, its own stack (n={len(ws)})")
    a.axvline(5, color=INK, ls="--", lw=1)
    a.text(5.4, 0.5, "S/N 5", color=INK, fontsize=8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(1e-4, 1.2)
    a.set_xlabel("noise-stack min-half LS S/N at the reported period (max of P, P/2, 2P)")
    a.set_ylabel("fraction with S/N >= x")
    a.legend(frameon=False, fontsize=7.5, loc="lower left")
    order = ["coherent noise signal at P_ref", "coherent secondary noise signal",
             "P_ref, not coherent (red-noise peak)", "no noise-period match"]
    classes = ["periodic_low_amplitude", "quiet", "systematic_common_mode", "misclassified_star"]
    tab = pd.crosstab(w.why, w.noise_classification).reindex(index=order, columns=classes, fill_value=0)
    left = np.zeros(len(order))
    for k, cl in enumerate(classes):
        b.barh(range(len(order)), tab[cl], left=left, color=C[k], label=cl, height=0.6, edgecolor="white", lw=1)
        left += tab[cl].values
    for i, v in enumerate(left):
        b.text(v + 0.3, i, str(int(v)), va="center", color=INK)
    b.set_yticks(range(len(order)), [o.replace(" (", "\n(") for o in order])
    b.invert_yaxis()
    b.set_xlabel("answered-wrong star-tables (v1 pick, 4 dev tables)")
    b.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="noise source class",
             title_fontsize=7)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wrong_vs_noise_snr.png"), dpi=150)


def fig_offenders(stack_dir):
    """Per offender source: the stack, its full-year LS, and a zoom on the wrong answers (both halves)."""
    st = stacks(stack_dir)
    sp = np.load(os.path.join(DATA, "stack_peaks_spectra.npz"))
    ex = pd.read_csv(os.path.join(DATA, "wrong_explained.csv"))
    fl = pd.read_csv(os.path.join(DATA, "source_flags.csv")).set_index("tic")
    n = len(OFFENDERS)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.0 * n), gridspec_kw=dict(width_ratios=[1.3, 1.2, 1]))
    for i, tic in enumerate(OFFENDERS):
        a, b, c = axes[i]
        k = list(st["tic"]).index(tic)
        a.plot(st["t"], (st["flux"][k] - 1) * 1e3, lw=0.4, color=INK)
        a.set_ylabel("flux - 1 (ppt)")
        a.set_title(f"TIC {tic} ({fl.classification[tic]}), P_ref {fl.P_ref[tic]:.2f} d", fontsize=8, loc="left")
        j = list(sp["tic"]).index(tic)
        f, p = sp["f_full"], sp["p_full"][j]
        b.plot(1 / f, p, lw=0.5, color=INK)
        b.set_xscale("log")
        b.axvline(fl.P_ref[tic], color=C[0], lw=1.5, alpha=0.7)
        wr = ex[(ex.noise_tic == tic) & (ex.category == "answered_wrong")]
        for P in wr.top_period:
            b.axvline(P, color=C[1], lw=1, ls="--")
        b.set_ylabel("LS power (full year)")
        # zoom on the wrong answers: +-8% around their median, each half overlaid
        P0 = float(np.median(wr.top_period)) if len(wr) else fl.P_ref[tic]
        fh = sp["f_half"]
        m = np.abs(fh * P0 - 1) < 0.08
        for h, col, lab in ((0, C[2], "first half"), (1, C[4], "second half")):
            c.plot(1 / fh[m], sp["p_half"][j][h][m], lw=1, color=col, label=lab)
        for P in wr.top_period:
            c.axvline(P, color=C[1], lw=1, ls="--")
        c.set_ylabel("LS power (half year)")
        c.set_title(f"zoom near the wrong answers ({len(wr)}), around {P0:.3f} d", fontsize=8, loc="left")
        if i == 0:
            b.plot([], [], color=C[0], label="P_ref")
            b.plot([], [], color=C[1], ls="--", label="v1 wrong answers")
            b.legend(frameon=False, fontsize=7, loc="upper left")
            c.legend(frameon=False, fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time (d)" if ax is axes[-1][0] else "period (d)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "offenders.png"), dpi=130)


def fig_population():
    """Secondary coherent signals (not P_ref) across all noise sources: period vs S/N by class."""
    pk = pd.read_csv(os.path.join(DATA, "stack_peaks.csv"))
    fl = pd.read_csv(os.path.join(DATA, "source_flags.csv")).set_index("tic")
    s = json.load(open(os.path.join(DATA, "summary.json")))
    sec = pd.read_csv(os.path.join(DATA, "secondary_sources.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    # best non-P_ref S/N per source, all sources, split kept vs excluded
    pk = pk.join(fl, on="tic")
    r = pk.P / pk.P_ref
    pk["is_pref"] = (abs(r - 1) < 0.02) | (abs(r - 0.5) < 0.01) | (abs(r - 2) < 0.04)
    best = pk[~pk.is_pref].sort_values("min_half_snr").groupby("tic").tail(1).set_index("tic")
    for k, cls in enumerate(["quiet", "systematic_common_mode", "periodic_low_amplitude"]):
        g = best[best.classification == cls]
        a.scatter(g.P, g.min_half_snr, s=9, color=C[k], alpha=0.7, label=f"{cls} ({len(g)})", lw=0)
    a.axhline(10, color=INK, ls="--", lw=1)
    a.text(0.11, 11.5, "S/N 10 (blind-search cut)", fontsize=7.5, color=INK)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("period of the source's strongest non-P_ref signal (d)")
    a.set_ylabel("min-half LS S/N")
    a.legend(frameon=False, fontsize=7.5, loc="upper right")
    a.set_title("most coherent secondary signal per noise source", fontsize=9, loc="left")
    bins = np.logspace(np.log10(0.1), np.log10(5), 40)
    for k, cls in enumerate(["quiet", "systematic_common_mode"]):
        b.hist(sec[sec.classification == cls].P_secondary, bins, color=C[k], alpha=0.75, label=cls,
               histtype="stepfilled" if k == 0 else "step", lw=1.5)
    b.set_xscale("log")
    b.set_xlabel("secondary period (d), S/N >= 10")
    b.set_ylabel("noise sources")
    b.set_title(f"{s['kept_sources_with_secondary_ge10']} kept-class sources carry one", fontsize=9, loc="left")
    b.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "secondary_population.png"), dpi=150)


def fig_categories():
    """How often the top-1 sits on a coherent noise signal, per v1 loss category."""
    s = json.load(open(os.path.join(DATA, "summary.json")))["noise_coh_rate_by_category"]
    ex = pd.read_csv(os.path.join(DATA, "wrong_explained.csv"))
    n = ex.category.value_counts()
    cats = sorted(s, key=s.get)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(range(len(cats)), [s[c] for c in cats], color=C[0], height=0.6)
    for i, c in enumerate(cats):
        ax.text(s[c] + 0.005, i, f"{s[c]:.1%} of {n[c]}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(cats)), cats)
    ax.set_xlabel("fraction of star-tables whose top-1 has noise-stack S/N >= 5")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "noise_coherent_by_category.png"), dpi=150)


def main():
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack-dir", default=None)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    fig_rates()
    fig_wrong_snr()
    fig_offenders(a.stack_dir)
    fig_population()
    fig_categories()


if __name__ == "__main__":
    main()
