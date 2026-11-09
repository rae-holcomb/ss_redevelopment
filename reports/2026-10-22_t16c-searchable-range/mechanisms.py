"""T16c per-method degradation mechanisms by coverage (diagnosis only; fixes are T16d).

Figures (figures/):
  mech_pair_fraction.png  valid-pair fraction vs lag for typical patterns, with
                          0.3 / 0.25 thresholds and R: why R ~ 0.69 x the longest
                          segment and never bridges a sector gap (ACF truncation).
  mech_ls_window.png      a clean P = 10 d sinusoid's Lomb-Scargle periodogram under
                          each sampling: sector-gap sidelobes (LS window aliasing).
  mech_methods.png        from the before runs: per-method Stage 1 failure signatures
                          by coverage class (pairwise_histogram list length,
                          LS / wavelet top-1 ratio scatter, wavelet recall vs gap).
  mech_examples.png       example stars: light curve, ACF truncated at R, true P and
                          each method's top-1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
import coverage as cov  # noqa: E402
from synthetic import make_lightcurve  # noqa: E402
from acf_utils import compute_acf  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
PATTERNS = {"single (1)": (1,), "consecutive-2 (1-2)": (1, 2), "gapped-2 (1-3)": (1, 3),
            "gapped-2 (1-13)": (1, 13), "gapped-3-5 (1-2-5)": (1, 2, 5), "full (1-13)": tuple(range(1, 14))}


def pair_fraction_figure():
    """Valid-pair fraction vs lag, R (0.3) and the 0.25 alternative, per pattern."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    rows = []
    for ax, (name, pat) in zip(axes.flat, PATTERNS.items()):
        lc = make_lightcurve(5.0, n_sectors=13, sectors=pat, seed=0)
        lags, counts, _ = cov.sampling_pair_counts(lc.time, lc.flux)
        frac = counts / counts[0]
        r30 = cov.reach_from_pair_counts(lags, counts, 0.3)
        r25 = cov.reach_from_pair_counts(lags, counts, 0.25)
        longest = cov.sampling_descriptors(lc.time, lc.flux)["longest_contiguous_days"]
        rows.append(dict(pattern=name, longest_segment=longest, R_030=r30, R_025=r25,
                         R_over_longest=r30 / longest, gain_025=r25 / r30 - 1))
        ax.plot(lags, frac, lw=0.8, color="0.3")
        for thr, col in ((0.3, "C3"), (0.25, "C1")):
            ax.axhline(thr, color=col, lw=0.7, ls=":")
        ax.axvline(r30, color="C3", lw=1, label=f"R(0.3) = {r30:.1f} d")
        ax.axvline(r25, color="C1", lw=1, ls="--", label=f"R(0.25) = {r25:.1f} d")
        ax.axvline(r30 / 2, color="C0", lw=1, label=f"P_max = R/2 = {r30 / 2:.1f} d")
        ax.set_xlim(0, min(lags[-1], max(3 * r30, 60)))
        ax.set_title(f"{name}; longest seg {longest:.1f} d", fontsize=9)
        ax.legend(fontsize=7)
    fig.supxlabel("lag (days)")
    fig.supylabel("valid-pair fraction count(L) / count(0)")
    fig.tight_layout()
    fig.savefig(FIG / "mech_pair_fraction.png", dpi=130)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "data" / "pair_fraction_patterns.csv", index=False)
    return t


def ls_window_figure(period=10.0):
    """LS periodogram of a noiseless P-day sinusoid under each sampling."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False, sharex=True)
    for ax, (name, pat) in zip(axes.flat, PATTERNS.items()):
        lc = make_lightcurve(period, n_sectors=13, sectors=pat, seed=0)
        m = np.isfinite(lc.flux)
        t = lc.time[m]
        y = np.sin(2 * np.pi * t / period)
        freq = np.linspace(1 / 60, 1 / 3, 6000)
        pw = LombScargle(t, y).power(freq)
        ax.plot(1 / freq, pw, lw=0.7)
        ax.axvline(period, color="C2", lw=0.8, ls=":")
        ax.axvspan(period * 0.9, period * 1.1, color="C2", alpha=0.12)
        ax.set_xscale("log")
        ax.set_title(name, fontsize=9)
    fig.supxlabel("period (days); green band = true P +/- REL_TOL")
    fig.supylabel("LS power")
    fig.suptitle(f"Lomb-Scargle of a noiseless P = {period:g} d sinusoid: sidelobes from the sector-gap window",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "mech_ls_window.png", dpi=130)
    plt.close(fig)


def method_signatures(label="dm_assigned", runs=("dm_assigned",)):
    """Per-method Stage 1 failure signatures by coverage class, from the before runs.

    Panels: (a) Stage 1 recall per method and coverage class; (b) pairwise_histogram
    list length (how many candidates it can still offer); (c) lombscargle top-1 /
    true-P ratio, gapped vs consecutive classes (window sidelobes sit at ~0.85 and
    ~1.2 for a one-sector gap); (d) wavelet recall vs largest gap; (e) ACF-based
    methods' recall vs true P/R (truncation edge).
    """
    import analysis as an
    pm = an.method_table("before", list(runs))
    classes = [c for c in an.CLASS_ORDER if c in set(pm.coverage_class)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = ax.flat
    rec = pm.pivot_table(index="method", columns="coverage_class", values="recall", aggfunc="mean")[classes]
    im = ax[0].imshow(rec.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax[0].set_yticks(range(len(rec)), rec.index, fontsize=7)
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            ax[0].text(j, i, f"{rec.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="w" if rec.values[i, j] < 0.5 else "k")
    ax[0].set_title("(a) Stage 1 recall (success anywhere in list)", fontsize=9)
    ph = pm[pm.method == "pairwise_histogram"]
    ax[1].boxplot([ph[ph.coverage_class == c]["n_candidates"] for c in classes], showfliers=False)
    ax[1].set_xticks(range(1, len(classes) + 1), classes, rotation=45, ha="right", fontsize=7)
    ax[1].set_ylabel("candidates returned (n_guesses = 10)")
    ax[1].set_title("(b) pairwise_histogram list length", fontsize=9)
    ls = pm[(pm.method == "lombscargle") & pm.top1.notna() & (pm.truth_over_R < 0.5)].copy()
    r = ls.top1_over_true
    ls["sidelobe"] = ((r > 0.75) & (r < 0.9)) | ((r > 1.1) & (r < 1.35))
    ls["within_tol"] = (r - 1).abs() <= 0.1
    lt = ls.groupby("coverage_class")[["within_tol", "sidelobe"]].mean().reindex(classes)
    xx = np.arange(len(classes))
    ax[2].bar(xx - 0.2, lt["within_tol"], 0.4, color="C2", label="top-1 within REL_TOL of P")
    ax[2].bar(xx + 0.2, lt["sidelobe"], 0.4, color="C3", label="top-1 in sidelobe zone (0.75-0.9, 1.1-1.35 P)")
    ax[2].set_xticks(xx, classes, rotation=45, ha="right", fontsize=7)
    ax[2].set_ylim(0, 0.8)
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) lombscargle top-1 (stars with P/R < 0.5)", fontsize=9)
    lt.to_csv(HERE / "data" / f"ls_top1_zones_{label}.csv")
    wv = pm[pm.method == "wavelet"].copy()
    gap_bins = [0, 2, 10, 30, 60, 120, 400]
    wv["gap_bin"] = pd.cut(wv.largest_gap_days, gap_bins)
    for pr_lo, pr_hi, col in ((0, 0.1, "C0"), (0.1, 0.5, "C1"), (0.5, 9, "C3")):
        sel = wv[(wv.truth_over_R >= pr_lo) & (wv.truth_over_R < pr_hi)]
        t = sel.groupby("gap_bin", observed=False)["recall"].mean()
        ax[3].plot(range(len(t)), t.values, marker="o", color=col, label=f"P/R in [{pr_lo}, {pr_hi})")
    ax[3].set_xticks(range(len(gap_bins) - 1), [f"{a}-{b}" for a, b in zip(gap_bins[:-1], gap_bins[1:])],
                     fontsize=7)
    ax[3].set_xlabel("largest gap (days)")
    ax[3].set_ylabel("wavelet recall")
    ax[3].legend(fontsize=7)
    ax[3].set_title("(d) wavelet (gap-filled input) vs gap length", fontsize=9)
    x = np.arange(len(an.PR_LABELS))
    for i, mth in enumerate(["acf_fft", "acf_fft_highpass", "pairwise_histogram", "lombscargle", "wavelet"]):
        t = pm[pm.method == mth].groupby("truth_bin", observed=True)["recall"].mean().reindex(an.PR_LABELS)
        ax[4].plot(x, t, marker="o", ms=3, color=f"C{i}", label=mth)
    ax[4].axvline(4.5, color="k", lw=0.8, ls=":")
    ax[4].set_xticks(x, an.PR_LABELS, rotation=60, fontsize=7)
    ax[4].set_xlabel("true P_lo / R (all classes)")
    ax[4].set_ylabel("recall")
    ax[4].legend(fontsize=7)
    ax[4].set_title("(e) recall vs P/R: the reach edge", fontsize=9)
    n = pm.drop_duplicates(["run", "star_id"]).groupby("coverage_class").size().reindex(classes)
    ax[5].bar(range(len(classes)), n.values)
    ax[5].set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax[5].set_title("stars per class", fontsize=9)
    fig.suptitle(f"{label}: Stage 1 per-method signatures by coverage (before run)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"mech_methods_{label}.png", dpi=130)
    plt.close(fig)
    rec.to_csv(HERE / "data" / f"method_recall_by_class_{label}.csv")
    return rec


if __name__ == "__main__":
    (HERE / "data").mkdir(exist_ok=True)
    print(pair_fraction_figure().to_string())
    ls_window_figure()
    sys.path.insert(0, str(HERE))
    for lab, runs in (("dm_assigned", ("dm_assigned",)),
                      ("synthetic", ("syn_single", "syn_consecutive-2", "syn_gapped-2", "syn_3-5", "syn_6-12"))):
        try:
            print(method_signatures(lab, runs).round(2).to_string())
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"skip {lab}: {exc}")
