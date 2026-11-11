"""Figure functions for the T5 report; the notebook calls these on the saved data/ tables."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
MODES = {"fy": "full-year", "asg": "assigned"}
C_OLD, C_NEW, C_TRUE = "#7f7f7f", "#1a9850", "#d62728"


def load_phase1() -> pd.DataFrame:
    """Phase-1 per-star table with detectability and pool-union columns."""
    d = pd.read_csv(DATA / "phase1_stage1.csv")
    d["detectable"] = d.snr_rot >= 5
    for v in ("old", "new"):
        d[f"{v}_pool"] = d[f"{v}_acf_fft"] | d[f"{v}_lombscargle"] | d.pairwise
    return d


def fig_diag(sid: str, save=True):
    """Light curve, ACF, ACF-FFT spectrum and LS periodogram for one star in both coverage modes,
    with the true period (red band = diffrot range), old top-10 (grey) and new top-10 (green) marked."""
    z = np.load(DATA / "examples.npz")
    P, plo, phi = z[f"{sid}|meta"]
    fig, ax = plt.subplots(2, 4, figsize=(19, 6.6))
    for row, mode in enumerate(("asg", "fy")):
        k = f"{sid}|{mode}"
        t, f = z[k + "|t"], z[k + "|f"]
        a = ax[row, 0]
        a.plot(t, f * 1e3, ",", color="k", alpha=0.4)
        a.set(xlabel="time [d]", ylabel="flux - 1 [ppt]", title=f"{MODES[mode]}: light curve, baseline {t[-1] - t[0]:.0f} d")
        lags, acf = z[k + "|lags"], z[k + "|acf"]
        a = ax[row, 1]
        a.plot(lags, acf, lw=0.4, color="C0")
        a.set(xlabel="lag [d]", ylabel="ACF", title=f"ACF (Stage-1 view, {lags[-1]:.0f} d)")
        ins = a.inset_axes([0.45, 0.5, 0.52, 0.45])
        m = lags < 8 * P
        ins.plot(lags[m], acf[m], lw=0.7, color="C0")
        for n in range(1, 8):
            ins.axvline(n * P, color=C_TRUE, lw=0.6, alpha=0.6)
        ins.set_title("first 8 P (red = n·P_true)", fontsize=8)
        ins.tick_params(labelsize=7)
        for j, (meth, lab) in enumerate((("acf_fft", "ACF FFT spectrum"), ("lombscargle", "Lomb-Scargle"))):
            a = ax[row, 2 + j]
            x, y = z[k + f"|{meth}|P"], z[k + f"|{meth}|pow"]
            a.loglog(x, y, lw=0.4, color="C0")
            a.axvspan(min(plo, 0.9 * P), max(phi, 1.1 * P), color=C_TRUE, alpha=0.25, lw=0)
            a.axvline(P, color=C_TRUE, lw=1)
            old, new = z[k + f"|{meth}|old"], z[k + f"|{meth}|new"]
            ytop = np.nanmax(y) * 3
            a.plot(old, np.full(len(old), ytop), "v", color=C_OLD, ms=7, label="old top-10 (power)")
            newonly = np.setdiff1d(new, old)
            a.plot(newonly, np.full(len(newonly), ytop * 2.2), "v", color=C_NEW, ms=7,
                   label="new picks (local SNR)")
            hit_old = np.any((old >= min(plo, 0.9 * P)) & (old <= max(phi, 1.1 * P)))
            hit_new = np.any((new >= min(plo, 0.9 * P)) & (new <= max(phi, 1.1 * P)))
            a.set(xlabel="period [d]", ylabel="power",
                  title=f"{lab}: true P in top-10 old {'yes' if hit_old else 'NO'} / new {'yes' if hit_new else 'NO'}")
            a.set_ylim(top=ytop * 6)
            if row == 0 and j == 0:
                a.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{sid}  P_true = {P:.4f} d (red line; band = diffrot range or ±10%)", y=1.0)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / f"fig_diag_{sid[-6:]}.png", dpi=110, bbox_inches="tight")
    return fig


def fig_rank(d: pd.DataFrame, save=True):
    """Rank (raw-power order) of the true-period peak for detectable <1 d stars, both modes,
    with the top-10 cutoff; plus number of spectral peaks vs Stage-1 lag range."""
    s = d[(d.eval_pbin == "<1d") & d.detectable]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    bins = np.logspace(0, 3, 31)
    for j, m in enumerate(("acf_fft", "lombscargle")):
        a = ax[j]
        for mode, c in (("asg", "C1"), ("fy", "C0")):
            x = s[s["mode"] == mode][f"rank_{m}"].fillna(999)
            a.hist(x, bins=bins, histtype="step", lw=2, color=c,
                   label=f"{MODES[mode]} (median {x.median():.0f}, n={len(x)})")
        a.axvline(10.5, color="k", ls="--", lw=1)
        a.text(11, a.get_ylim()[1] * 0.9, "top-10 cut", fontsize=9)
        a.set(xscale="log", xlabel="rank of true-period peak by raw power (999 = absent)", ylabel="detectable <1 d stars",
              title=f"{m}: where the true peak sits")
        a.legend(fontsize=9)
    a = ax[2]
    for mode, c in (("asg", "C1"), ("fy", "C0")):
        x = d[d["mode"] == mode]
        a.scatter(x.stage1_lag_range, x.n_peaks_lombscargle, s=5, color=c, alpha=0.5, label=f"LS, {MODES[mode]}")
        a.scatter(x.stage1_lag_range, x.n_peaks_acf_fft, s=5, color=c, marker="x", alpha=0.5, label=f"ACF FFT, {MODES[mode]}")
    a.set(xlabel="Stage-1 ACF lag range [d]", ylabel="spectral peaks competing for 10 slots", yscale="log",
          title="more baseline → more peaks")
    a.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_rank_true_peak.png", dpi=110, bbox_inches="tight")
    return fig


def fig_stage1_before_after(d: pd.DataFrame, save=True):
    """Stage-1 recall (true period in the method's top-10) before/after, per method and coverage mode,
    for <1 d and 1-3 d (all stars and detectable), with the shuffled-truth chance level."""
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
    cols = [("acf_fft", "acf_fft"), ("lombscargle", "lombscargle"), ("pool", "3-method pool (union)")]
    for r, sub_name in enumerate(("all stars", "detectable (snr_rot ≥ 5)")):
        sub = d if r == 0 else d[d.detectable]
        for c, b in enumerate(("<1d", "1-3d")):
            for h, mode in enumerate(("fy", "asg")):
                a = ax[r, 2 * c + h]
                g = sub[(sub.eval_pbin == b) & (sub["mode"] == mode)]
                x = np.arange(3)
                old = [g[f"old_{k}"].mean() for k, _ in cols]
                new = [g[f"new_{k}"].mean() for k, _ in cols]
                a.bar(x - 0.2, old, 0.4, color=C_OLD, label="before (raw power)")
                a.bar(x + 0.2, new, 0.4, color=C_NEW, label="after (hybrid)")
                for i in range(2):
                    k = cols[i][0]
                    a.plot([i - 0.4, i], [g[f"chance_old_{k}"].mean()] * 2, color="k", lw=2)
                    a.plot([i, i + 0.4], [g[f"chance_new_{k}"].mean()] * 2, color="k", lw=2)
                for i, (o, n) in enumerate(zip(old, new)):
                    a.text(i - 0.2, o + 0.01, f"{o:.2f}", ha="center", fontsize=8)
                    a.text(i + 0.2, n + 0.01, f"{n:.2f}", ha="center", fontsize=8)
                a.set_xticks(x, ["acf_fft", "LS", "pool"])
                a.set_title(f"{b}, {MODES[mode]}, {sub_name} (n={len(g)})", fontsize=9)
                a.set_ylim(0, 1.08)
                if c == 0 and h == 0:
                    a.set_ylabel("Stage-1 recall")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    ax[1, 0].text(0.02, 0.02, "black ticks = shuffled-truth chance level", transform=ax[1, 0].transAxes, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_stage1_before_after.png", dpi=110, bbox_inches="tight")
    return fig


def fig_stage1_all_bins(d: pd.DataFrame, save=True):
    """Pool-union Stage-1 recall in every period bin, before/after, both modes (regression check)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for a, mode in zip(ax, ("fy", "asg")):
        for k, sub, ls in (("all", d, "-"), ("detectable", d[d.detectable], "--")):
            g = sub[sub["mode"] == mode].groupby("eval_pbin")[["old_pool", "new_pool"]].mean().reindex(BINS)
            a.plot(BINS, g.old_pool, ls, marker="o", color=C_OLD, label=f"before, {k}")
            a.plot(BINS, g.new_pool, ls, marker="o", color=C_NEW, label=f"after, {k}")
        a.set(title=f"3-method pool Stage-1 recall, {MODES[mode]}", xlabel="period bin", ylim=(0, 1.02))
    ax[0].set_ylabel("true period among raw guesses")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_stage1_all_bins.png", dpi=110, bbox_inches="tight")
    return fig


def fig_scorecard_bins(save=True):
    """Final outcome fractions (success / wrong / alias / declined) by period bin, before vs after,
    both coverage modes, plus the fitted-candidate ceiling (all and detectable)."""
    sb = pd.read_csv(DATA / "scorecard_bins.csv")
    fc = pd.read_csv(DATA / "fitted_ceiling.csv")
    order = BINS + ["all"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for j, mode in enumerate(("fy", "asg")):
        a = ax[0, j]
        x = np.arange(len(order))
        for k, (when, off) in enumerate((("before", -0.2), ("after", 0.2))):
            g = sb[(sb["mode"] == mode) & (sb.when == when)].set_index("bin").reindex(order)
            bottom = np.zeros(len(order))
            for col, c, lab in (("success", C_NEW, "success"), ("alias", "#fdae61", "alias"),
                                ("wrong", C_TRUE, "wrong"), ("none", "#d9d9d9", "declined")):
                a.bar(x + off, g[col], 0.38, bottom=bottom, color=c, edgecolor="k" if when == "after" else "none",
                      lw=0.6, label=lab if k == 1 else None)
                bottom += g[col].to_numpy()
            for i, v in enumerate(g.success):
                a.text(x[i] + off, v / 2, f"{v:.3f}", ha="center", va="center", fontsize=7, rotation=90)
        a.set_xticks(x, order)
        a.set(title=f"final outcome, {MODES[mode]} (left bar before, outlined bar after)", ylabel="fraction of stars")
        a.legend(fontsize=8, loc="upper right", ncol=4)
        a = ax[1, j]
        for name, ls in (("all", "-"), ("detectable", "--")):
            g = fc[fc["mode"] == mode]
            if name == "detectable":
                g = g[g.detectable]
            t = g.groupby(["when", "eval_pbin"]).fitted_success.mean().unstack().reindex(columns=BINS)
            a.plot(BINS, t.loc["before"], ls, marker="o", color=C_OLD, label=f"before, {name}")
            a.plot(BINS, t.loc["after"], ls, marker="o", color=C_NEW, label=f"after, {name}")
            for i, b in enumerate(BINS):
                if b == "<1d":
                    a.annotate(f"{t.loc['before', b]:.2f} → {t.loc['after', b]:.2f}", (i, t.loc["after", b]),
                               textcoords="offset points", xytext=(8, 4), fontsize=8, color=C_NEW)
        a.set(title=f"fitted-candidate ceiling (≥1 fitted success candidate), {MODES[mode]}", ylim=(0, 1.02),
              ylabel="fraction of stars")
        a.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_scorecard_bins.png", dpi=110, bbox_inches="tight")
    return fig


CAND = {"base+highpass": ("acf_fft_highpass", "C3"), "base+acf_short": ("acf_fft_short", "C4"),
        "base+ls_short": ("lombscargle_short", "C9")}
SUCCESS = {"correct", "diffrot"}


def phase2_summary():
    """Per mode x candidate method: marginal recall (fitted, after cap; and raw) per bin, added seconds,
    extra wrong fitted candidates, final success/wrong change, and FP rate on non-rotators."""
    rot = pd.read_csv(DATA / "phase2_rot.csv.gz")
    snr = pd.read_csv(DATA / "t26_snr_rot.csv").set_index("star_id")
    rows = []
    for mode, g in rot.groupby("mode"):
        w = g.pivot_table(index="star_id", columns="config",
                          values=["fitted_success", "n_fitted_wrong", "fit_seconds", "added_gen_seconds"],
                          aggfunc="first")
        fin = g.pivot(index="star_id", columns="config", values="final_outcome")
        first = g.drop_duplicates("star_id").set_index("star_id")
        pbin = first.eval_pbin
        rawbase = first.raw_pairwise_histogram | first.raw_lombscargle | first.raw_acf_fft
        det = snr[f"snr_rot_{mode}"].reindex(w.index) >= 5
        for c, (m, _) in CAND.items():
            gain = w.fitted_success[c].astype(bool) & ~w.fitted_success["base"].astype(bool)
            loss = ~w.fitted_success[c].astype(bool) & w.fitted_success["base"].astype(bool)
            rawgain = first[f"raw_{m}"] & ~rawbase
            dsec = w.added_gen_seconds[c] + w.fit_seconds[c] - w.fit_seconds["base"]
            row = dict(mode=mode, config=c, method=m, added_seconds_median=dsec.median(),
                       added_seconds_mean=dsec.mean(), base_fit_seconds_median=w.fit_seconds["base"].median(),
                       extra_wrong_fitted=(w.n_fitted_wrong[c] - w.n_fitted_wrong["base"]).mean(),
                       d_success=(fin[c].isin(SUCCESS).mean() - fin["base"].isin(SUCCESS).mean()),
                       d_wrong=((fin[c] == "wrong").mean() - (fin["base"] == "wrong").mean()))
            for b in BINS:
                sel = pbin == b
                row[f"gain_{b}"] = gain[sel].mean()
                row[f"loss_{b}"] = loss[sel].mean()
                row[f"gain_det_{b}"] = gain[sel & det].mean()
                row[f"rawgain_{b}"] = rawgain[sel].mean()
                row[f"dsucc_{b}"] = fin[c][sel].isin(SUCCESS).mean() - fin["base"][sel].isin(SUCCESS).mean()
            rows.append(row)
    s = pd.DataFrame(rows)
    neg = pd.read_csv(DATA / "phase2_neg.csv.gz")
    neg["answered"] = neg.final_outcome == "answered"
    fp = neg.groupby(["config", "full"]).answered.mean().unstack()
    fp.columns = ["fp_assigned_draws", "fp_fullyear"]
    fp["fp_all"] = neg.groupby("config").answered.mean()
    fp["n_fitted_mean"] = neg.groupby("config").n_fitted.mean()
    fp["fit_seconds_median"] = neg.groupby("config").fit_seconds.median()
    return s, fp


def fig_tradeoff(s, save=True):
    """Marginal <1 d and 1-3 d recall (fitted candidates, after the short-period cap) vs added seconds per star."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, mode in zip(ax, ("fy", "asg")):
        g = s[s["mode"] == mode]
        for _, r in g.iterrows():
            col = CAND[r.config][1]
            for b, mk in (("<1d", "o"), ("1-3d", "s")):
                a.scatter(r.added_seconds_median, r[f"gain_{b}"], s=90, marker=mk, color=col,
                          edgecolor="k", zorder=3)
                a.scatter(r.added_seconds_median, r[f"rawgain_{b}"], s=50, marker=mk, facecolor="none",
                          edgecolor=col, zorder=3)
        a.axhline(0, color="k", lw=0.6)
        a.set(title=f"{MODES[mode]}: gain over the fixed 3-method pool", xlabel="added wall-clock per star [s] (median, i5, 1 thread)")
        a.text(0.35, 0.95, "filled = fitted candidates (after cap)\nopen = raw Stage-1 guesses\n● <1 d   ■ 1-3 d",
               transform=a.transAxes, va="top", fontsize=8)
        for c, (m, col) in CAND.items():
            a.scatter([], [], color=col, s=60, label=m)
        a.legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("fraction of bin's stars newly having a success candidate")
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_tradeoff_recall_vs_compute.png", dpi=110, bbox_inches="tight")
    return fig


def fig_fp(s, fp, save=True):
    """Left: answer (FP) rate on dev_medium non-rotators per pool config. Right: extra wrong fitted
    candidates per rotator star from each added method."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    cfg = ["base_old", "base"] + list(CAND)
    f = fp.reindex(cfg)
    x = np.arange(len(cfg))
    ax[0].bar(x - 0.2, f.fp_assigned_draws, 0.4, color="#ef9a9a", label="assigned-coverage draws (3 per source)")
    ax[0].bar(x + 0.2, f.fp_fullyear, 0.4, color="#c62828", label="full-year")
    for i, (a_, b_) in enumerate(zip(f.fp_assigned_draws, f.fp_fullyear)):
        ax[0].text(i - 0.2, a_ + 0.001, f"{a_:.3f}", ha="center", fontsize=7)
        ax[0].text(i + 0.2, b_ + 0.001, f"{b_:.3f}", ha="center", fontsize=7)
    ax[0].set_xticks(x, ["before\n(raw power)", "after fix\n(hybrid)", "+highpass", "+acf_short", "+ls_short"])
    ax[0].set(ylabel="fraction of non-rotators given a period", title="hand-arbitration FP rate, dev_medium negatives")
    ax[0].legend(fontsize=8)
    for k, mode in enumerate(("fy", "asg")):
        g = s[s["mode"] == mode].set_index("config").reindex(list(CAND))
        ax[1].bar(np.arange(3) + (k - 0.5) * 0.4, g.extra_wrong_fitted, 0.4, color=["C0", "C1"][k],
                  label=MODES[mode])
    ax[1].set_xticks(np.arange(3), ["+highpass", "+acf_short", "+ls_short"])
    ax[1].set(ylabel="extra wrong fitted candidates per star", title="rotators: wrong candidates each method adds")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_fp_and_wrong_candidates.png", dpi=110, bbox_inches="tight")
    return fig


def fig_gates(save=True):
    """Fraction of fitted success candidates failing each comb-fit gate, by period bin (after the fix)."""
    g = pd.read_csv(DATA / "gates_success_candidates.csv")
    gates = ["frac_positive_heights", "mean_height_snr", "frac_vertex_in_window", "half_lag_consistency"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for a, mode in zip(ax, ("fy", "asg")):
        t = g[g["mode"] == mode].set_index("bin").reindex(BINS)
        for i, gate in enumerate(gates):
            a.bar(np.arange(5) + (i - 1.5) * 0.2, t[gate], 0.2, label=gate)
        a.plot(np.arange(5), t.passed, "k*", ms=10, label="passed all gates")
        a.set_xticks(np.arange(5), BINS)
        a.set(title=f"{MODES[mode]}: why fitted success candidates are rejected", ylim=(0, 1))
    ax[0].set_ylabel("fraction of success candidates failing the gate")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save:
        fig.savefig(FIG / "fig_gates_success_candidates.png", dpi=110, bbox_inches="tight")
    return fig
