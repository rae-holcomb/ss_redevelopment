"""T32 step 1 figures: tracked ACF-peak residual vs peak index n for three star groups."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "data/step1_peaks.csv")
d = d[d.label == "local"].copy()
d["r_win"] = (d.tracked - d.win_center) / d.P0
d["r_line"] = (d.tracked - (d.t0_fit + d.n * d.P_fit)) / d.P_fit
G = [("gain_vertex", "(a) local's gains\n(lsq declined on vertex gate)", "tab:blue"),
     ("none_to_wrong", "(b) none -> wrong under local", "tab:red"),
     ("shared_correct", "(c) shared correct (sample of 60)", "tab:green")]


def per_star(g):
    """Per-star residual scatter about the fitted comb line (RMS of r_line over teeth n>=1)."""
    x = g[g.n >= 1].r_line.dropna()
    return np.sqrt(np.mean(x**2)) if len(x) else np.nan


fig, ax = plt.subplots(3, 3, figsize=(13, 10), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
stats = {}
for j, (grp, title, col) in enumerate(G):
    x = d[d.group == grp]
    for (sid, mode), g in x.groupby(["star_id", "mode"]):
        g = g.sort_values("n")
        ax[0, j].plot(g.n, g.r_win, "-o", color=col, alpha=0.35, ms=3, lw=1)
        ax[1, j].plot(g.n, g.r_line, "-o", color=col, alpha=0.35, ms=3, lw=1)
    for k in (0, 1):
        med = x.groupby("n")[["r_win", "r_line"][k]].apply(lambda s: np.nanmedian(np.abs(s)))
        ax[k, j].plot(med.index, med.values, "k-", lw=2.5, label="median |residual|")
    ax[0, j].axhspan(-0.25, 0.25, color="0.9", zorder=-1)
    ax[0, j].axhline(0.25, color="0.4", ls="--", lw=1); ax[0, j].axhline(-0.25, color="0.4", ls="--", lw=1)
    ax[1, j].axhline(0.05, color="0.4", ls=":", lw=1); ax[1, j].axhline(-0.05, color="0.4", ls=":", lw=1)
    ax[0, j].set_title(title + f"  [{x.groupby(['star_id','mode']).ngroups} stars]", fontsize=10)
    ax[0, j].set_ylim(-0.6, 0.6); ax[1, j].set_ylim(-0.3, 0.3)
    ax[1, j].set_xlabel("peak index n")
    stats[grp] = x.groupby(["star_id", "mode"]).apply(per_star, include_groups=False).dropna()
ax[0, 0].set_ylabel("(tracked peak - frozen window centre t0+n P0) / P0\n(grey = window, dashed = vertex gate edge)")
ax[1, 0].set_ylabel("(tracked peak - fitted comb t0+n P) / P\n(dotted = local's +/-5% jitter clip)")
ax[0, 0].legend(fontsize=8)
gs = ax[2, 0].get_gridspec()
for a in ax[2, :]:
    a.remove()
axb = fig.add_subplot(gs[2, :])
bins = np.linspace(0, 0.2, 41)
for grp, title, col in G:
    axb.hist(stats[grp], bins=bins, histtype="step", lw=2, color=col, density=True,
             label=f"{title.splitlines()[0]}: median {np.median(stats[grp]):.3f}")
axb.set_xlabel("per-star RMS residual about the fitted comb, teeth n >= 1 (units of P)")
axb.set_ylabel("density"); axb.legend(fontsize=8)
fig.suptitle("T32 step 1: where do the ACF peaks sit? (local's selected candidate; peaks tracked sequentially in the raw ACF)")
fig.tight_layout()
fig.savefig(HERE / "figures/fig_step1_residual_vs_n.png", dpi=110)
pd.concat({k: v for k, v in stats.items()}).rename("rms_r_line").to_csv(HERE / "data/step1_rms_per_star.csv")
print({k: (len(v), round(float(np.median(v)), 4)) for k, v in stats.items()})
